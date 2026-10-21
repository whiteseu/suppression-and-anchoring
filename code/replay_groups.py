"""Replay the GRPO rollout on contaminated training states and record, per
prompt group, how many of the G samples are rewarded.

Why this exists. With beta=0 and scale_rewards=True (grpo_recovery.py), a
prompt group whose G rewards are all equal has advantage 0 on every completion
and contributes nothing to the gradient. TRL logs that fraction as
frac_reward_zero_std, but only pooled over all prompt conditions and wrong
sources; it cannot say whether the zero-signal groups are the all-correct clean
prompts or the all-wrong own-error prefixes. This script samples G completions
per contaminated prompt with the training sampler (temperature 1.0, top_p 1.0)
and writes one record per group, keyed by condition and wrong_source, so the
zero-variance fraction can be reported per provenance stratum.

Two modes:
  --mode train   render selfprefix / unflagged records with build_dataset.make_record
                 for every wrong attempt in the training bank (capped per source),
                 rendered exactly as grpo_recovery.load_prompts renders them
                 (continue_final_message for selfprefix).
  --mode bank    the evaluation prefix condition (fullprefix, with the
                 continue-instruction user turn, as in eval_ckpt.build) on the
                 items of a manifest -- pass@k of escaping the anchor on the
                 gated bank.

Usage (train mode):
  python replay_groups.py --mode train --model models/Qwen3-8B [--adapter ckpts/v1_treat/checkpoint-400]
      --wrong data/wrong_gsm8k_train.jsonl --out results/replay_base_train.jsonl --G 8 --cap 300
Usage (bank mode):
  python replay_groups.py --mode bank --model models/Qwen3-8B [--adapter ...]
      --wrong data/wrong_gsm8k_selfnat_pool.jsonl --manifest data/manifest_gated.json
      --out results/replay_base_gated.jsonl --G 32
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import make_record  # noqa: E402
from eval_ckpt import build as eval_build  # noqa: E402

# --- reward: copied verbatim from grpo_recovery.answer_reward so the replay
# scores exactly what the trainer scored (importing it would pull in trl).
BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    starts = [m.end() for m in BOXED_RE.finditer(text or "")]
    if not starts:
        return None
    s, depth = starts[-1], 1
    for i in range(s, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[s:i]
    return text[s:]


def _num(x):
    try:
        return float(x.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def reward(comp, g):
    pred = extract_boxed(comp)
    if pred is None:
        return -0.1
    pn, gn = _num(pred), _num(g)
    if pn is not None and gn is not None:
        return 1.0 if abs(pn - gn) < 1e-6 else 0.0
    try:
        from math_verify import parse, verify
        gp = parse(f"${g}$", parsing_timeout=None)
        pp = parse(f"${pred}$", parsing_timeout=None)
        return 1.0 if verify(gp, pp, timeout_seconds=None) else 0.0
    except Exception:
        return 1.0 if pred.strip() == g.strip() else 0.0


def matches(comp, target):
    """Same equality as reward(), aimed at the contaminated answer instead of
    gold: 1.0 means the sample returned the answer the prefix supplied."""
    return reward(comp, target) == 1.0


def train_jobs(args, rng):
    # the wrong bank carries problem/gold/id itself (data/gsm8k_train.jsonl is
    # raw GSM8K with question/answer and no id), so it is the only input needed
    by_source = defaultdict(list)
    for line in Path(args.wrong).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("wrong") and d.get("problem") and d.get("gold") is not None:
            by_source[d.get("wrong_source", "?")].append(d)
    jobs = []
    for src, rows in sorted(by_source.items()):
        if src == "FAILED":
            continue
        rows = sorted(rows, key=lambda r: r["id"])
        if args.cap and len(rows) > args.cap:
            rows = rng.sample(rows, args.cap)
        for d in rows:
            for cond in args.conds:
                rec = make_record(cond, d, d["wrong"], rng)
                jobs.append(dict(id=d["id"], cond=cond, wrong_source=src, gold=d["gold"],
                                 wrong=d["wrong"], messages=rec["messages"],
                                 continue_final=rec["continue_final"]))
    return jobs


def bank_jobs(args):
    keep = None
    if args.manifest:
        # build_manifests.py output: {"eligible": [{"problem_id", ...}], ...}
        m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        keep = {r["problem_id"] for r in m["eligible"]}
    jobs = []
    for line in Path(args.wrong).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if not d.get("wrong") or (keep is not None and d["id"] not in keep):
            continue
        for cond in args.conds:
            msgs, _ = eval_build(cond, d)
            jobs.append(dict(id=d["id"], cond=cond, wrong_source=d.get("wrong_source", "?"),
                             gold=d["gold"], wrong=d["wrong"], messages=msgs,
                             continue_final=False))
    if keep is not None:
        assert len({j["id"] for j in jobs}) == len(keep), \
            f"manifest has {len(keep)} ids, bank matched {len({j['id'] for j in jobs})}"
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "bank"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--problems", default=None,
                    help="unused: the wrong bank carries problem/gold; accepted so "
                         "launchers written for the earlier signature still run")
    ap.add_argument("--wrong", required=True)
    ap.add_argument("--manifest", default=None, help="bank mode: restrict to these ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default=None)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--cap", type=int, default=300, help="train mode: max prompts per wrong_source")
    ap.add_argument("--batch", type=int, default=8, help="prompts per generate() (x G sequences)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.conds = ([c.strip() for c in args.conds.split(",")] if args.conds
                  else (["selfprefix", "unflagged"] if args.mode == "train" else ["fullprefix"]))
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    jobs = train_jobs(args, rng) if args.mode == "train" else bank_jobs(args)
    # the silent-empty-run guard: refuse to start on an implausibly small set
    assert len(jobs) >= 20, f"only {len(jobs)} prompts rendered; wrong inputs?"
    print("prompt groups:", len(jobs), "by source x cond:",
          dict(Counter((j["wrong_source"], j["cond"]) for j in jobs)), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for j in jobs:
        j["prompt"] = tok.apply_chat_template(
            j["messages"], tokenize=False,
            add_generation_prompt=not j["continue_final"],
            continue_final_message=j["continue_final"],
            enable_thinking=False)

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done.add((d["id"], d["cond"]))
        print(f"resuming: {len(done)} groups already written", flush=True)
    todo = [j for j in jobs if (j["id"], j["cond"]) not in done]

    def sample(chunk, G=None):
        """G samples for each prompt in chunk. On a shared card the prefill's
        SDPA score matrix (sequences x L^2 under a padding mask) can exceed
        what a co-tenant left free at that instant; a fixed batch then dies six
        times in a row on the same chunk. On OOM, halve the number of prompts
        sharing one generate() call, and once a single prompt is left, halve
        the number of samples drawn per call -- the samples are independent
        draws either way, so their distribution is unchanged. Only a single
        prompt with a single sample that still OOMs is a real failure."""
        G = args.G if G is None else G
        enc = tok([j["prompt"] for j in chunk], return_tensors="pt", padding=True).to("cuda")
        in_len = enc["input_ids"].shape[1]
        try:
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=G,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
        except torch.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if len(chunk) > 1:
                h = len(chunk) // 2
                print(f"[oom] splitting {len(chunk)} prompts -> {h}+{len(chunk) - h}", flush=True)
                return sample(chunk[:h], G) + sample(chunk[h:], G)
            if G > 1:
                h = G // 2
                print(f"[oom] splitting {G} samples -> {h}+{G - h} (prompt {chunk[0]['id']})", flush=True)
                a, b = sample(chunk, h), sample(chunk, G - h)
                return [a[0] + b[0]]
            raise
        # decode per prompt; pad rows are padded to the longest sequence
        texts = [tok.decode(s[in_len:], skip_special_tokens=True) for s in gen]
        return [texts[i * G:(i + 1) * G] for i in range(len(chunk))]

    with open(out_path, "a", encoding="utf-8") as out_f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            for j, texts in zip(chunk, sample(chunk)):
                rw = [reward(t, j["gold"]) for t in texts]
                # anchored := returned the contaminated answer (train mode has
                # the whole wrong attempt; its boxed answer is the target)
                anchored = [matches(t, extract_boxed(j["wrong"]) or "") for t in texts]
                rec = {"id": j["id"], "cond": j["cond"], "wrong_source": j["wrong_source"],
                       "gold": j["gold"], "rewards": rw, "n_correct": sum(r == 1.0 for r in rw),
                       "n_anchored": sum(anchored), "zero_std": len(set(rw)) == 1,
                       "responses": texts}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            torch.cuda.empty_cache()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}", flush=True)

    # summary per stratum
    agg = defaultdict(lambda: [0, 0, 0, 0])  # groups, zero_std, all_wrong, any_correct
    for line in out_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        a = agg[(d["wrong_source"], d["cond"])]
        a[0] += 1
        a[1] += d["zero_std"]
        a[2] += d["n_correct"] == 0
        a[3] += d["n_correct"] > 0
    print("stratum                       groups  zero_std  all_wrong  any_correct")
    for k in sorted(agg):
        n, z, w, c = agg[k]
        print(f"{k[0]:14s} {k[1]:12s} {n:6d}  {z / n:8.3f}  {w / n:9.3f}  {c / n:11.3f}")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
