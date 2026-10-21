"""Build the contamination-augmented GRPO dataset.

Core idea: a "contaminated state" is just a different prompt. GRPO computes its
baseline per prompt group, so contaminated states automatically get their own
baseline — no trainer surgery needed. We emit a JSONL of prompt records; the
training script renders them with the tokenizer's chat template.

Record schema:
  {id, cond, gold, messages: [...], continue_final: bool}

Conditions per source problem (ratios via --mix, default 0.5 clean / 0.25 unflagged / 0.25 selfprefix):
  clean       user: problem
  unflagged   user: problem + neutral wrong attempt in context
  selfprefix  user: problem; assistant: truncated wrong attempt (continue_final=True)

Inputs:
  --problems  JSONL with {problem, gold, id}  (train split!)
  --wrong     JSONL wrong-attempt bank keyed by id ({id, wrong, wrong_source})
              Problems missing a wrong attempt only emit `clean`.

Usage:
  python build_dataset.py --problems data/gsm8k_train.jsonl --wrong data/wrong_gsm8k_train.jsonl \
      --out data/grpo_mix.jsonl --mix 0.5,0.25,0.25 --seed 0
"""
import argparse
import json
import random
from pathlib import Path

SOLVE = ("Solve the following problem step by step. "
         "Put ONLY the final answer in \\boxed{{}}.\n\nProblem: {problem}")

UNFLAGGED = ("For reference, here is a student's solution to a problem:\n\n"
             "--- student's solution ---\n{wrong}\n--- end ---\n\n"
             "Now solve the problem yourself, step by step. "
             "Put ONLY the final answer in \\boxed{{}}.\n\nProblem: {problem}")


def truncate_frac(text: str, frac: float = 0.6) -> str:
    cut = int(len(text) * frac)
    nl = text.rfind("\n", 0, cut)
    if nl > len(text) * 0.3:
        cut = nl
    out = text[:cut].rstrip()
    if "\\boxed" in out:
        out = out[: out.rfind("\\boxed")].rstrip()
    return out


def make_record(cond, row, wrong, rng):
    p = row["problem"]
    if cond == "clean":
        msgs = [{"role": "user", "content": SOLVE.format(problem=p)}]
        return dict(cond=cond, messages=msgs, continue_final=False)
    if cond == "unflagged":
        msgs = [{"role": "user", "content": UNFLAGGED.format(problem=p, wrong=wrong)}]
        return dict(cond=cond, messages=msgs, continue_final=False)
    if cond == "selfprefix":
        # Pilot finding (2026-07-26): shallow truncation often cuts the error out
        # (drag +1.3pp n.s. at frac=0.6 vs +9.3pp for full prefix). Skew deep and
        # include the answer-stripped full prefix half the time.
        if rng.random() < 0.5:
            prefix = truncate_frac(wrong, rng.uniform(0.7, 0.95))
        else:
            prefix = truncate_frac(wrong, 1.0)  # strips only the \boxed line
        msgs = [{"role": "user", "content": SOLVE.format(problem=p)},
                {"role": "assistant", "content": prefix}]
        return dict(cond=cond, messages=msgs, continue_final=True)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--wrong", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", default="0.5,0.25,0.25",
                    help="clean,unflagged,selfprefix ratios")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ratios = [float(x) for x in args.mix.split(",")]
    assert abs(sum(ratios) - 1.0) < 1e-6 and len(ratios) == 3

    wrong_by_id = {}
    for line in Path(args.wrong).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("wrong"):
            wrong_by_id[d["id"]] = d["wrong"]

    n_out, n_clean_only = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in Path(args.problems).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            wrong = wrong_by_id.get(row["id"])
            if wrong is None:
                cond = "clean"
                n_clean_only += 1
            else:
                cond = rng.choices(["clean", "unflagged", "selfprefix"], ratios)[0]
            rec = make_record(cond, row, wrong, rng)
            rec.update(id=row["id"], gold=row["gold"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"wrote {n_out} records ({n_clean_only} clean-only fallbacks) -> {args.out}")


if __name__ == "__main__":
    main()
