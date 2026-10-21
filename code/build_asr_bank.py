"""Build an ASR-corrupted copy of a wrong-attempt bank.

Why this exists. A deployed reasoning model often reads a *transcript* of an
earlier spoken turn, so the erroneous context it continues from is not clean
text: it carries recognition errors. This bank asks whether the two behaviours
survive that. The corruption is applied to the PROSE of the wrong attempt
only:

  - the final \\boxed{...} answer is never touched, so the anchoring target is
    the same number as in the clean-text bank and every manifest entry stays
    valid;
  - inline math ($...$, \\(...\\), digits) is never touched, so the arithmetic
    the model would re-derive is unchanged;
  - only word-level substitutions typical of read-speech recognition are made
    (homophones, number-word confusions, article/function-word deletion,
    compound splits).

The result is a prefix that says the same wrong thing in degraded English.
Comparing it against the clean-text prefix isolates the effect of transcript
noise on suppression and anchoring, and is a controlled contrast rather than a
new phenomenon: same problems, same wrong answers, same eligibility.

Usage:
  python build_asr_bank.py --wrong data/wrong_gsm8k.jsonl \
      --out data/wrong_gsm8k_asr.jsonl --rate 0.15 --seed 0
"""
import argparse
import json
import random
import re
from pathlib import Path

# Confusion pairs observed in read-speech ASR of arithmetic narration. Each is
# applied in one direction only, so the corruption never silently repairs text.
HOMOPHONES = {
    "two": "to", "too": "to", "four": "for", "eight": "ate", "one": "won",
    "ate": "eight", "sum": "some", "cent": "sent", "cents": "sends",
    "buy": "by", "bought": "brought", "whole": "hole", "week": "weak",
    "weeks": "weeks", "sale": "sail", "sales": "sails", "their": "there",
    "there": "their", "its": "it's", "then": "than", "than": "then",
    "piece": "peace", "pieces": "peaces", "hours": "ours", "our": "hour",
    "no": "know", "know": "no", "right": "write", "made": "maid",
    "plane": "plain", "read": "red", "seen": "scene", "so": "sew",
    "some": "sum", "won": "one", "for": "four", "to": "two",
}
DELETABLE = {"the", "a", "an", "of", "is", "are", "that", "and"}
SPLITS = {"into": "in to", "another": "an other", "someone": "some one",
          "everyone": "every one", "himself": "him self", "cannot": "can not",
          "altogether": "all together", "already": "all ready"}
# tokens we must never alter: math, markup, numerals
PROTECTED = re.compile(r"(\$[^$]*\$|\\\([^)]*\\\)|\\[a-zA-Z]+|\{[^{}]*\}|[\d.,%$]+)")


def corrupt_words(text, rng, rate):
    """Substitute/delete/split prose words to reach approximately `rate`.

    Eligible positions are collected first and then sampled, because only a
    fraction of tokens have a plausible recognition confusion: drawing an
    independent coin per token would cap the achievable rate at that fraction.
    """
    toks = re.split(r"(\s+)", text)
    eligible = []
    words = 0
    for i, tok in enumerate(toks):
        if not tok.strip() or PROTECTED.fullmatch(tok.strip()):
            continue
        core = tok.strip(".,;:!?()")
        if not core.isalpha():
            continue
        words += 1
        low = core.lower()
        if low in SPLITS or low in HOMOPHONES or low in DELETABLE:
            eligible.append(i)
    if not words:
        return text, 0, 0
    k = min(len(eligible), round(rate * words))
    chosen = set(rng.sample(eligible, k)) if k else set()
    out = []
    for i, tok in enumerate(toks):
        if i not in chosen:
            out.append(tok)
            continue
        core = tok.strip(".,;:!?()")
        lead = tok[: len(tok) - len(tok.lstrip(".,;:!?()"))]
        trail = tok[len(tok.rstrip(".,;:!?()")):]
        low = core.lower()
        if low in SPLITS:
            new = SPLITS[low]
        elif low in HOMOPHONES:
            new = HOMOPHONES[low]
        else:
            continue  # deletion of a function word
        if core[0].isupper():
            new = new[0].upper() + new[1:]
        out.append(lead + new + trail)
    return "".join(out), len(chosen), words


def corrupt_attempt(w, rng, rate):
    """Corrupt everything before the final \\boxed{...}; leave the answer intact."""
    cut = w.rfind("\\boxed")
    if cut < 0:
        cut = len(w)
    head, tail = w[:cut], w[cut:]
    new_head, edits, words = corrupt_words(head, rng, rate)
    return new_head + tail, edits, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrong", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in
            Path(args.wrong).read_text(encoding="utf-8").splitlines() if l.strip()]
    n_edits = n_words = n_changed = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("wrong"):
                new, e, wds = corrupt_attempt(r["wrong"], rng, args.rate)
                # the anchoring target must be untouched
                b_old = r["wrong"].rfind("\\boxed")
                assert new[new.rfind("\\boxed"):] == r["wrong"][b_old:], \
                    f"{r['id']}: boxed answer was altered"
                n_edits += e
                n_words += wds
                n_changed += new != r["wrong"]
                r = dict(r, wrong=new, wrong_source_text="asr-corrupted")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert n_words > 0, "no prose words seen; wrong bank empty or all protected"
    print(f"wrote {args.out}: {len(rows)} rows, {n_changed} corrupted, "
          f"word-level edit rate {n_edits / n_words:.3f} over {n_words} prose words")


if __name__ == "__main__":
    main()
