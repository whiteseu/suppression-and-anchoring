"""The one answer-equality function the whole paper is allowed to use.

extract_boxed() and check_answer() are copied VERBATIM from train/eval_ckpt.py,
which is what produced the stored `correct` field in every results/*.jsonl.
Copying rather than importing avoids pulling torch/transformers onto a laptop;
facts.py asserts that this copy reproduces the stored field on every row, so
any future drift between the two is caught immediately.

Correctness and anchoring are then defined symmetrically, against the same
verifier, differing only in the target:

    C(o) = check_answer(o, g)      the response solves the problem
    A(o) = check_answer(o, w)      the response returns the contaminated answer

which makes C and A mutually exclusive whenever the bank guarantees
check_answer(w, g) is False -- the invariant whose absence let 56 correct
solutions into the gated bank as "wrong attempts".
"""
import re

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
        return float(x.replace(",", "").replace("$", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return None


def check_answer(pred_text, gold):
    """True iff the final boxed answer of pred_text equals gold."""
    pred = extract_boxed(pred_text)
    if pred is None:
        return False
    pn, gn = _num(pred), _num(gold)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${gold}$", parsing_timeout=None),
                           parse(f"${pred}$", parsing_timeout=None), timeout_seconds=None))
    except Exception:
        return pred.strip() == gold.strip()


_CACHE = {}


def answers_match(pred_text, target_answer):
    """check_answer with memoisation on (final boxed answer, target)."""
    pred = extract_boxed(pred_text)
    if pred is None or target_answer is None:
        return False
    key = (pred, target_answer)
    if key not in _CACHE:
        _CACHE[key] = check_answer(pred_text, target_answer)
    return _CACHE[key]


def numeric_answer(s):
    """Read a GSM8K-style answer as a number, tolerating LaTeX currency and
    braces ('\\$9.00' -> 9.0). Returns None when the string is not a number,
    which for GSM8K means the attempt has no usable final answer."""
    if s is None:
        return None
    t = re.sub(r"\\(text|mathrm|mbox)\s*\{([^}]*)\}", r"\2", str(s))
    t = t.replace("\\", "").replace("{", "").replace("}", "")
    t = re.sub(r"[^0-9eE+\-./]", "", t.replace(",", ""))
    if t in ("", "-", "+", ".", "/"):
        return None
    try:
        if "/" in t:
            a, b = t.split("/")[:2]
            return round(float(a) / float(b), 6)
        return round(float(t), 6)
    except (ValueError, ZeroDivisionError):
        return None


def reasoning_len(text):
    """Characters of derivation before the final boxed answer."""
    t = text or ""
    if "\\boxed" in t:
        t = t[: t.rfind("\\boxed")]
        nl = t.rfind("\n")
        if nl > len(t) * 0.5:
            t = t[:nl]
    return len(t.rstrip())
