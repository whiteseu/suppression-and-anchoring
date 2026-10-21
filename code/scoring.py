"""The outcome partition, the suppression rate, and the paired test.

These four functions are copied verbatim from the paper's analysis pipeline
(facts.py). They are the operational definitions behind every number in the
paper.

Definitions, for a problem x with gold answer g and a verified-wrong
attempt w:

  outcome     each response is exactly one of three, scored by the SAME
              verifier differing only in its target --
                  correct   check_answer(o, g)
                  anchored  check_answer(o, w)
                  other     a different error
              C and A are mutually exclusive because every admitted w
              satisfies check_answer(w, g) == False.

  suppression an ORTHOGONAL measurement over the same responses, not a
              fourth outcome: the share whose pre-answer text is shorter
              than 40 characters. A suppressed response is still scored
              C / A / O, so suppression and anchoring overlap (on the
              difficulty-gated set the base model is 94.4% anchored and
              57.7% suppressed).
"""
import random
from math import comb

from verifier import answers_match, reasoning_len

SUPPRESSION_CHARS = 40


def outcome(response, correct, contaminated_answer):
    """'C' | 'A' | 'O' for one response.

    `correct` is check_answer(response, gold), stored at generation time.
    """
    a = answers_match(response, contaminated_answer)
    assert not (correct and a), (
        "response counted BOTH correct and anchored -- w is not verifiably "
        "wrong. This invariant is what caught 56 correct solutions sitting "
        "in a bank as 'wrong attempts'.")
    return "C" if correct else ("A" if a else "O")


def cell(responses):
    """The three-way partition plus the suppression rate for one condition.

    `responses` is a list of (response_text, correct, contaminated_answer).
    Returns percentages; C + A + O == 100 by construction, while
    suppression is measured independently and does not enter the sum.
    """
    n = len(responses)
    o = [outcome(*r) for r in responses]
    c = sum(1 for v in o if v == "C")
    a = sum(1 for v in o if v == "A")
    other = n - c - a
    assert c + a + other == n

    lens = [reasoning_len(r[0]) for r in responses]
    supp = sum(1 for x in lens if x < SUPPRESSION_CHARS)

    return {"n": n,
            "C": 100.0 * c / n,          # accuracy
            "A": 100.0 * a / n,          # anchor rate
            "O": 100.0 * other / n,      # a different error
            "suppression": 100.0 * supp / n}


def paired(outcomes_a, outcomes_b, cat="C", n_boot=20000, seed=0):
    """Paired difference in the share of `cat`, with bootstrap CI and McNemar.

    `outcomes_a` / `outcomes_b` are dicts {problem_id: 'C'|'A'|'O'} over the
    SAME problems, from two conditions evaluated in a single run (batched
    greedy decoding is not deterministic across runs: 1.9-3.2% of items
    flip, which is why comparisons are made within one run).
    """
    ids = sorted(set(outcomes_a) & set(outcomes_b))
    assert len(ids) == len(outcomes_a) == len(outcomes_b), \
        "the two conditions do not cover the same problems"

    d = [(1 if outcomes_b[i] == cat else 0) - (1 if outcomes_a[i] == cat else 0)
         for i in ids]
    rng = random.Random(seed)
    bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))

    x = sum(1 for i in ids if outcomes_b[i] == cat and outcomes_a[i] != cat)
    y = sum(1 for i in ids if outcomes_a[i] == cat and outcomes_b[i] != cat)
    m = x + y
    p = 1.0 if m == 0 else min(
        1.0, 2 * sum(comb(m, k) for k in range(min(x, y) + 1)) / 2 ** m)

    lo = int(0.025 * n_boot)          # 499 at n_boot=20000
    hi = int(0.975 * n_boot) - 1      # 19499
    return {"n": len(ids),
            "delta": 100.0 * sum(d) / len(d),
            "ci_lo": 100.0 * bs[lo], "ci_hi": 100.0 * bs[hi],
            "gain": x, "loss": y, "p": p}


def zero_signal_share(groups):
    """Share of GRPO prompt groups that produce no learning signal.

    `groups` is a list of per-group reward lists. GRPO normalises within
    each prompt group,

        A_i = (r_i - mean(R)) / (std(R) + eps),   eps = 1e-4,

    so a group whose samples all earn the same reward has numerator exactly
    zero and contributes no gradient. With a binary correctness reward this
    is the share of groups containing no correct sample -- the quantity
    plotted in Fig. 2 (88.3% of base-policy groups on model-natural
    erroneous prefixes, against 49.7% on injected ones).
    """
    return 100.0 * sum(1 for R in groups if len(set(R)) == 1) / len(groups)
