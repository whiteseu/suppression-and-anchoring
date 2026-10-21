# Suppression and Anchoring — core code

Reference code for *Suppression and Anchoring: Decomposing Context-Contamination
Failure in Reasoning Models* (ICASSP 2027).

| file | what it defines |
|---|---|
| `verifier.py` | answer extraction and equality; the pre-answer length used for the suppression rate |
| `scoring.py` | the three-way outcome partition, the suppression rate, the paired bootstrap / McNemar test, and the zero-signal group count |
| `build_dataset.py` | the RA-GRPO intervention: how contaminated rollout states are constructed and mixed |
| `replay_groups.py` | how the reward signal reaching the learner is counted, per provenance stratum |
| `build_asr_bank.py` | the ASR-style surface corruption of Sec. 4.5 |

`verifier.py`, `build_dataset.py`, `replay_groups.py` and `build_asr_bank.py`
are the project's own source files, unmodified. `scoring.py` collects the four
scoring functions from the analysis pipeline, copied verbatim.

## The definitions, in short

**Outcome partition.** Every response is exactly one of three, scored by the
same verifier differing only in its target:

```
correct   check_answer(o, g)      solves the problem
anchored  check_answer(o, w)      returns the answer the prefix supplied
other     a different error
```

`correct` and `anchored` are mutually exclusive because every admitted wrong
attempt `w` satisfies `check_answer(w, g) == False`.

**Suppression is not a fourth outcome.** It is an orthogonal measurement over
the same responses — the share whose text before the boxed answer is shorter
than 40 characters. A suppressed response is still scored `C`/`A`/`O`, so
suppression and anchoring overlap: on the difficulty-gated set the base model
is 94.4% anchored and 57.7% suppressed.

**Contaminated rollout states.** A contaminated state is just a different
prompt, so GRPO gives each one its own group baseline and no trainer change is
needed. Three conditions are mixed 0.5 / 0.25 / 0.25:

```
clean       user: problem
unflagged   user: problem + the wrong attempt as a reference
selfprefix  user: problem; assistant: truncated wrong attempt (continued)
```

**Zero signal.** GRPO normalises within each prompt group,
`A_i = (r_i - mean(R)) / (std(R) + eps)` with `eps = 1e-4`, so a group whose
samples all earn the same reward has numerator exactly zero and contributes no
gradient. Under a binary correctness reward that is the share of groups with no
correct sample — the quantity behind the paper's signal-bottleneck result.

## Provenance and license

Problems come from GSM8K (MIT) and MATH-500. Erroneous attempts were generated
with Qwen3-8B / Qwen3-4B (Apache-2.0) and Tulu-3-8B (built on Llama-3.1).
This code is released under the MIT License (see `LICENSE`).
