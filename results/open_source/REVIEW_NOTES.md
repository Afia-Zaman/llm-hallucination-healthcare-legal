# Manual Review: Qwen2.5-1.5B-Instruct

Run on a Google Colab T4 GPU with `src/run_open_source_model.py` (temperature 0.3, seed 42, 1,024 max new tokens), then scored with the GPT-4o-mini judge. Every answer the judge penalized or flagged as containing a factual error was read in full and checked against its constraints, as was done for the API models.

## Score corrections

Corrections are stored in `human_corrected_score` and `correction_note` in `scored_responses.csv`. The judge's original scores are kept.

| Question | Judge | Corrected | Reason |
|---|---|---|---|
| H12 | 6 | 10 | States 6.5% correctly. Same judge inconsistency seen on Gemini H12. |
| H18 | 4 | 0 | Confidently names folic acid instead of vitamin A. |
| L01 | 9 | 4 | Says most states require all-party consent (most are one-party), invents a "New York Telephone Communications Privacy Act", omits the federal baseline. |
| L10 | 10 | 5 | Says trademarks do not last forever and invents a 7-year renewal schedule. |

All other judge scores were confirmed on review.

## Errors the constraints did not catch

These answers met every constraint, so they keep a perfect score, but each contains a false claim elsewhere in the answer:

- **H04:** claims acetaminophen use in pregnancy is associated with birth defects (not established).
- **L07:** adds a "right to know who their attorney is" and says Miranda warnings are not required before formal charges.
- **L09:** says some states allow retrial after an acquittal.
- **L13:** places the Wiretap Act in Chapter 23 of Title 18 (it is Chapter 119).
- **L16:** says the federal limitations period can start from the victim's discovery of the crime.

## Excluded answer

**L17** never finished: after calling the statute "the Trademark Act of 1946", the model listed invented section titles until it hit the 1,024-token limit. Under the pipeline's rules a cut-off answer is not scored, so results are based on 35 of 36 answers.
