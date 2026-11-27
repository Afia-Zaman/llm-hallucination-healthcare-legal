# Bugs Found and Fixed

The first version of this pipeline produced numbers that looked plausible but were wrong. This file records each bug, how it showed up, and what fixed it. Every fix has a regression test in `tests/`, so the bug can't come back unnoticed.

## 1. Cut-off answers were scored as if they were complete

**Symptom.** In the pilot run, with a 900-token output limit, many answers stopped mid-sentence. The judge scored those partial answers as wrong. DeepSeek's answer to H09 (ear infection treatment) was cut off before it reached the main recommendation and scored 6/10, even though nothing in it was incorrect.

**Cause.** The pipeline saved whatever text came back and never checked why generation stopped.

**Fix.** Every row now records the provider's own stop signal (`finish_reason`: `stop`, `end_turn`, `STOP`). Anything else (`length`, `MAX_TOKENS`) is treated as truncated: the runner re-queries it on the next run, and the scorer refuses to score it. The output limit was raised to 2,200 tokens, and both scripts print an end-of-run summary listing any row that still needs attention.

**Tests.** `test_api_runner_only_counts_clean_finishes_as_done`, `test_scoring_skips_truncated_and_errored_rows`

## 2. Hidden "thinking" tokens used up the output budget

**Symptom.** In the early runs, Gemini and DeepSeek answers were cut off far more often than the other models' answers.

**Cause.** Both models can reason internally before answering, and those hidden tokens count against the output limit.

**Fix.** DeepSeek's thinking mode is switched off explicitly (`"thinking": {"type": "disabled"}`). For Gemini, the first attempted fix used the `thinkingBudget` parameter, which only applies to Gemini 2.5 models and is not valid for Gemini 3.1 Flash-Lite. That model uses a `thinkingLevel` setting that already defaults to minimal, so the parameter was removed. The real fix was the larger token limit plus the `finish_reason` check from bug 1.

## 3. The judge sometimes returned JSON wrapped in extra text

**Symptom.** Scoring failed with JSON parse errors on some rows.

**Cause.** Despite instructions, the judge model occasionally wrapped its JSON in markdown code fences or added commentary before or after it.

**Fix.** `extract_json()` strips code fences, falls back to the outermost `{...}` block, and raises a clear error if no JSON is found. Parse failures are retried before a row is marked as failed.

**Tests.** `test_extract_json_strips_code_fences`, `test_extract_json_ignores_surrounding_commentary`, `test_extract_json_raises_when_no_json`

## 4. The judge could return the wrong number of constraint results

**Symptom.** The judge occasionally returned a different number of constraint results than it was given, and the scorer accepted it.

**Cause.** Scores are computed by pairing each result with its constraint's weight. If the counts differ, weights silently attach to the wrong constraints.

**Fix.** The judge prompt states the exact count. The scorer rejects any response whose count doesn't match and retries it.

**Tests.** `test_judge_rejects_wrong_number_of_constraint_results`, `test_judge_accepts_matching_constraint_results`

## 5. A row went missing between runs

**Symptom.** After re-running to fix failed rows, Claude Haiku 4.5's answer to H04 was missing from the results.

**Cause.** The early re-run logic didn't reliably track which (model, question) pairs were already complete.

**Fix.** Both runners and the scorer key every row on its (model, question) pair. A re-run keeps every clean row, drops stale failed or truncated rows, and redoes only what is missing. Already-good rows are never re-billed.

**Tests.** `test_open_source_runner_retries_truncated_rows`, `test_rescoring_keeps_existing_scores_and_human_corrections`

## 6. Temporary API errors could stop a whole run

**Risk.** A single rate-limit (429) or server (5xx) error would have stopped a whole run.

**Fix.** Calls retry with exponential backoff (5s, 10s, 20s) on 429 and 5xx responses and on network timeouts. Errors that won't fix themselves, such as 401 (bad key) or 400 (bad request), fail immediately with the provider's message.

## 7. The judge's score contradicted its own explanation

**Symptom.** Two Gemini answers (H12, H14) received a structured score of 6/10, while the judge's written explanation called both fully accurate.

**Fix.** Every non-perfect score was reviewed by hand. Corrections go in separate `human_corrected_score` and `correction_note` columns, so the judge's original score is never overwritten. The scorer never touches those columns on re-runs, and the analysis applies each correction and prints it.

**Tests.** `test_rescoring_keeps_existing_scores_and_human_corrections`, `test_human_correction_overrides_judge_score`

## 8. The analysis printed meaningless statistics for a single model

**Symptom.** Found while adding the open-source model: running the analysis on a results file with one model printed `rho = nan, p = nan` as if it were a result.

**Fix.** Cross-model tests are skipped with a clear message when there are too few models to compare.

**Test.** `test_single_model_file_does_not_crash`

## Reproducibility check

`test_published_results_reproduce_exactly` recomputes every metric from `results/scored_responses.csv` and confirms it matches `results/summary_metrics.csv`. It runs on every push through GitHub Actions, both directly and inside the Docker image.
