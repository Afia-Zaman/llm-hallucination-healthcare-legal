# Setup and Run Guide

Detailed notes for rerunning the experiment from scratch. If you only want to recompute the metrics from the saved data, you don't need any of this. Run `python src/analyze_results.py`.

## 1. Get API keys

| Provider | Sign up | Notes |
|---|---|---|
| OpenAI | platform.openai.com | Pay-as-you-go. Add ~$5 credit. GPT-4o-mini is also the judge model. |
| Anthropic | console.anthropic.com | Pay-as-you-go. Add ~$5 credit. |
| Google AI Studio | aistudio.google.com | Free API key, no billing needed at this scale. |
| DeepSeek Platform | platform.deepseek.com | Pay-as-you-go. A ~$2 top-up covers the whole run. |

All 36 questions × 4 models, plus judge scoring, costs well under $2.

Providers rename models fairly often. If a call fails with a 404 ("model not found"), check the provider's current docs (linked at the top of `src/run_experiment.py`) and update the `model_id` values in the `MODELS` list.

## 2. Install

```bash
pip install -r requirements.txt
cp .env.example .env
# open .env and paste in your four API keys
```

`.env` is listed in `.gitignore`, so your keys never get committed.

## 3. Run the pipeline in order

Run these from the repo root:

```bash
python src/run_experiment.py      # queries all 4 models -> results/raw_responses.csv
python src/score_responses.py     # judge scores each response -> results/scored_responses.csv
python src/analyze_results.py     # metrics, tier/domain comparisons, tests -> results/summary_metrics.csv
```

Rerunning is cheap:
- `run_experiment.py` only queries pairs that are missing, errored, or came back truncated. It checks the API's own `finish_reason` field to tell. Rows that are already good never get billed twice.
- `score_responses.py` only scores rows that are missing or failed before. It refuses to score a truncated raw response and tells you to rerun `run_experiment.py` first.
- Adding new questions to `data/questions.json` costs nothing for questions you've already run, because both scripts key off (model, question_id) pairs.

Read the summary each script prints at the end. It lists anything that needs attention, such as errors, truncation or scoring failures. If it flags rows, rerun the same script and it will retry just those.

## 4. Checking the judge

Automated scoring isn't ground truth. Read through the judge's `explanation` column in `results/scored_responses.csv` and compare it with the raw response. Check at least every non-perfect score.

If you disagree with a score, put the corrected value in `human_corrected_score` and your reasoning in `correction_note`. The scripts never overwrite those two columns. `analyze_results.py` applies the corrections and prints each one it used, so every change stays visible.

## 5. Interpreting results

- **n = 4 models is small.** `analyze_results.py` prints a warning on every t-test and correlation for this reason. Report them as exploratory.
- **The canonical tier is expected to hit the ceiling.** Near-perfect canonical scores mean those questions no longer separate frontier models. The adversarial tier and the gap between the two tiers are the comparisons to watch.
- **Model scale ranks are rough.** All four models are each provider's small, fast tier. The ranks in `MODEL_SCALE_RANK` are an approximation, not parameter counts.
