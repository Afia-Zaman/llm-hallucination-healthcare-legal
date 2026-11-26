# Do Fast-Tier LLMs Hallucinate on Healthcare and Legal Questions?

**Author:** Afia Zaman

This repo holds a small, fully reproducible experiment that measures hallucination in four widely used "fast" language models on healthcare and legal questions. Every response came from a real API call. The raw outputs, the judge's scores, any human corrections and the final metrics are all here, so anyone can check the numbers or rerun the study.

## Models tested

| Model | Provider |
|---|---|
| GPT-4o-mini | OpenAI |
| Claude Haiku 4.5 | Anthropic |
| Gemini 3.1 Flash-Lite | Google |
| DeepSeek-V4-Flash | DeepSeek |

## What the experiment does

The question set has 36 questions: 18 healthcare and 18 legal. They're split into two tiers.

- **Canonical (20 questions):** well-known facts and misconceptions, such as whether antibiotics treat the common cold or whether a landlord can change the locks without going to court.
- **Adversarial (16 questions):** precision questions with one exact right answer and plenty of room to make something up: FDA approval years, CYP enzyme names, clinical trial names, U.S. Code titles, dollar thresholds, and Supreme Court case names with their years.

Each question comes with weighted constraints that add up to 10 points. GPT-4o-mini, running at temperature 0, acts as the judge. It checks every response against the constraints and also flags any outright factual error. Before scoring, the pipeline confirms each response finished cleanly using the provider's own `finish_reason`, so a cut-off answer can't be scored as if it were complete.

The two metrics are:
- **Constraint Score (CS):** weighted points earned, out of 10
- **Perfect Rate (PR):** share of responses that scored a full 10

## Results

144 responses (36 questions × 4 models). All 144 finished cleanly and none hit the token limit.

| Model | Healthcare canonical | Healthcare adversarial | Legal canonical | Legal adversarial |
|---|---|---|---|---|
| GPT-4o-mini | 100% | 100% | 100% | 100% |
| Claude Haiku 4.5 | 90% | 100% | 100% | 100% |
| Gemini 3.1 Flash-Lite | 100% | 100% | 100% | 100% |
| DeepSeek-V4-Flash | 100% | 100% | 100% | 100% |

*Perfect Rate by model, domain and tier, after human review of the judge's scores. Full numbers are in `results/summary_metrics.csv`.*

**Main findings**

1. **No hallucinations at this scale.** The judge flagged zero factual errors across all 144 responses, including the adversarial tier, where inventing a plausible-sounding case name or year would be easy.
2. **The one imperfect score was an omission, not an error.** Claude Haiku 4.5 got 8/10 on H01 (first-line treatment for type 2 diabetes). It named metformin correctly but never mentioned lifestyle changes, which one constraint required.
3. **The judge made more mistakes than the models did.** The judge scored two Gemini responses (H12 and H14) at 6/10, yet its own written explanations called both answers accurate. A manual check confirmed both were correct, and they were corrected to 10/10. The corrections are kept in separate `human_corrected_score` and `correction_note` columns, so the judge's original scores are never overwritten.
4. **Neither domain nor tier made a meaningful difference.** The canonical vs. adversarial paired t-test (t = −1.00, p = 0.39) and the healthcare vs. legal test (t = −1.00, p = 0.39) both found nothing. With only four models these tests have very little power, so read them as exploratory.

**What this means**

These questions sit at the models' ceiling. That's a real result: single-answer factual questions like these no longer separate current fast-tier models. The fact that the judge got two scores wrong while the models got none also shows that LLM-as-judge pipelines need human review. It shouldn't be treated as an afterthought.

## Limitations

- Four models and 36 questions is pilot scale. The statistical tests are underpowered.
- Scoring relies on an LLM judge. All three non-perfect scores were checked by hand, but the perfect scores have not been sampled for human agreement (for example with Cohen's kappa).
- Each question was run once per model at temperature 0.3, so run-to-run variance isn't measured.
- The canonical tier has a ceiling effect by design. Harder question types such as false-premise questions and citation generation are the natural next step.

## Repository layout

```
├── data/
│   └── questions.json          # 36 questions with domain, tier and weighted constraints
├── src/
│   ├── run_experiment.py       # queries all four models, records finish_reason
│   ├── score_responses.py      # LLM-judge scoring against the constraints
│   └── analyze_results.py      # PR, CS, tier and domain comparisons, t-tests, Spearman
├── results/
│   ├── raw_responses.csv       # every model response, unedited
│   ├── scored_responses.csv    # judge scores + human corrections
│   └── summary_metrics.csv     # final metrics by model × domain × tier
├── docs/
│   └── SETUP.md                # detailed setup, cost and rerun notes
├── requirements.txt
└── .env.example
```

## Reproducing the results

You can recompute every metric from the saved data without any API keys:

```bash
pip install -r requirements.txt
python src/analyze_results.py
```

To run the whole experiment from scratch, which costs under $2 in API credits, see [docs/SETUP.md](docs/SETUP.md). The short version:

```bash
cp .env.example .env              # add your four API keys
python src/run_experiment.py      # writes results/raw_responses.csv
python src/score_responses.py     # writes results/scored_responses.csv
python src/analyze_results.py     # writes results/summary_metrics.csv
```

Every script can be rerun safely. Each one skips (model, question) pairs that already succeeded, so a partial failure only retries the rows that failed.

## License

Code is released under the MIT License. See [LICENSE](LICENSE).
