"""
analyze_results.py

Computes the metrics your paper defines, from real scored data, broken down
by BOTH domain (Healthcare/Legal) and tier (canonical/adversarial):
  - Perfect Rate (PR) and Constraint Score (CS) per model x domain x tier
  - Canonical vs. adversarial comparison per model (the key new analysis --
    this is what actually tests whether hallucination shows up when the
    questions are designed to be harder to answer from memorized associations)
  - Domain Variance and Consistency Score per model (canonical tier, for
    comparability with the original paper's design)
  - Paired t-test comparing domains (matched by model)
  - Spearman correlation between model scale rank and performance rank

Applies any human corrections to the automated judge's scores (see the
human_corrected_score / correction_note columns documented in
score_responses.py) before computing anything else, and reports how many
corrections were applied so that's never silently invisible.

Run this AFTER score_responses.py. Prints a summary and writes
results/summary_metrics.csv.
"""

import json
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.json"
SCORED_PATH = ROOT / "results" / "scored_responses.csv"

# Rough capability/scale tier for RQ4 (edit if your model line-up differs).
# These four are all "small/fast" tier models within their own families,
# so this ranking is a rough proxy at best -- say so explicitly in the paper
# rather than treating it as a clean parameter-count comparison.
# Lower rank number = smaller/cheaper tier.
MODEL_SCALE_RANK = {
    "DeepSeek-V4-Flash": 1,
    "Gemini-3.1-Flash-Lite": 1,
    "GPT-4o-mini": 2,
    "Claude-Haiku-4.5": 2,
}


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    tier_by_id = {q["id"]: q["tier"] for q in questions}

    df = pd.read_csv(SCORED_PATH)
    df = df[df["error"].isna() | (df["error"] == "")]  # drop any rows that failed to score

    # Apply any human corrections to the judge's automated score. This column is
    # blank unless you manually reviewed a row and disagreed with the judge (see
    # score_responses.py). Doing this openly, rather than silently editing the
    # CSV, keeps the correction auditable and reproducible.
    if "human_corrected_score" in df.columns:
        has_correction = df["human_corrected_score"].notna() & (df["human_corrected_score"] != "")
        n_corrected = int(has_correction.sum())
        if n_corrected:
            print(f"NOTE: {n_corrected} row(s) have a human-corrected score overriding the automated judge:")
            for _, r in df[has_correction].iterrows():
                print(f"    {r['model_label']} / {r['question_id']}: judge={r['constraint_score']} -> "
                      f"human={r['human_corrected_score']} ({r.get('correction_note', '')})")
            df.loc[has_correction, "constraint_score"] = pd.to_numeric(df.loc[has_correction, "human_corrected_score"])
            df.loc[has_correction, "perfect_rate_flag"] = (df.loc[has_correction, "constraint_score"] == 10).astype(int)
            print()
    df["tier"] = df["question_id"].map(tier_by_id)

    # === Perfect Rate / Constraint Score by model x domain x tier ===
    grouped = df.groupby(["model_label", "domain", "tier"]).agg(
        perfect_rate=("perfect_rate_flag", "mean"),
        constraint_score=("constraint_score", "mean"),
        n=("constraint_score", "count"),
    ).reset_index()
    print("=== Perfect Rate / Constraint Score by model x domain x tier ===")
    print(grouped.to_string(index=False))

    # === THE KEY COMPARISON: canonical vs. adversarial, per model ===
    tier_summary = df.groupby(["model_label", "tier"]).agg(
        perfect_rate=("perfect_rate_flag", "mean"),
        constraint_score=("constraint_score", "mean"),
        n=("constraint_score", "count"),
    ).reset_index()
    tier_pivot = tier_summary.pivot(index="model_label", columns="tier", values="perfect_rate")
    print("\n=== Canonical vs. Adversarial Perfect Rate, per model ===")
    print(tier_pivot.to_string())

    if "canonical" in tier_pivot.columns and "adversarial" in tier_pivot.columns:
        gap = (tier_pivot["canonical"] - tier_pivot["adversarial"]).round(3)
        print("\nGap (canonical - adversarial), per model:")
        print(gap.to_string())
        if len(tier_pivot) >= 2:
            t_stat, p_val = stats.ttest_rel(tier_pivot["canonical"], tier_pivot["adversarial"])
            print(f"\nPaired t-test (canonical vs adversarial, n={len(tier_pivot)} models): t={t_stat:.3f}, p={p_val:.3f}")
            if len(tier_pivot) < 8:
                print("NOTE: n is small (few models) -- treat as suggestive, not conclusive.")

    # === Domain Variance & Consistency Score (canonical tier only, for
    #     comparability with the original paper's design) ===
    canonical = grouped[grouped["tier"] == "canonical"]
    pivot = canonical.pivot(index="model_label", columns="domain", values="perfect_rate")
    domains = list(pivot.columns)
    if len(domains) == 2:
        pivot["domain_variance"] = pivot[domains].std(axis=1, ddof=0)
        pivot["consistency_score"] = 1 - (pivot[domains].max(axis=1) - pivot[domains].min(axis=1)) / pivot[domains].max(axis=1)
        print("\n=== Domain Variance & Consistency Score per model (canonical tier) ===")
        print(pivot.to_string())

        d1_vals, d2_vals = pivot[domains[0]], pivot[domains[1]]
        t_stat, p_val = stats.ttest_rel(d1_vals, d2_vals)
        print(f"\n=== Paired t-test: {domains[0]} vs {domains[1]} (canonical tier, n={len(d1_vals)} models) ===")
        print(f"t = {t_stat:.3f}, p = {p_val:.3f}")
        if len(d1_vals) < 8:
            print("NOTE: with only a few models, this test has very low statistical power.")

    # === Domain Difficulty Index, split by tier ===
    ddi = grouped.groupby(["tier", "domain"])["perfect_rate"].mean().apply(lambda pr: 1 - pr)
    print("\n=== Domain Difficulty Index (avg hallucination rate), by tier ===")
    print(ddi.to_string())

    # === Spearman correlation: model scale rank vs overall performance rank ===
    overall_pr = df.groupby("model_label")["perfect_rate_flag"].mean()
    scale_ranks = overall_pr.index.map(lambda m: MODEL_SCALE_RANK.get(m))
    if scale_ranks.isnull().any():
        print("\n[!] Some models in your data aren't in MODEL_SCALE_RANK -- edit the dict at the top of this script.")
    else:
        rho, p_val = stats.spearmanr(scale_ranks, overall_pr.values)
        print(f"\n=== Spearman correlation: model scale vs overall performance (n={len(overall_pr)} models) ===")
        print(f"rho = {rho:.3f}, p = {p_val:.3f}")
        if len(overall_pr) < 8:
            print("NOTE: n is small here too -- report this as exploratory, not as strong evidence of a scale effect.")

    grouped.to_csv(ROOT / "results" / "summary_metrics.csv", index=False)
    print("\nSaved results/summary_metrics.csv")


if __name__ == "__main__":
    main()
