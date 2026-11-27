"""Tests for the analysis step: human corrections are applied, the published
results reproduce exactly, and a single-model results file doesn't crash."""

import csv
from pathlib import Path

import pandas as pd

import analyze_results as ar

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["model_label", "question_id", "domain", "constraint_score", "perfect_rate_flag",
          "contains_factual_error", "explanation", "error", "human_corrected_score", "correction_note"]


def run_analysis(monkeypatch, scored, out):
    monkeypatch.setattr(ar, "SCORED_PATH", scored)
    monkeypatch.setattr(ar, "SUMMARY_PATH", out)
    ar.main()
    return pd.read_csv(out)


def test_published_results_reproduce_exactly(tmp_path, monkeypatch):
    out = run_analysis(monkeypatch, ROOT / "results" / "scored_responses.csv", tmp_path / "summary.csv")
    published = pd.read_csv(ROOT / "results" / "summary_metrics.csv")
    pd.testing.assert_frame_equal(out, published)


def test_human_correction_overrides_judge_score(tmp_path, monkeypatch):
    scored = tmp_path / "scored.csv"
    with open(scored, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow({"model_label": "Gemini-3.1-Flash-Lite", "question_id": "H12", "domain": "Healthcare",
                    "constraint_score": 6, "perfect_rate_flag": 0, "human_corrected_score": 10,
                    "correction_note": "judge inconsistency"})
        w.writerow({"model_label": "Gemini-3.1-Flash-Lite", "question_id": "H11", "domain": "Healthcare",
                    "constraint_score": 10, "perfect_rate_flag": 1})
    out = run_analysis(monkeypatch, scored, tmp_path / "summary.csv")
    row = out.iloc[0]
    assert row["perfect_rate"] == 1.0
    assert row["constraint_score"] == 10.0


def test_single_model_file_does_not_crash(tmp_path, monkeypatch):
    full = pd.read_csv(ROOT / "results" / "scored_responses.csv")
    one = full[full["model_label"] == "GPT-4o-mini"]
    scored = tmp_path / "one_model.csv"
    one.to_csv(scored, index=False)
    out = run_analysis(monkeypatch, scored, tmp_path / "summary.csv")
    assert set(out["model_label"]) == {"GPT-4o-mini"}
    assert len(out) == 4  # 2 domains x 2 tiers
