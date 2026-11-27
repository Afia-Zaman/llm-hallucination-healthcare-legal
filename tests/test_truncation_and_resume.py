"""Regression tests for truncation and resume bugs.

In the pilot run, a 900-token limit cut off answers from two of the four
models, and those partial answers were scored as if they were complete. A
(model, question) row was also once dropped between runs. These tests pin down
the fixes: truncated rows are never treated as done or scored, re-runs only
redo what is missing, and human corrections are never overwritten.
"""

import csv

import run_experiment as rx
import run_open_source_model as osm
import score_responses as sr

RAW_FIELDS = ["model_label", "provider", "question_id", "domain", "question", "response", "finish_reason", "error"]


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def raw_row(model, provider, qid, finish, error="", response="An answer."):
    return {"model_label": model, "provider": provider, "question_id": qid, "domain": "Healthcare",
            "question": "q", "response": response, "finish_reason": finish, "error": error}


# ---- what counts as "done" -------------------------------------------------

def test_api_runner_only_counts_clean_finishes_as_done(tmp_path, monkeypatch):
    out = tmp_path / "raw.csv"
    write_csv(out, RAW_FIELDS, [
        raw_row("GPT-4o-mini", "openai", "H01", "stop"),
        raw_row("GPT-4o-mini", "openai", "H02", "length"),           # truncated
        raw_row("Gemini-3.1-Flash-Lite", "google", "H01", "STOP"),   # Gemini spells it in caps
        raw_row("Gemini-3.1-Flash-Lite", "google", "H02", "MAX_TOKENS"),
        raw_row("Claude-Haiku-4.5", "anthropic", "H01", "end_turn"),
        raw_row("Claude-Haiku-4.5", "anthropic", "H02", "", error="HTTP 529"),
    ])
    monkeypatch.setattr(rx, "OUTPUT_PATH", out)
    assert rx.load_completed_pairs() == {
        ("GPT-4o-mini", "H01"), ("Gemini-3.1-Flash-Lite", "H01"), ("Claude-Haiku-4.5", "H01"),
    }


def test_open_source_finish_reason_from_tokens():
    eos = {2}
    assert osm.finish_reason_from_tokens([5, 9, 2], eos, max_new_tokens=10) == "stop"
    assert osm.finish_reason_from_tokens([5] * 10, eos, max_new_tokens=10) == "length"
    assert osm.finish_reason_from_tokens([5, 6, 7], eos, max_new_tokens=10) == "stop"


def test_open_source_runner_retries_truncated_rows(tmp_path):
    out = tmp_path / "os_raw.csv"
    write_csv(out, RAW_FIELDS, [
        raw_row("Qwen", "huggingface", "H01", "stop"),
        raw_row("Qwen", "huggingface", "H02", "length"),
        raw_row("Qwen", "huggingface", "H03", "", error="CUDA out of memory"),
    ])
    assert osm.load_completed_pairs(out) == {("Qwen", "H01")}


# ---- scoring ----------------------------------------------------------------

def run_scoring(tmp_path, monkeypatch, raw_rows, existing_scored=None):
    raw, scored = tmp_path / "raw.csv", tmp_path / "scored.csv"
    write_csv(raw, RAW_FIELDS, raw_rows)
    fields = ["model_label", "question_id", "domain", "constraint_score", "perfect_rate_flag",
              "contains_factual_error", "explanation", "error", "human_corrected_score", "correction_note"]
    if existing_scored is not None:
        write_csv(scored, fields, existing_scored)
    calls = []

    def fake_judge(question, response, constraints):
        calls.append(response)
        return {"constraint_results": [{"satisfied": True} for _ in constraints],
                "contains_factual_error": False, "explanation": "ok"}

    monkeypatch.setattr(sr, "RAW_PATH", raw)
    monkeypatch.setattr(sr, "OUTPUT_PATH", scored)
    monkeypatch.setattr(sr, "call_judge_with_retries", fake_judge)
    sr.main()
    return read_csv(scored), calls


def test_scoring_skips_truncated_and_errored_rows(tmp_path, monkeypatch):
    rows, calls = run_scoring(tmp_path, monkeypatch, [
        raw_row("GPT-4o-mini", "openai", "H01", "stop", response="complete"),
        raw_row("GPT-4o-mini", "openai", "H02", "length", response="cut off mid-sent"),
        raw_row("GPT-4o-mini", "openai", "H03", "", error="HTTP 500"),
        raw_row("Qwen", "huggingface", "H04", "stop", response="local model answer"),
    ])
    assert calls == ["complete", "local model answer"]
    assert {(r["model_label"], r["question_id"]) for r in rows} == {("GPT-4o-mini", "H01"), ("Qwen", "H04")}
    assert all(r["constraint_score"] == "10.0" for r in rows)


def test_rescoring_keeps_existing_scores_and_human_corrections(tmp_path, monkeypatch):
    existing = [{"model_label": "Gemini-3.1-Flash-Lite", "question_id": "H12", "domain": "Healthcare",
                 "constraint_score": "6.0", "perfect_rate_flag": "0", "contains_factual_error": "False",
                 "explanation": "accurate", "error": "", "human_corrected_score": "10",
                 "correction_note": "judge inconsistency"}]
    rows, calls = run_scoring(tmp_path, monkeypatch, [
        raw_row("Gemini-3.1-Flash-Lite", "google", "H12", "STOP", response="6.5%"),
        raw_row("Gemini-3.1-Flash-Lite", "google", "H13", "STOP", response="2 to 6 months"),
    ], existing_scored=existing)
    assert calls == ["2 to 6 months"]  # H12 was not re-scored (not re-billed)
    h12 = next(r for r in rows if r["question_id"] == "H12")
    assert h12["constraint_score"] == "6.0"
    assert h12["human_corrected_score"] == "10"
    assert h12["correction_note"] == "judge inconsistency"
