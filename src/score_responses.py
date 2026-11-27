"""
score_responses.py

Uses an LLM judge (GPT-4o-mini by default) to score each raw response against
its question's weighted constraints, and computes Perfect Rate / Constraint
Score inputs. Saves results/scored_responses.csv.

Run this AFTER run_experiment.py.

Safe to re-run: only rows that are missing or previously errored get retried,
so a partial failure costs at most a handful of extra judge calls, not a
full re-score.

IMPORTANT: this is automated scoring, not ground truth. The original paper's
plan called for validating ~10% of automated scores against a human rater
(Cohen's kappa) -- do that for real if you want to defend the numbers. Don't
just report the judge's numbers as if a human never checked them.

Two columns support that human validation step: "human_corrected_score" and
"correction_note". Both are blank by default. If you manually review a row
(especially any non-perfect score -- there are usually few enough to check
ALL of them, not just a 10% sample) and disagree with the judge, fill in
human_corrected_score with the score you believe is correct and explain why
in correction_note. This script never overwrites either column once you've
set it -- re-running only touches rows that are missing or errored.
analyze_results.py uses human_corrected_score in place of constraint_score
wherever it's set, and reports how many corrections were applied.
"""

import os
import json
import csv
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = "gpt-4o-mini"

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.json"
RAW_PATH = ROOT / "results" / "raw_responses.csv"
OUTPUT_PATH = ROOT / "results" / "scored_responses.csv"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Must match run_experiment.py's FINISH_REASON_OK -- a raw response with any other
# finish_reason was cut off and should never be scored as if it were complete,
# even though run_experiment.py didn't record it as an "error" per se.
FINISH_REASON_OK = {
    "openai": "stop",
    "anthropic": "end_turn",
    "google": "STOP",
    "deepseek": "stop",
    "huggingface": "stop",  # open-source models run locally (src/run_open_source_model.py)
}

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator checking whether an AI response satisfies a list of factual/behavioral constraints.

Question asked to the model:
{question}

Model's response:
{response}

Constraints to check (each has a description and weight). There are exactly {n_constraints} constraints, numbered below:
{constraints_block}

For EACH of the {n_constraints} constraints, in the SAME order and SAME count as listed above, decide if it is
SATISFIED (true) or NOT SATISFIED (false) based strictly on the response content.
Also indicate whether the response contains any claim that is factually WRONG (not just incomplete) relative to
well-established facts -- this is separate from the constraints and used for a hallucination flag.

Respond with ONLY valid JSON and nothing else (no markdown fences, no commentary before or after), in this exact
format, with "constraint_results" having EXACTLY {n_constraints} entries:
{{
  "constraint_results": [{{"description": "...", "satisfied": true}}, ...],
  "contains_factual_error": true,
  "explanation": "one sentence"
}}
"""


def build_constraints_block(constraints):
    lines = []
    for i, c in enumerate(constraints, 1):
        lines.append(f"{i}. ({c['weight']} pts) {c['description']}")
    return "\n".join(lines)


def extract_json(raw_text):
    """Try direct parse first; fall back to extracting the outermost {...} block,
    since judge models occasionally wrap JSON in commentary or code fences despite
    instructions not to."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])  # let this raise if still invalid
    raise ValueError(f"Could not locate JSON object in judge response: {raw_text[:200]!r}")


def call_judge_once(question, response_text, constraints):
    n = len(constraints)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response=response_text,
        constraints_block=build_constraints_block(constraints),
        n_constraints=n,
    )
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 600,
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    result = extract_json(raw)

    if "constraint_results" not in result:
        raise ValueError(f"Judge response missing 'constraint_results' key: {result!r}")
    if len(result["constraint_results"]) != n:
        raise ValueError(
            f"Judge returned {len(result['constraint_results'])} constraint results, expected {n} -- "
            f"cannot safely map to weights"
        )
    return result


def call_judge_with_retries(question, response_text, constraints):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_judge_once(question, response_text, constraints)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_error = f"HTTP {status}: {e.response.text[:300] if e.response is not None else str(e)}"
            if status not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise RuntimeError(last_error) from e
        except (ValueError, json.JSONDecodeError) as e:
            # Malformed/mismatched judge output -- worth retrying once or twice,
            # since it's often a one-off formatting slip, not a permanent failure.
            last_error = f"Parse error: {e}"
            if attempt == MAX_RETRIES:
                raise RuntimeError(last_error) from e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                raise RuntimeError(last_error) from e

        wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        print(f"  Retrying in {wait}s (attempt {attempt}/{MAX_RETRIES}) after: {last_error}")
        time.sleep(wait)
    raise RuntimeError(last_error)


def main():
    questions_by_id = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text())}

    with open(RAW_PATH, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    fieldnames = [
        "model_label", "question_id", "domain",
        "constraint_score", "perfect_rate_flag", "contains_factual_error",
        "explanation", "error", "human_corrected_score", "correction_note",
    ]

    # Only rows that scored successfully (no error) count as "already done" --
    # anything that previously errored gets retried automatically.
    existing_rows = []
    already_scored = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        for row in existing_rows:
            if not row.get("error"):
                already_scored.add((row["model_label"], row["question_id"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            key = (row["model_label"], row["question_id"])
            if key in already_scored:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        out_f.flush()

        for row in raw_rows:
            key = (row["model_label"], row["question_id"])
            if key in already_scored:
                continue
            if row.get("error"):
                continue  # the original API call failed -- fix that with run_experiment.py first

            ok_reason = FINISH_REASON_OK.get(row.get("provider"))
            if row.get("finish_reason") != ok_reason:
                print(f"  SKIPPING {row['model_label']} / {row['question_id']}: raw response was truncated "
                      f"(finish_reason='{row.get('finish_reason')}') -- run run_experiment.py again to fix it first")
                continue

            q = questions_by_id[row["question_id"]]
            print(f"Scoring {row['model_label']} on {row['question_id']}...")

            score_row = {
                "model_label": row["model_label"], "question_id": row["question_id"],
                "domain": row["domain"], "constraint_score": "", "perfect_rate_flag": "",
                "contains_factual_error": "", "explanation": "", "error": "",
                "human_corrected_score": "", "correction_note": "",
            }
            try:
                result = call_judge_with_retries(q["question"], row["response"], q["constraints"])
                total_weight = sum(c["weight"] for c in q["constraints"])
                satisfied_weight = sum(
                    c["weight"] for c, r in zip(q["constraints"], result["constraint_results"])
                    if r["satisfied"]
                )
                constraint_score = round((satisfied_weight / total_weight) * 10, 2)
                score_row["constraint_score"] = constraint_score
                score_row["perfect_rate_flag"] = 1 if constraint_score == 10 else 0
                score_row["contains_factual_error"] = result.get("contains_factual_error", "")
                score_row["explanation"] = result.get("explanation", "")
            except Exception as e:
                score_row["error"] = str(e)
                print(f"  ERROR scoring: {e}")

            writer.writerow(score_row)
            out_f.flush()

    with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
        final_rows = list(csv.DictReader(f))
    bad_rows = [r for r in final_rows if r.get("error")]

    scored_pairs = {(r["model_label"], r["question_id"]) for r in final_rows}
    truncated_unscored = [
        r for r in raw_rows
        if (r["model_label"], r["question_id"]) not in scored_pairs
        and not r.get("error")
        and r.get("finish_reason") != FINISH_REASON_OK.get(r.get("provider"))
    ]

    print(f"\nDone. {len(final_rows)} total rows saved to {OUTPUT_PATH}")
    if bad_rows:
        print(f"[!] {len(bad_rows)} rows failed to score -- these are NOT included in constraint_score/perfect_rate_flag:")
        for r in bad_rows:
            print(f"    {r['model_label']} / {r['question_id']}: {r['error']}")
        print("Re-run this script again to retry just these rows.")
    if truncated_unscored:
        print(f"[!] {len(truncated_unscored)} rows were skipped because the RAW response was truncated (not this script's fault):")
        for r in truncated_unscored:
            print(f"    {r['model_label']} / {r['question_id']}: finish_reason={r.get('finish_reason')}")
        print("Run run_experiment.py again first to fix these, then re-run this script.")
    if not bad_rows and not truncated_unscored:
        print("All rows scored cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score raw model responses with the LLM judge.")
    parser.add_argument("--raw", default=str(RAW_PATH), help="raw responses CSV to score")
    parser.add_argument("--out", default=str(OUTPUT_PATH), help="where to write scored responses")
    args = parser.parse_args()
    RAW_PATH, OUTPUT_PATH = Path(args.raw), Path(args.out)
    main()
