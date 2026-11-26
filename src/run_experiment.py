"""
run_experiment.py

Queries GPT-4o-mini (OpenAI), Claude Haiku 4.5 (Anthropic), Gemini 3.1 Flash-Lite
(Google), and DeepSeek-V4-Flash (DeepSeek) on every question in questions.json,
and saves the raw responses to results/raw_responses.csv.

These four are the four most-used AI providers by web/app traffic as of 2026,
which is a stronger "why these models" justification than the original plan.

Setup:
  1. pip install -r requirements.txt
  2. Copy .env.example to .env and fill in your API keys
  3. python src/run_experiment.py

Safe to re-run: it skips (model, question_id) pairs that already succeeded in
raw_responses.csv, so if it fails partway through you can just run it again --
only the missing/failed rows get retried, not the whole batch.

Every row records the provider's own finish_reason (stop / length / etc), so
truncation is detected directly from the API's own signal, not guessed at from
response text. Check the "finish_reason" column after running -- anything other
than a clean stop (see FINISH_REASON_OK below) means that response was cut off
and should not be trusted for scoring.
"""

import os
import json
import csv
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.json"
OUTPUT_PATH = ROOT / "results" / "raw_responses.csv"
TEMPERATURE = 0.3
MAX_TOKENS = 2200  # generous headroom -- empirically these models' verbose/formatted
                   # answers stay well under this. The finish_reason column confirms
                   # whether any given response actually got cut off regardless.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5  # doubles each retry: 5s, 10s, 20s

# What each provider calls "the response finished normally, nothing was cut off".
# Anything else recorded in finish_reason means: don't trust that row's completeness.
FINISH_REASON_OK = {
    "openai": "stop",
    "anthropic": "end_turn",
    "google": "STOP",
    "deepseek": "stop",
}

MODELS = [
    # (label used in results, provider, model_id_for_that_provider)
    {"label": "GPT-4o-mini", "provider": "openai", "model_id": "gpt-4o-mini"},
    {"label": "Claude-Haiku-4.5", "provider": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
    {"label": "Gemini-3.1-Flash-Lite", "provider": "google", "model_id": "gemini-3.1-flash-lite"},
    {"label": "DeepSeek-V4-Flash", "provider": "deepseek", "model_id": "deepseek-v4-flash"},
]
# NOTE: All four of these providers update model names/pricing fairly often.
# If a call starts failing with a 404 or "model not found", check the
# provider's current docs and update the "model_id" value above:
#   OpenAI:    https://platform.openai.com/docs/models
#   Anthropic: https://docs.claude.com/en/docs/about-claude/models
#   Google:    https://ai.google.dev/gemini-api/docs/models
#   DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing


def call_openai(model_id, question):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    finish_reason = data["choices"][0].get("finish_reason", "")
    return text, finish_reason


def call_anthropic(model_id, question):
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model_id,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": question}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["content"][0]["text"]
    finish_reason = data.get("stop_reason", "")
    return text, finish_reason


def call_deepseek(model_id, question):
    # DeepSeek's API is OpenAI-compatible -- same request/response shape as OpenAI,
    # just a different base URL and key. Explicitly disable thinking mode (confirmed
    # syntax per DeepSeek's own docs: {"type": "disabled"}) so the token budget goes
    # entirely to the visible answer, not hidden reasoning.
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "disabled"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    finish_reason = data["choices"][0].get("finish_reason", "")
    return text, finish_reason


def call_google(model_id, question):
    # Gemini's request/response shape is different from the OpenAI-style APIs above.
    # NOTE: gemini-3.1-flash-lite uses the "thinkingLevel" enum parameter (not the
    # older "thinkingBudget" integer parameter, which is only for Gemini 2.5 models
    # and is not valid here) and already defaults to "minimal" thinking for this
    # model -- so no thinking-mode override is needed or sent. The real fix for
    # truncation is simply a large enough maxOutputTokens, verified via finishReason.
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        headers={"x-goog-api-key": GOOGLE_API_KEY, "content-type": "application/json"},
        json={
            "contents": [{"parts": [{"text": question}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    candidate = data["candidates"][0]
    text = candidate["content"]["parts"][0]["text"]
    finish_reason = candidate.get("finishReason", "")
    return text, finish_reason


CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "deepseek": call_deepseek,
}

# HTTP status codes worth retrying (transient server-side issues), vs. failing fast
# on things like 401 (bad key) or 400 (bad request) that won't fix themselves.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def call_with_retries(provider, model_id, question):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return CALLERS[provider](model_id, question)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_error = f"HTTP {status}: {e.response.text[:300] if e.response is not None else str(e)}"
            if status not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise RuntimeError(last_error) from e
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  Transient error ({status}), retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                raise RuntimeError(last_error) from e
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  Network error, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError(last_error)


def load_completed_pairs():
    # Only count a (model, question) pair as done if it succeeded with a clean
    # finish_reason -- error rows AND truncated rows get retried on the next run.
    completed = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                provider = row["provider"]
                ok_reason = FINISH_REASON_OK.get(provider)
                if not row.get("error") and row.get("finish_reason") == ok_reason:
                    completed.add((row["model_label"], row["question_id"]))
    return completed


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    completed = load_completed_pairs()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Load any existing rows so we can rewrite the file cleanly (dropping stale
    # error/truncated rows for pairs we're about to redo), rather than appending
    # duplicates on top of them.
    existing_rows = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    fieldnames = ["model_label", "provider", "question_id", "domain", "question", "response", "finish_reason", "error"]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Re-write good existing rows first.
        for row in existing_rows:
            key = (row["model_label"], row["question_id"])
            if key in completed:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()

        for model in MODELS:
            for q in questions:
                key = (model["label"], q["id"])
                if key in completed:
                    continue

                print(f"Querying {model['label']} on {q['id']}...")
                try:
                    response_text, finish_reason = call_with_retries(model["provider"], model["model_id"], q["question"])
                    error = ""
                    ok_reason = FINISH_REASON_OK.get(model["provider"])
                    if finish_reason != ok_reason:
                        print(f"  WARNING: finish_reason='{finish_reason}' (expected '{ok_reason}') -- response may be cut off")
                except Exception as e:
                    response_text = ""
                    finish_reason = ""
                    error = str(e)
                    print(f"  ERROR: {error}")

                writer.writerow({
                    "model_label": model["label"],
                    "provider": model["provider"],
                    "question_id": q["id"],
                    "domain": q["domain"],
                    "question": q["question"],
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "error": error,
                })
                f.flush()
                time.sleep(1)  # gentle pacing to avoid rate limits

    # Summary so truncation/errors are obvious without opening the CSV.
    with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
        final_rows = list(csv.DictReader(f))
    bad_rows = [
        r for r in final_rows
        if r.get("error") or r.get("finish_reason") != FINISH_REASON_OK.get(r["provider"])
    ]
    print(f"\nDone. {len(final_rows)} total rows saved to {OUTPUT_PATH}")
    if bad_rows:
        print(f"[!] {len(bad_rows)} rows have an error or non-clean finish_reason -- these are NOT safe to score as-is:")
        for r in bad_rows:
            print(f"    {r['model_label']} / {r['question_id']}: error={r.get('error') or '(none)'}, finish_reason={r.get('finish_reason')}")
        print("Re-run this script again to retry just these rows.")
    else:
        print("All rows finished cleanly (no errors, no truncation).")


if __name__ == "__main__":
    main()
