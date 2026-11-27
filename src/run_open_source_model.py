"""
run_open_source_model.py

Runs an open-source model from the Hugging Face Hub locally (PyTorch +
Transformers) on every question in data/questions.json and saves the answers in
the same format as run_experiment.py, so the existing judge and analysis
scripts work on them unchanged.

The default model is Qwen2.5-1.5B-Instruct: small enough for a laptop CPU or a
free Google Colab GPU, openly licensed, and no access request needed.

Usage:
  pip install -r requirements-open-source.txt
  python src/run_open_source_model.py                                 # default model
  python src/run_open_source_model.py --model Qwen/Qwen2.5-0.5B-Instruct

Then score and analyze the new answers, kept separate from the main results:
  python src/score_responses.py --raw results/open_source/raw_responses.csv \
                                --out results/open_source/scored_responses.csv
  python src/analyze_results.py --scored results/open_source/scored_responses.csv \
                                --out results/open_source/summary_metrics.csv

Like the other scripts, this one is safe to re-run: it skips (model, question)
pairs that already finished cleanly and only redoes missing or truncated rows.

Truncation is detected the same way as for the API models: if generation stops
because it hit max_new_tokens instead of producing an end-of-sequence token,
finish_reason is recorded as "length" and the row is not trusted for scoring.
"""

import argparse
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.json"
DEFAULT_OUTPUT = ROOT / "results" / "open_source" / "raw_responses.csv"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROVIDER = "huggingface"
FIELDNAMES = ["model_label", "provider", "question_id", "domain", "question", "response", "finish_reason", "error"]


def finish_reason_from_tokens(new_token_ids, eos_token_ids, max_new_tokens):
    """'stop' if the model ended on its own, 'length' if it ran out of tokens.

    Checking the actual tokens (rather than guessing from the text) mirrors how
    run_experiment.py trusts each API's own finish_reason field.
    """
    if any(t in eos_token_ids for t in new_token_ids):
        return "stop"
    if len(new_token_ids) >= max_new_tokens:
        return "length"
    return "stop"


def load_completed_pairs(output_path):
    """(model_label, question_id) pairs that already finished cleanly."""
    done = set()
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("error") and row.get("finish_reason") == "stop":
                    done.add((row["model_label"], row["question_id"]))
    return done


def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, device, question, max_new_tokens, temperature, seed):
    import torch

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.manual_seed(seed)  # same seed per question -> reproducible sampling
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:].tolist()

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else {tokenizer.eos_token_id}
    finish_reason = finish_reason_from_tokens(new_tokens, eos_ids, max_new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, finish_reason


def main():
    parser = argparse.ArgumentParser(description="Run an open-source Hugging Face model on the question set.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path")
    parser.add_argument("--label", default=None, help="name used in the results (default: last part of --model)")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="CSV to write (appends to existing results)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3, help="same as the API models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (quick test)")
    args = parser.parse_args()

    import torch

    label = args.label or args.model.rstrip("/").split("/")[-1]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(QUESTIONS_PATH.read_text())
    if args.limit:
        questions = questions[: args.limit]
    completed = load_completed_pairs(output_path)
    todo = [q for q in questions if (label, q["id"]) not in completed]
    if not todo:
        print(f"All {len(questions)} questions already answered cleanly by {label}. Nothing to do.")
        return

    device = pick_device(torch)
    print(f"Loading {args.model} on {device} (first run downloads the weights)...")
    tokenizer, model = load_model(args.model, device)

    # Rewrite the file: keep every clean row (any model), drop stale error/truncated
    # rows for this model that we're about to redo.
    existing = []
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    redo_ids = {q["id"] for q in todo}
    keep = [r for r in existing if not (r["model_label"] == label and r["question_id"] in redo_ids)]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in keep:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
        f.flush()

        for i, q in enumerate(todo, 1):
            start = time.time()
            print(f"[{i}/{len(todo)}] {label} on {q['id']}...", end=" ", flush=True)
            try:
                text, finish_reason = generate_answer(
                    tokenizer, model, device, q["question"], args.max_new_tokens, args.temperature, args.seed
                )
                error = ""
                note = "" if finish_reason == "stop" else "  WARNING: hit max_new_tokens, answer was cut off"
                print(f"{time.time() - start:.1f}s{note}")
            except Exception as e:  # keep going; the summary below lists failures
                text, finish_reason, error = "", "", str(e)
                print(f"ERROR: {error}")
            writer.writerow({
                "model_label": label, "provider": PROVIDER, "question_id": q["id"], "domain": q["domain"],
                "question": q["question"], "response": text, "finish_reason": finish_reason, "error": error,
            })
            f.flush()

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model_label"] == label]
    bad = [r for r in rows if r.get("error") or r.get("finish_reason") != "stop"]
    print(f"\nDone. {len(rows)} rows for {label} in {output_path}")
    if bad:
        print(f"[!] {len(bad)} rows errored or were cut off -- re-run to retry them, "
              f"or raise --max-new-tokens if they keep getting cut off:")
        for r in bad:
            print(f"    {r['question_id']}: error={r.get('error') or '(none)'}, finish_reason={r.get('finish_reason')}")
    else:
        print("All rows finished cleanly.")


if __name__ == "__main__":
    main()
