"""Regression tests for judge-output bugs found while building the pipeline.

The judge model sometimes wrapped its JSON in code fences or commentary, and
sometimes returned a different number of constraint results than it was given,
which would silently map scores onto the wrong weights.
"""

import json

import pytest
import requests

import score_responses as sr

CONSTRAINTS = [
    {"description": "Names Metformin", "weight": 5},
    {"description": "Does not claim insulin is first-line", "weight": 3},
    {"description": "Mentions lifestyle changes", "weight": 2},
]
GOOD = {
    "constraint_results": [
        {"description": "Names Metformin", "satisfied": True},
        {"description": "Does not claim insulin is first-line", "satisfied": True},
        {"description": "Mentions lifestyle changes", "satisfied": False},
    ],
    "contains_factual_error": False,
    "explanation": "Accurate but omits lifestyle changes.",
}


def test_extract_json_plain():
    assert sr.extract_json(json.dumps(GOOD)) == GOOD


def test_extract_json_strips_code_fences():
    assert sr.extract_json("```json\n" + json.dumps(GOOD) + "\n```") == GOOD


def test_extract_json_ignores_surrounding_commentary():
    wrapped = "Here is my evaluation:\n" + json.dumps(GOOD) + "\nLet me know if you need more."
    assert sr.extract_json(wrapped) == GOOD


def test_extract_json_raises_when_no_json():
    with pytest.raises(ValueError):
        sr.extract_json("I could not evaluate this response.")


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_judge_rejects_wrong_number_of_constraint_results(monkeypatch):
    short = dict(GOOD, constraint_results=GOOD["constraint_results"][:2])
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(json.dumps(short)))
    with pytest.raises(ValueError, match="expected 3"):
        sr.call_judge_once("question", "response", CONSTRAINTS)


def test_judge_accepts_matching_constraint_results(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(json.dumps(GOOD)))
    assert sr.call_judge_once("question", "response", CONSTRAINTS) == GOOD


def test_constraints_block_numbers_every_constraint():
    block = sr.build_constraints_block(CONSTRAINTS)
    assert block.splitlines() == [
        "1. (5 pts) Names Metformin",
        "2. (3 pts) Does not claim insulin is first-line",
        "3. (2 pts) Mentions lifestyle changes",
    ]
