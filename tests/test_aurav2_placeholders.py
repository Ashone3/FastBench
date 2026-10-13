"""Verify opt-in AURAv2 prefixes, factual exclusions, and unchanged time rules."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import run_stream_eval as evaluator


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config.yaml"
ISOLATED = ROOT / "configs/aurav2.yaml"
CORPUS = ROOT / "configs/aurav2_placeholder_examples.json"
PROMISE = "Okay, I'll keep an eye out and let you know when that happens."


@pytest.fixture
def configured(monkeypatch):
    config = yaml.safe_load(ISOLATED.read_bytes())
    for name in ("PLACEHOLDERS_LOWER", "PLACEHOLDER_PREFIX_PATTERNS", "FORWARD_TASK_TYPES"):
        monkeypatch.setattr(evaluator, name, getattr(evaluator, name))
    evaluator.init_runtime_constants(config)
    return config


def test_only_opt_in_placeholder_settings_differ_from_original_config():
    base = yaml.safe_load(BASE.read_bytes())
    isolated = yaml.safe_load(ISOLATED.read_bytes())
    original = base["scoring"]["placeholder_responses"]
    expanded = isolated["scoring"]["placeholder_responses"]
    assert expanded[:len(original)] == original
    assert len(expanded) - len(original) == 13
    assert len(isolated["scoring"].pop("placeholder_response_prefix_patterns")) == 2
    isolated["scoring"]["placeholder_responses"] = original
    assert isolated == base


def test_reviewed_corpus_is_covered_and_factual_candidates_preserved(configured):
    corpus = json.loads(CORPUS.read_text())
    assert corpus["included"]
    for entry in corpus["included"]:
        assert entry["occurrences"]
        assert evaluator.is_placeholder(entry["text"])
    for entry in corpus["excluded_candidates"] + corpus["additional_reviewed_factual_candidates"]:
        assert not evaluator.is_placeholder(entry["text"]), entry["text"]


def test_prefixes_cover_variants_but_preserve_separate_factual_answers(configured):
    assert evaluator.is_placeholder("  " + PROMISE.upper() + "  ")
    assert not evaluator.is_placeholder(PROMISE + " The balloon was popped with the right foot.")
    assert evaluator.is_placeholder(PROMISE.replace("I'll", "I’ll"))
    assert evaluator.is_placeholder("Answer: Understood, I will keep a close eye on the purple kite.")
    assert evaluator.is_placeholder("Sure, I'll let you know when the purple kite lands.")
    assert evaluator.is_placeholder("I'll keep an eye on car No. 103. I'll let you know when it stops.")
    assert not evaluator.is_placeholder("Okay, the pen is now spinning counterclockwise around the thumb.")
    assert not evaluator.is_placeholder("Answer: Right foot.")
    assert not evaluator.is_placeholder("He said: " + PROMISE)
    assert not evaluator.is_placeholder("I'll let you know when it happens; Right foot.")
    assert not evaluator.is_placeholder("I'll keep an eye out.\nRight foot.")
    # Residual meta replies still use exact matching.
    assert evaluator.is_placeholder("Answer:")
    assert not evaluator.is_placeholder("The user's question has been answered. Right foot.")


def test_config_without_prefixes_resets_to_original_matching(configured):
    evaluator.init_runtime_constants(yaml.safe_load(BASE.read_bytes()))
    assert not evaluator.PLACEHOLDER_PREFIX_PATTERNS
    assert not evaluator.is_placeholder(PROMISE)
    assert not evaluator.is_placeholder("Sure, I'll let you know when the purple kite lands.")
    assert evaluator.is_placeholder("  OKAY  ")
    assert evaluator.is_placeholder("native control reply", {"native control reply"})


@pytest.mark.parametrize("invalid", ["foo", [""], ["["], [".*"], [None]])
def test_invalid_prefix_config_is_rejected(configured, invalid):
    configured["scoring"]["placeholder_response_prefix_patterns"] = invalid
    with pytest.raises(ValueError, match="placeholder_response_prefix_patterns"):
        evaluator.init_runtime_constants(configured)


def test_forward_skips_collected_promise_and_scores_later_answer(configured):
    sample = {
        "id": 1,
        "sqa": [
            {"timestamp": "00:12", "type": "forward", "question": "Which foot popped the balloon?"},
            {"timestamp": "00:13", "response": "Right foot."},
        ],
        "responses": [
            {"timestamp": "00:12", "response": PROMISE},
            {"timestamp": "00:13", "response": "<|silent|>"},
            {"timestamp": "00:15", "response": "Right foot."},
        ],
    }
    judge = MagicMock()
    judge.judge.return_value = {"score": 5, "explanation": "fixture answer"}
    result = evaluator.evaluate_sample(sample, judge, time_window=4, native_aura=True)[0]
    assert result["response"] == "Right foot."
    assert result["response_time"] == "00:15"
    judge.judge.assert_called_once_with("Which foot popped the balloon?", "Right foot.", "Right foot.")


def test_backward_deadline_remains_question_time(configured):
    sample = {
        "id": 1,
        "sqa": [{"timestamp": "00:12", "type": "backward", "question": "Which foot?", "response": "Right foot."}],
        "responses": [
            {"timestamp": "00:12", "response": PROMISE},
            {"timestamp": "00:13", "response": "Right foot."},
        ],
    }
    judge = MagicMock()
    result = evaluator.evaluate_sample(sample, judge, time_window=4, native_aura=True)[0]
    assert result["category"] == "LateResponse"
    assert result["score"] == 0
    judge.judge.assert_not_called()
