import logging

import pytest

from gateway.pricing import cost


def test_known_model_cost_is_per_million_tokens() -> None:
    # gpt-5-mini: $0.25 in, $2.00 out per MTok
    assert cost("gpt-5-mini", 1_000_000, 1_000_000) == pytest.approx(2.25)
    assert cost("claude-haiku-5-5", 100_000, 100_000) == pytest.approx(0.06)


def test_long_prompt_rate_card_applies_over_threshold() -> None:
    assert cost("claude-haiku-5-5", 200_000, 0) == pytest.approx(200_000 * 0.50 / 1e6)


def test_unknown_model_costs_zero_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="gateway.pricing"):
        assert cost("mystery-model-9", 1000, 1000) == 0.0
    assert "mystery-model-9" in caplog.text


def test_local_models_are_free() -> None:
    assert cost("qwen2.5:7b-instruct", 5000, 5000, local=True) == 0.0
