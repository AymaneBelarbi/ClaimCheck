"""USD per million tokens, copied from the providers' pricing pages.

Checked 2026-10-20:
  Anthropic  https://platform.claude.com/docs/en/about-claude/pricing
  OpenAI     https://developers.openai.com/api/docs/pricing  (standard tier)
A model missing from this table is not guessed at: cost() returns 0.0 and logs a warning.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    # Claude Haiku 5.5 bills prompts above 100K tokens at a second, higher rate card.
    long_prompt_tokens: int | None = None
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "gpt-5": Price(1.25, 10.00),
    "gpt-5-mini": Price(0.25, 2.00),
    "gpt-5-nano": Price(0.05, 0.40),
}

_warned: set[str] = set()


def cost(model: str, input_tokens: int, output_tokens: int, *, local: bool = False) -> float:
    """Dollar cost of one call. Local (Ollama) models cost nothing."""
    if local:
        return 0.0
    price = PRICES.get(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            logger.warning("no price recorded for model %r; reporting cost as 0.0", model)
        return 0.0
    rate_in, rate_out = price.input_per_mtok, price.output_per_mtok
    if (
        price.long_prompt_tokens is not None
        and price.long_input_per_mtok is not None
        and price.long_output_per_mtok is not None
        and input_tokens > price.long_prompt_tokens
    ):
        rate_in, rate_out = price.long_input_per_mtok, price.long_output_per_mtok
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
