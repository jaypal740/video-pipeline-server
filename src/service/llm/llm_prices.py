from pydantic import BaseModel

# USD per 1M tokens, standard tier, short context. Neither SDK returns a dollar cost, so every model we
# call must be listed here; update when prices change.
# OpenAI: developers.openai.com/api/docs/pricing (checked 2026-10-28; gpt-5.6-sol is on a promotional rate
# until at least 2026-11-21). Anthropic: the claude-api reference (cached 2026-10-06).


class Price(BaseModel):
    input: float
    cached_input: float
    output: float


PRICES: dict[str, Price] = {
    "gpt-6-astra": Price(input=10.00, cached_input=1.00, output=50.00),
    "gpt-6.1-sol": Price(input=2.00, cached_input=0.10, output=10.00),
    "gpt-6.1-terra": Price(input=1.00, cached_input=0.05, output=5.00),
    "gpt-6-sol": Price(input=2.00, cached_input=0.20, output=10.00),
    "gpt-6-luna": Price(input=0.10, cached_input=0.01, output=0.50),
    "gpt-5.6-sol": Price(input=4.00, cached_input=0.40, output=20.00),
    "gpt-5.6-terra": Price(input=2.00, cached_input=0.20, output=12.00),
    "gpt-5.6-luna": Price(input=0.20, cached_input=0.02, output=1.20),
    "claude-fable-5-1": Price(input=10.00, cached_input=1.00, output=50.00),
    "claude-opus-5-5": Price(input=4.00, cached_input=0.20, output=20.00),
    "claude-opus-5": Price(input=5.00, cached_input=0.50, output=25.00),
    "claude-opus-4-8": Price(input=5.00, cached_input=0.50, output=25.00),
    "claude-sonnet-5-5": Price(input=2.00, cached_input=0.20, output=10.00),
    "claude-haiku-5-5": Price(input=0.10, cached_input=0.01, output=0.50),
}


def price_of(model: str) -> Price:
    if model not in PRICES:
        raise ValueError(f"No price for model {model!r}; add it to src/service/llm/llm_prices.py before calling it")
    return PRICES[model]
