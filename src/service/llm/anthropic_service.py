import base64

import anthropic
from pydantic import BaseModel

from src.service.llm.llm_prices import PRICES, price_of
from src.service.llm.llm_service import LabelledImage, LlmRefusal, ProviderReply

# Thinking and the answer share this limit; above about 21k tokens the SDK only allows a streamed call.
MAX_TOKENS = 64000
# Server-side refusal fallback: if a safety classifier declines, the API re-runs the request on a model it
# picks for that refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicService:
    """Claude structured outputs, streamed so long plans have room. The reply is read as JSON only after the
    stop reason is known, so a cut-off reply is reported as such. The client is built on first use, so the
    container works without a key until a call is made."""

    name = "anthropic"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client: anthropic.Anthropic | None = None

    def parse(self, model: str, system: str, user: str, output_type: type[BaseModel], effort: str) -> ProviderReply:
        return self._parse(model, system, [{"role": "user", "content": user}], output_type, effort)

    def inspect(self, model: str, system: str, user: str, images: list[LabelledImage],
                output_type: type[BaseModel], effort: str) -> ProviderReply:
        content = []
        for i, image in enumerate(images, start=1):
            content.append({"type": "text", "text": f"Image {i}: {image.label}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": image.media_type,
                                                        "data": base64.b64encode(image.data).decode()}})
        content.append({"type": "text", "text": user})
        return self._parse(model, system, [{"role": "user", "content": content}], output_type, effort)

    def _parse(self, model: str, system: str, messages: list, output_type: type[BaseModel], effort: str) -> ProviderReply:
        schema = {"type": "json_schema", "schema": anthropic.transform_schema(output_type.model_json_schema())}
        with self._get_client().beta.messages.stream(
                model=model, max_tokens=MAX_TOKENS, system=system, messages=messages,
                output_config={"effort": effort, "format": schema}, betas=[FALLBACK_BETA],
                fallbacks="default") as stream:
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            details = response.stop_details
            raise LlmRefusal(details.explanation if details else "no explanation")
        if response.stop_reason == "max_tokens":
            raise ValueError(f"{model} hit max_tokens={MAX_TOKENS}")
        usage = response.usage
        price = PRICES.get(response.model) or price_of(model)    # a fallback may have answered
        cached = usage.cache_read_input_tokens or 0
        written = usage.cache_creation_input_tokens or 0
        cost = (usage.input_tokens * price.input + cached * price.cached_input + written * price.input * 1.25
                + usage.output_tokens * price.output) / 1_000_000
        text = "".join(b.text for b in response.content if b.type == "text")
        return ProviderReply(output=output_type.model_validate_json(text), model=response.model,
                             input_tokens=usage.input_tokens + cached + written,
                             output_tokens=usage.output_tokens, cost_usd=cost)

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("CLAUDE_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client
