from openai import OpenAI
from pydantic import BaseModel

from src.service.llm.llm_prices import price_of
from src.service.llm.llm_service import LabelledImage, LlmRefusal, ProviderReply


class OpenAiService:
    """OpenAI Structured Outputs through the Responses API. The client is built on first use, so the
    container works without a key until a call is made."""

    name = "openai"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client: OpenAI | None = None

    def inspect(self, model: str, system: str, user: str, images: list[LabelledImage],
                output_type: type[BaseModel]) -> ProviderReply:
        raise NotImplementedError("image checks through OpenAI are not wired yet; use a claude-* model")

    def parse(self, model: str, system: str, user: str, output_type: type[BaseModel], effort: str) -> ProviderReply:
        response = self._get_client().responses.parse(
            model=model, instructions=system, input=user, text_format=output_type,
            reasoning={"effort": effort})
        for item in response.output:
            for part in getattr(item, "content", None) or []:
                if part.type == "refusal":
                    raise LlmRefusal(part.refusal)
        if response.status == "incomplete":
            raise ValueError(f"{model} stopped early: {response.incomplete_details}")
        usage = response.usage
        cached = usage.input_tokens_details.cached_tokens or 0
        price = price_of(model)                           # response.model is the dated id of the same model
        cost = ((usage.input_tokens - cached) * price.input + cached * price.cached_input
                + usage.output_tokens * price.output) / 1_000_000
        return ProviderReply(output=response.output_parsed, model=response.model, input_tokens=usage.input_tokens,
                             output_tokens=usage.output_tokens, cost_usd=cost)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=self._api_key)
        return self._client
