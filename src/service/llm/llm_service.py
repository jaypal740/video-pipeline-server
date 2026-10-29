import logging
from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel

from src.service.llm.llm_prices import price_of

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LlmCall(BaseModel):
    """What one parse() cost, summed over its attempts."""

    provider: str
    model: str
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ProviderReply(BaseModel):
    output: BaseModel
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LlmError(Exception):
    """A call that produced no usable output; call still carries what the failed attempts cost."""

    def __init__(self, message: str, call: LlmCall) -> None:
        super().__init__(message)
        self.call = call


class LlmRefusal(Exception):
    """The model declined; raised by a provider and never retried."""


class LabelledImage(BaseModel):
    """One picture for a vision call, introduced as "Image N: label" the way the Claude docs recommend."""

    label: str
    data: bytes
    media_type: str = "image/png"


class LlmProvider(Protocol):
    name: str

    def parse(self, model: str, system: str, user: str, output_type: type[BaseModel], effort: str) -> ProviderReply: ...

    def inspect(self, model: str, system: str, user: str, images: list[LabelledImage],
                output_type: type[BaseModel]) -> ProviderReply: ...


class LlmService:
    """One way to get a validated Pydantic object from OpenAI or Claude. The SDKs enforce the schema and
    retry transport errors; this adds the checks a schema cannot express (check returns a list of
    problems), re-asking with those problems until the output passes or attempts run out."""

    def __init__(self, openai: LlmProvider, anthropic: LlmProvider) -> None:
        self._openai = openai
        self._anthropic = anthropic

    def parse(self, model: str, system: str, user: str, output_type: type[T],
              check: Callable[[T], list[str]] | None = None, effort: str = "medium",
              max_attempts: int = 3) -> tuple[T, LlmCall]:
        price_of(model)                                   # refuse an unpriced model before spending anything
        provider = self._anthropic if model.startswith("claude-") else self._openai
        call = LlmCall(provider=provider.name, model=model)
        prompt, problems = user, []
        while call.attempts < max_attempts:
            call.attempts += 1
            try:
                reply = provider.parse(model, system, prompt, output_type, effort)
            except LlmRefusal as exc:
                raise LlmError(f"{model} refused: {exc}", call) from exc
            call.input_tokens += reply.input_tokens
            call.output_tokens += reply.output_tokens
            call.cost_usd += reply.cost_usd
            logger.info("%s attempt %d: %d in + %d out tokens, $%.4f", model, call.attempts, reply.input_tokens,
                        reply.output_tokens, reply.cost_usd)
            problems = check(reply.output) if check else []
            if not problems:
                return reply.output, call
            logger.warning("%s attempt %d failed %d checks, asking again:\n%s", model, call.attempts,
                           len(problems), "\n".join(f"  - {p}" for p in problems))
            prompt = (f"{user}\n\nYour previous answer was:\n{reply.output.model_dump_json()}\n\n"
                      "It has these problems. Return the whole answer again with every problem fixed:\n"
                      + "\n".join(f"- {p}" for p in problems))
        raise LlmError(f"{model} gave no valid answer in {max_attempts} attempts: " + "; ".join(problems), call)

    def inspect(self, model: str, system: str, user: str, images: list[LabelledImage],
                output_type: type[T]) -> tuple[T, LlmCall]:
        """One vision call: the model reads the labelled images and answers in the given shape."""
        price_of(model)
        provider = self._anthropic if model.startswith("claude-") else self._openai
        call = LlmCall(provider=provider.name, model=model, attempts=1)
        try:
            reply = provider.inspect(model, system, user, images, output_type)
        except LlmRefusal as exc:
            raise LlmError(f"{model} refused: {exc}", call) from exc
        call.input_tokens, call.output_tokens, call.cost_usd = reply.input_tokens, reply.output_tokens, reply.cost_usd
        logger.info("%s inspect: %d images, %d in + %d out tokens, $%.4f", model, len(images), reply.input_tokens,
                    reply.output_tokens, reply.cost_usd)
        return reply.output, call
