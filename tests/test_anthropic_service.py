from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.service.llm import anthropic_service
from src.service.llm.anthropic_service import AnthropicService
from src.service.llm.llm_service import LlmError


class Answer(BaseModel):
    value: int


class FakeStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return self.response


def service(stop_reason="end_turn", text='{"value": 1}'):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    response = SimpleNamespace(stop_reason=stop_reason, usage=usage, model="claude-opus-5-5",
                               content=[SimpleNamespace(type="text", text=text)])
    claude = AnthropicService("key")
    claude._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **kwargs: FakeStream(response))))
    return claude


def test_a_good_answer_is_returned_with_its_cost():
    reply = service().parse("claude-opus-5-5", "s", "u", Answer, "low")
    assert reply.output.value == 1 and reply.input_tokens == 1000 and reply.cost_usd > 0


def test_a_cut_off_answer_is_an_llm_error_carrying_its_cost():
    with pytest.raises(LlmError, match="max_tokens") as err:
        service(stop_reason="max_tokens").parse("claude-opus-5-5", "s", "u", Answer, "low")
    assert err.value.call.cost_usd > 0 and err.value.call.output_tokens == 500 and err.value.problems


def test_an_answer_that_breaks_the_schema_is_an_llm_error_carrying_its_cost():
    with pytest.raises(LlmError, match="does not fit the schema") as err:
        service(text='{"value": "x"}').parse("claude-opus-5-5", "s", "u", Answer, "low")
    assert err.value.call.cost_usd > 0 and err.value.problems


def test_the_client_gets_the_configured_timeout_and_retries(monkeypatch):
    seen = {}
    monkeypatch.setattr(anthropic_service.anthropic, "Anthropic", lambda **kwargs: seen.update(kwargs) or object())
    AnthropicService("key", timeout_s=30.0, max_retries=5)._get_client()
    assert seen == {"api_key": "key", "timeout": 30.0, "max_retries": 5}


def test_the_defaults_are_the_sdk_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(anthropic_service.anthropic, "Anthropic", lambda **kwargs: seen.update(kwargs) or object())
    AnthropicService("key")._get_client()
    assert seen["timeout"] == 600.0 and seen["max_retries"] == 2
