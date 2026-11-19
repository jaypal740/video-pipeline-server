import pytest
from pydantic import BaseModel

from src.service.llm.llm_service import LlmCall, LlmError, LlmRefusal, LlmService, LlmTruncated, ProviderReply


class Answer(BaseModel):
    value: int


class FakeProvider:
    def __init__(self, name, values):
        self.name = name
        self.values = list(values)
        self.prompts = []

    def parse(self, model, system, user, output_type, effort):
        self.prompts.append(user)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return ProviderReply(output=output_type(value=value), model=model, input_tokens=100, output_tokens=10,
                             cost_usd=0.01)


def service(openai_values=(), claude_values=()):
    return LlmService(openai=FakeProvider("openai", openai_values), anthropic=FakeProvider("anthropic", claude_values))


def test_models_route_to_their_provider():
    llm = service(openai_values=[1], claude_values=[2])
    assert llm.parse("gpt-6.1-terra", "s", "u", Answer)[0].value == 1
    output, call = llm.parse("claude-opus-5-5", "s", "u", Answer)
    assert output.value == 2 and call.provider == "anthropic" and call.attempts == 1


def test_failed_checks_are_re_asked_with_the_problems_and_cost_adds_up():
    llm = service(claude_values=[3, 4])
    output, call = llm.parse("claude-opus-5-5", "s", "u", Answer, check=lambda a: [] if a.value % 2 == 0 else ["odd"])
    assert output.value == 4 and call.attempts == 2 and call.cost_usd == pytest.approx(0.02)
    assert "- odd" in llm._anthropic.prompts[1] and '"value":3' in llm._anthropic.prompts[1]


def test_giving_up_still_reports_what_was_spent():
    llm = service(claude_values=[1, 3, 5])
    with pytest.raises(LlmError) as err:
        llm.parse("claude-opus-5-5", "s", "u", Answer, check=lambda a: ["odd"], max_attempts=3)
    assert err.value.call.attempts == 3 and err.value.call.cost_usd == pytest.approx(0.03)


def test_a_refusal_is_not_retried():
    llm = service(claude_values=[LlmRefusal("no"), 2])
    with pytest.raises(LlmError, match="refused"):
        llm.parse("claude-opus-5-5", "s", "u", Answer)
    assert len(llm._anthropic.prompts) == 1


def unusable(cost=0.02):
    return LlmError("cut off", LlmCall(provider="anthropic", model="claude-opus-5-5", attempts=1, input_tokens=50,
                                       output_tokens=900, cost_usd=cost), problems=["the answer was cut off"])


def test_an_unusable_answer_is_asked_again_and_its_cost_is_kept():
    llm = service(claude_values=[unusable(), 2])
    output, call = llm.parse("claude-opus-5-5", "s", "u", Answer)
    assert output.value == 2 and call.attempts == 2
    assert call.cost_usd == pytest.approx(0.03) and call.input_tokens == 150 and call.output_tokens == 910


def test_unusable_answers_to_the_last_attempt_still_surface_with_their_cost():
    llm = service(claude_values=[unusable(), unusable(), unusable()])
    with pytest.raises(LlmError, match="no valid answer in 3 attempts: the answer was cut off") as err:
        llm.parse("claude-opus-5-5", "s", "u", Answer, max_attempts=3)
    assert err.value.call.attempts == 3 and err.value.call.cost_usd == pytest.approx(0.06)
    assert err.value.output is None


def test_a_cut_off_answer_is_not_retried_and_its_cost_is_kept():
    cut = LlmTruncated("cut off", unusable().call, problems=["the answer was cut off"])
    llm = service(claude_values=[cut, 2])
    with pytest.raises(LlmError, match="cut off") as err:
        llm.parse("claude-opus-5-5", "s", "u", Answer)
    assert len(llm._anthropic.prompts) == 1
    assert err.value.call.attempts == 1 and err.value.call.cost_usd == pytest.approx(0.02)


def test_an_unpriced_model_is_refused_before_any_call():
    llm = service(openai_values=[1])
    with pytest.raises(ValueError, match="No price"):
        llm.parse("gpt-unknown", "s", "u", Answer)
    assert llm._openai.prompts == []


class InspectProvider(FakeProvider):
    def __init__(self, name):
        super().__init__(name, [])
        self.efforts = []

    def inspect(self, model, system, user, images, output_type, effort):
        self.efforts.append(effort)
        return ProviderReply(output=output_type(value=1), model=model, input_tokens=10, output_tokens=1, cost_usd=0.001)


def test_inspect_passes_effort():
    llm = LlmService(openai=FakeProvider("openai", []), anthropic=InspectProvider("anthropic"))
    llm.inspect("claude-opus-5-5", "s", "u", [], Answer, effort="high")
    llm.inspect("claude-opus-5-5", "s", "u", [], Answer)
    assert llm._anthropic.efforts == ["high", "low"]
