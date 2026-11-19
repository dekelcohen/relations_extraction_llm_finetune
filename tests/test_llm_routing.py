import sys
import types
from unittest import mock

import pytest

from re_synth import llm


@pytest.mark.parametrize(
    "model,module_name,function_name,model_arg,provider_model",
    [
        ("aws-eu.model", "aws_bedrock", "call_llm", "bedrock_model_id", "eu.model"),
        ("gemini-2.5-flash", "gemini", "call_gemini", "model", "gemini-2.5-flash"),
        ("or-provider/model", "openrouter", "call_openrouter", "model", "provider/model"),
        ("provider/model", "openrouter", "call_openrouter", "model", "provider/model"),
        ("azure-deployment", "azure_openai", "call_llm", "azure_deployment_model", "deployment"),
        ("copilot-model", "copilot_cli", "call_copilot", "model", "model"),
    ],
)
def test_chat_routes_models(monkeypatch, model, module_name, function_name, model_arg, provider_model):
    provider = types.ModuleType("llm_providers." + module_name)
    provider_call = mock.Mock(return_value="answer")
    setattr(provider, function_name, provider_call)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    messages = [{"role": "user", "content": "hi"}]
    assert llm.chat(messages, model, max_tokens=123, reasoning_effort="low", use_cache=False) == "answer"
    assert provider_call.call_args.args == (messages,)
    assert provider_call.call_args.kwargs[model_arg] == provider_model
    assert provider_call.call_args.kwargs["max_tokens"] == 123
    assert provider_call.call_args.kwargs["reasoning_effort"] == "low"


def test_chat_preserves_openrouter_json_mode(monkeypatch):
    provider = types.ModuleType("llm_providers.openrouter")
    provider.call_openrouter = mock.Mock(return_value="{}")
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    assert llm.chat([], "provider/model", json_mode=True, use_cache=False) == "{}"
    assert provider.call_openrouter.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_chat_rejects_unsupported_json_mode():
    with pytest.raises(ValueError, match="only for OpenRouter"):
        llm.chat([], "gemini-2.5-flash", json_mode=True, use_cache=False)


def test_chat_preserves_copilot_json_best_effort(monkeypatch):
    provider = types.ModuleType("llm_providers.copilot_cli")
    provider.call_copilot = mock.Mock(return_value="{}")
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    assert llm.chat([], "copilot-model", json_mode=True, use_cache=False) == "{}"
    assert "response_format" not in provider.call_copilot.call_args.kwargs
