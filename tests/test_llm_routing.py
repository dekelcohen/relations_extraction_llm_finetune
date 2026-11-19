import io
import json
import sys
import types
import random
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


def test_default_model_and_effort_reach_bedrock(monkeypatch):
    provider = types.ModuleType("llm_providers.aws_bedrock")
    provider.call_llm = mock.Mock(return_value="answer")
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    assert llm.chat([], use_cache=False) == "answer"
    assert provider.call_llm.call_args.kwargs["bedrock_model_id"] == "eu.anthropic.claude-sonnet-5-5"
    assert provider.call_llm.call_args.kwargs["reasoning_effort"] == "medium"
    assert llm.MODELS["gen"] == llm.DEFAULT_MODEL
    assert llm.MODELS["weak"] == "meta-llama/llama-3.3-70b-instruct"


def test_default_chat_builds_bedrock_adaptive_request(monkeypatch):
    from llm_providers import aws_bedrock

    client = mock.Mock()
    client.invoke_model.return_value = {"body": io.BytesIO(b'{"content":[{"type":"text","text":"ok"}]}')}
    monkeypatch.setattr(aws_bedrock, "_build_client", lambda: client)
    assert llm.chat([{"role": "user", "content": "hi"}], use_cache=False) == "ok"
    request = client.invoke_model.call_args.kwargs
    body = json.loads(request["body"])
    assert request["modelId"] == "eu.anthropic.claude-sonnet-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert "temperature" not in body


def test_cli_build_passes_default_and_overrides(monkeypatch, tmp_path):
    from re_synth import __main__, build

    builder = mock.Mock()
    monkeypatch.setattr(build, "build", builder)
    monkeypatch.setattr(llm, "usage_totals", lambda: {})
    __main__.main(["build", "--out", str(tmp_path / "out.jsonl")])
    assert builder.call_args.args[4] == llm.DEFAULT_MODEL
    assert builder.call_args.kwargs["reasoning_effort"] == "medium"
    __main__.main(["build", "--model", "or-provider/model", "--reasoning-effort", "low"])
    assert builder.call_args.args[4] == "or-provider/model"
    assert builder.call_args.kwargs["reasoning_effort"] == "low"


@pytest.mark.parametrize("realistic", [False, True])
def test_generators_forward_reasoning_effort(monkeypatch, realistic):
    from re_synth import hardcases, inject, realistic as long_docs

    module = long_docs if realistic else inject
    generate = long_docs.generate_realistic if realistic else inject.generate_sample
    rng = random.Random(1)
    plan = (long_docs.plan_realistic(rng, "he", "text") if realistic
            else hardcases.plan_sample(rng, "he", "text"))
    chat = mock.Mock(return_value="")
    monkeypatch.setattr(module, "chat", chat)

    generate(plan, {"text": "source"}, rng)
    assert chat.call_args.kwargs["reasoning_effort"] == "medium"
    generate(plan, {"text": "source"}, rng, reasoning_effort="high")
    assert chat.call_args.kwargs["reasoning_effort"] == "high"


@pytest.mark.parametrize("command", ["predict", "teacher-cot"])
def test_cli_other_llm_commands_pass_defaults(monkeypatch, tmp_path, command):
    from re_synth import __main__, build, mine, teacher_cot

    monkeypatch.setattr(build, "load_samples", lambda path: [])
    monkeypatch.setattr(llm, "usage_totals", lambda: {})
    target = mine if command == "predict" else teacher_cot
    run = mock.Mock()
    monkeypatch.setattr(target, "predict" if command == "predict" else "run", run)
    __main__.main([command, "--samples", str(tmp_path / "in.jsonl"), "--out", str(tmp_path / "out")])
    assert run.call_args.args[1] == llm.DEFAULT_MODEL
    assert run.call_args.kwargs["reasoning_effort"] == "medium"


def test_inss_defaults_and_override(monkeypatch):
    from scripts import build_hard_inss as inss
    from scripts import build_hard_inss2 as inss2

    run = mock.Mock()
    monkeypatch.setattr(inss, "stage_translate", run)
    monkeypatch.setattr(sys, "argv", ["build_hard_inss.py", "translate"])
    inss.main()
    assert run.call_args.args[0].model == llm.DEFAULT_MODEL
    assert run.call_args.args[0].reasoning_effort == "medium"

    monkeypatch.setattr(inss2, "genlong", run)
    monkeypatch.setattr(sys, "argv", ["build_hard_inss2.py", "genlong", "--reasoning-effort", "high"])
    inss2.main()
    args = run.call_args.args[0]
    assert args.model == llm.DEFAULT_MODEL
    assert args.judge == "copilot-gpt-6.1-terra"
    assert args.reasoning_effort == "high"
