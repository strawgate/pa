from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Agent
from pydantic_ai import NativeOutput, PromptedOutput, StructuredDict
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai_harness import CodeMode

from pa import primitives
from pa.builtin_instructions import PA_BUILTIN_INSTRUCTIONS
from pa.capability import PaRegistrations
from pa.pydantic_ai_compat import apply_pydantic_ai_v2_harness_compat
from pa.runtime_capabilities import PaPrimitiveTools, PaRuntimeContext
from pa.state import ensure_state, resolve_state

DEFAULT_AGENT_SPEC = Path("agent.yaml")

# Maps sdk name → model class import path
_SDK_MAP: dict[str, str] = {
    "openai": "pydantic_ai.models.openai.OpenAIChatModel",
    "openai-chat": "pydantic_ai.models.openai.OpenAIChatModel",
    "anthropic": "pydantic_ai.models.anthropic.AnthropicModel",
    "groq": "pydantic_ai.models.groq.GroqModel",
    "google-cloud": "pydantic_ai.models.google.GoogleModel",
}


def _resolve_model_from_yaml(spec_path: Path) -> Model | None:
    """If agent.yaml has `sdk` set, construct the model+provider directly.

    This bypasses pydantic-ai's model string parsing so you can use any
    routing group without needing it to be a known upstream provider.

    YAML fields:
        model: gateway/<route>:<model-name>   # e.g. gateway/minimax.io:MiniMax-M2.7-Highspeed
        sdk: <sdk-format>                     # e.g. openai, anthropic, groq
        base_url: <url>                       # optional, direct provider URL (bypasses gateway)
    """
    if not spec_path.exists():
        return None

    with open(spec_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        return None

    sdk = raw.get("sdk")
    if not sdk:
        return None  # fall through to normal model string parsing

    model_str = raw.get("model", "")
    if not model_str:
        return None

    base_url = raw.get("base_url")

    # Parse route and model name from gateway/<route>:<model>
    route: str | None = None
    model_name = model_str
    if model_str.startswith("gateway/"):
        rest = model_str.removeprefix("gateway/")
        if ":" in rest:
            route, model_name = rest.split(":", 1)
        else:
            route = rest
            model_name = rest

    # Look up the SDK class
    class_path = _SDK_MAP.get(sdk)
    if not class_path:
        return None  # unknown SDK; caller will use defer_model_check

    # Import the model class dynamically
    import importlib

    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    model_cls = getattr(mod, class_name)

    # Build provider: direct base_url or gateway
    if base_url:
        provider = _make_direct_provider(sdk, base_url)
    else:
        from pydantic_ai.providers.gateway import gateway_provider

        provider = gateway_provider(sdk, route=route)

    return model_cls(model_name, provider=provider)


def _make_direct_provider(sdk: str, base_url: str):
    """Construct a provider pointing directly at a base URL (no gateway)."""
    import os

    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("PYDANTIC_AI_GATEWAY_API_KEY", "")

    if sdk == "anthropic":
        from anthropic import AsyncAnthropic
        from pydantic_ai.providers.anthropic import AnthropicProvider

        client = AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=60.0)
        return AnthropicProvider(anthropic_client=client)
    elif sdk in ("openai", "openai-chat"):
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, base_url=base_url)
    elif sdk == "groq":
        from pydantic_ai.providers.groq import GroqProvider

        return GroqProvider(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Direct base_url not supported for sdk={sdk!r}")


def build_agent(
    agent_spec_path: str | Path = DEFAULT_AGENT_SPEC,
    *,
    model: Model | str | None = None,
) -> Agent[Any, str]:
    """Build the pa agent from an agent spec YAML.

    Args:
        agent_spec_path: Path to the agent.yaml spec file.
        model: Optional model override. If provided, replaces the model in the YAML.
               Useful for testing with TestModel or FunctionModel.
    """
    apply_pydantic_ai_v2_harness_compat()

    state = resolve_state(agent_spec_path)
    ensure_state(state)
    spec = _load_agent_spec(state.agent_spec_path)
    _inject_registration_manifest_path(spec, state.registrations_path)

    # If no explicit model override, try to resolve from provider/route fields
    if model is None:
        model = _resolve_model_from_yaml(state.agent_spec_path)

    agent: Agent = Agent.from_spec(
        spec,
        custom_capability_types=[PaRegistrations, CodeMode],
        model=model,
        instructions=PA_BUILTIN_INSTRUCTIONS,
        defer_model_check=model is None,
        capabilities=[
            PaPrimitiveTools(),
            PaRuntimeContext(working_dir=state.working_dir, project_root=state.project_root),
        ],
    )

    # Inject the completion function so `complete()` can call the same model
    _inject_complete_fn(agent)

    return agent


def _load_agent_spec(agent_spec_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(agent_spec_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{agent_spec_path}: agent spec must be a YAML mapping")
    return raw


def _inject_registration_manifest_path(spec: dict[str, Any], manifest_path: Path) -> None:
    capabilities = spec.get("capabilities")
    if not isinstance(capabilities, list):
        return
    for capability in capabilities:
        if not isinstance(capability, dict) or "PaRegistrations" not in capability:
            continue
        config = capability["PaRegistrations"]
        if config is None:
            config = {}
        if not isinstance(config, dict):
            return
        config.setdefault("manifest_path", str(manifest_path))
        capability["PaRegistrations"] = config
        return


def _inject_complete_fn(agent: Agent[Any, str]) -> None:
    """Wire up primitives.complete() to use a lightweight Agent for sub-completions."""
    sub_agent = Agent(
        model=agent.model,
        instructions="You are a helpful assistant. Respond concisely.",
    )

    async def _do_complete(
        prompt: str,
        system: str = "",
        data: str = "",
        output_schema: dict[str, Any] | None = None,
        output_mode: str = "native",
    ) -> Any:
        a = sub_agent
        if system:
            a = Agent(model=agent.model, instructions=system)
        user_msg = prompt
        if data:
            user_msg = prompt + "\n\n<data>\n" + data + "\n</data>"

        if output_schema is None:
            result = await a.run(user_msg)
            return result.output

        validation_error: ValueError | None = None
        current_msg = user_msg
        for attempt in range(3):
            output_type = _structured_output_type(output_schema, output_mode)
            try:
                result = await a.run(current_msg, output_type=output_type)
            except UserError as e:
                if output_mode == "native" and "Native structured output is not supported" in str(e):
                    result = await a.run(current_msg, output_type=_structured_output_type(output_schema, "prompted"))
                else:
                    raise
            try:
                primitives.reject_unresolved_async_placeholders(result.output)
                primitives.validate_json_schema_subset(output_schema, result.output)
            except ValueError as e:
                validation_error = e
                current_msg = _structured_retry_prompt(user_msg, output_schema, e, attempt + 1)
            else:
                return result.output

        if validation_error is not None:
            raise validation_error
        raise ValueError("structured completion failed without a validation error")

    primitives._complete_fn = _do_complete


def _structured_output_type(output_schema: dict[str, Any], output_mode: str) -> Any:
    structured = StructuredDict(output_schema)
    if output_mode == "native":
        return NativeOutput(structured)
    if output_mode == "prompted":
        return PromptedOutput(structured)
    raise ValueError("complete output_mode must be 'native' or 'prompted'")


def _structured_retry_prompt(user_msg: str, output_schema: dict[str, Any], error: ValueError, attempt: int) -> str:
    return (
        f"{user_msg}\n\n"
        f"The previous structured output attempt {attempt} did not match the schema: {error}.\n"
        "Return a value that satisfies this JSON schema exactly.\n\n"
        f"<json_schema>\n{json.dumps(output_schema, indent=2)}\n</json_schema>"
    )
