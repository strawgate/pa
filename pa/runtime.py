from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai_harness import CodeMode

from pa.capability import PaRegistrations
from pa.manifest import Manifest
from pa.monty_bridge import execute_registration, MontyBridgeError
from pa import primitives, registration_tools

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


# All primitive tools and their names
_PRIMITIVES: dict[str, Callable[..., Any]] = {
    "read_file": primitives.read_file,
    "write_file": primitives.write_file,
    "bash": primitives.bash,
    "http_get": primitives.http_get,
    "complete": primitives.complete,
}


def _apply_tool_filters(manifest: Manifest, tool_names: list[str]) -> list[str]:
    """Run all tool_filter registrations at build time to determine which primitives to include."""
    filters = manifest.by_slot("tool_filter")
    if not filters:
        return tool_names

    names = list(tool_names)
    for reg in filters:
        try:
            res = asyncio.run(
                execute_registration(
                    slot="tool_filter",
                    name=reg.name,
                    code=reg.code,
                    inputs={"tool_names": names},
                )
            )
            names = [n for n in res.value if n in names]
        except (MontyBridgeError, Exception):
            pass  # fail-safe: keep current list if filter crashes
    return names


def _make_registered_tool(reg) -> Callable[..., Any]:
    """Create a callable tool function from a 'tool' registration.

    The resulting function accepts **kwargs, runs the Monty code with
    args=kwargs, and returns the result.
    """

    async def _tool(**kwargs: Any) -> Any:
        res = await execute_registration(
            slot="tool",
            name=reg.name,
            code=reg.code,
            inputs={"args": kwargs},
        )
        return res.value

    _tool.__name__ = reg.name
    _tool.__doc__ = reg.description or f"User-defined tool: {reg.name}"
    return _tool


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
    # Determine which primitives survive tool_filters
    manifest = Manifest.load()
    allowed = _apply_tool_filters(manifest, list(_PRIMITIVES.keys()))

    # If no explicit model override, try to resolve from provider/route fields
    if model is None:
        model = _resolve_model_from_yaml(Path(agent_spec_path))

    agent: Agent = Agent.from_file(
        str(agent_spec_path),
        custom_capability_types=[PaRegistrations, CodeMode],
        model=model,
        defer_model_check=model is None,
    )
    # Register only the primitives that pass filters
    for name in allowed:
        agent.tool_plain(_PRIMITIVES[name])
    # Registration-management tools (always available)
    for fn in (
        registration_tools.register_instruction,
        registration_tools.register_compaction,
        registration_tools.register_guard,
        registration_tools.register_tool_filter,
        registration_tools.register_tool,
        registration_tools.list_registrations,
        registration_tools.remove_registration,
    ):
        agent.tool_plain(fn)  # ty: ignore
    # User-defined tools from registrations
    for reg in manifest.by_slot("tool"):
        agent.tool_plain(_make_registered_tool(reg))

    # Inject dynamic context: date, cwd, and optional AGENTS.md (à la pi)
    agent.system_prompt(_build_context_prompt)

    # Inject the completion function so `complete()` can call the same model
    _inject_complete_fn(agent)

    return agent


def _build_context_prompt() -> str:
    """Dynamic system prompt fragment injected at runtime.

    Appended after the static instructions so it is always current.
    Mirrors pi's pattern of injecting date + cwd last, plus project context
    from AGENTS.md if present.
    """
    import datetime

    date = datetime.date.today().isoformat()
    cwd = str(Path.cwd())

    parts = ["\nCurrent date: " + date, "\nCurrent working directory: " + cwd]

    # Load AGENTS.md from cwd (project-specific context, like pi/Claude)
    agents_md = Path("AGENTS.md")
    if agents_md.exists():
        content = agents_md.read_text(encoding="utf-8").strip()
        if content:
            parts.append("\n<project_context>\n" + content + "\n</project_context>")

    return "".join(parts)


def _inject_complete_fn(agent: Agent[Any, str]) -> None:
    """Wire up primitives.complete() to use a lightweight Agent for sub-completions."""
    sub_agent = Agent(
        model=agent.model,
        system_prompt="You are a helpful assistant. Respond concisely.",
    )

    async def _do_complete(prompt: str, system: str = "", data: str = "") -> str:
        a = sub_agent
        if system:
            a = Agent(model=agent.model, system_prompt=system)
        user_msg = prompt
        if data:
            user_msg = prompt + "\n\n<data>\n" + data + "\n</data>"
        result = await a.run(user_msg)
        return result.output

    primitives._complete_fn = _do_complete
