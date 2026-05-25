from __future__ import annotations

from pathlib import Path
import shutil

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install as _install_rich_tracebacks
import pydantic
import pydantic_ai
import pydantic_monty
import pydantic_ai_harness
import pydantic_core
import yaml
import httpx
from pa.conversation import run_coro_sync, run_with_incremental_history
from pa.runtime import build_agent
from pa import history as _history
from pa.manifest import Manifest
from pa.state import PaState, ensure_project_agent, ensure_state, resolve_state

# Load .env from the pa package directory — lets `pa run` work from any cwd
_dotenv = Path(__file__).parent.parent / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv

    load_dotenv(_dotenv, override=True)

# ---- Traceback filtering (pattern from fastmcp) ----
# Import modules to suppress from tracebacks. Rich accepts module objects
# directly and resolves each to os.path.dirname(module.__file__), filtering
# out frames whose filename starts with that directory.

_TRACEBACK_SUPPRESS = [
    pydantic,
    pydantic_ai,
    pydantic_monty,
    pydantic_ai_harness,
    pydantic_core,
    typer,
    yaml,
    httpx,
]

_install_rich_tracebacks(
    show_locals=False,
    max_frames=4,
    suppress=_TRACEBACK_SUPPRESS,
    extra_lines=1,
    theme=None,
    word_wrap=False,
)


app = typer.Typer(add_completion=False, no_args_is_help=True, help="pa — self-evolving Pydantic-AI agent harness.")
state_app = typer.Typer(help="Inspect or wipe pa's local state for this project.")
app.add_typer(state_app, name="state")
console = Console()

_TEMPLATE = Path(__file__).parent / "agent_template.yaml"
_GUIDE_TEMPLATE = Path(__file__).parent.parent / "docs" / "registrations.md"


def _ensure_config() -> PaState:
    """Create the home default, project fork, docs, and state dir if needed."""
    agent_path, created_agent, default_path, created_default = ensure_project_agent(
        target_path=Path("agent.yaml"),
        template_path=_TEMPLATE,
    )
    if created_default:
        console.print(f"[green]wrote {default_path}[/green]")
    if created_agent:
        console.print(f"[green]wrote {Path('agent.yaml')}[/green]")
    guide_path = Path("docs") / "registrations.md"
    if _GUIDE_TEMPLATE.exists() and not guide_path.exists():
        guide_path.parent.mkdir(exist_ok=True)
        shutil.copyfile(_GUIDE_TEMPLATE, guide_path)
        console.print(f"[green]wrote {guide_path}[/green]")
    state = resolve_state(agent_path)
    for note in ensure_state(state):
        console.print(f"[green]{note}[/green]")
    return state


def _resolve_existing_state() -> PaState:
    """Resolve state for an existing project agent without writing files."""
    agent_path = Path("agent.yaml")
    if not agent_path.is_absolute():
        agent_path = Path.cwd() / agent_path
    agent_path = agent_path.resolve()
    if not agent_path.exists():
        console.print("[red]agent.yaml not found; run `pa init` first[/red]")
        raise typer.Exit(1)
    return resolve_state(agent_path)


@app.command()
def init() -> None:
    """Create the home default, project agent.yaml fork, docs, and local state."""
    _ensure_config()


@app.command(name="clear-history")
def clear_history() -> None:
    """Delete the saved conversation history for this project."""
    state = _resolve_existing_state()
    _history.clear(state.history_path)
    console.print("[green]history cleared[/green]")


@app.command()
def doctor() -> None:
    """Smoke-check registrations and print their health."""
    state = _resolve_existing_state()
    from pa.registration_tools import check_registrations_at

    console.print_json(json=check_registrations_at(state.registrations_path))


@app.command()
def run(
    prompt: str,
    no_history: bool = typer.Option(False, "--no-history", help="Ignore saved history for this run."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Print compact tool call/result summaries."),
) -> None:
    """Run the agent once with the given prompt, resuming from saved history."""
    _try_logfire()
    state = _ensure_config()
    agent = build_agent(state.agent_spec_path)
    prior = [] if no_history else _history.load(state.history_path)
    if prior:
        console.print(f"[dim]resuming from {len(prior)} saved messages[/dim]")
    try:
        result = run_coro_sync(
            lambda: run_with_incremental_history(
                agent,
                prompt,
                prior,
                state.history_path,
                progress=_print_progress if progress else None,
            )
        )
    except Exception as e:
        _print_error(e)
        raise typer.Exit(1)
    console.print(Panel(str(result.output), title="agent"))


@app.command()
def repl(
    no_history: bool = typer.Option(False, "--no-history", help="Start with a blank history."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Print compact tool call/result summaries."),
) -> None:
    """Interactive REPL. History is loaded from and saved to pa's local state."""
    _try_logfire()
    state = _ensure_config()
    agent = build_agent(state.agent_spec_path)
    console.print("[bold]pa repl[/bold] — /exit /list /health /clear")
    history = [] if no_history else _history.load(state.history_path)
    if history:
        console.print(f"[dim]resuming from {len(history)} saved messages[/dim]")
    while True:
        try:
            line = Prompt.ask("[cyan]>[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            _history.save(history, state.history_path)
            return
        cmd = line.strip()
        if cmd == "/exit":
            _history.save(history, state.history_path)
            return
        if cmd == "/list":
            from pa.registration_tools import list_registrations_at

            console.print(list_registrations_at(state.registrations_path))
            continue
        if cmd == "/health":
            from pa.registration_tools import check_registrations_at

            console.print_json(json=check_registrations_at(state.registrations_path))
            continue
        if cmd == "/clear":
            history = []
            _history.clear(state.history_path)
            console.print("[dim]history cleared[/dim]")
            continue
        try:
            result = run_coro_sync(
                lambda: run_with_incremental_history(
                    agent,
                    line,
                    history,
                    state.history_path,
                    progress=_print_progress if progress else None,
                )
            )
        except Exception as e:
            _print_error(e)
            if not no_history:
                history = _history.load(state.history_path)
            continue
        history = result.all_messages()
        console.print(Panel(str(result.output), title="agent"))


@state_app.command("path")
def state_path() -> None:
    """Print the state directory for this project."""
    state = _resolve_existing_state()
    console.print(str(state.state_dir))


@state_app.command("ls")
def state_ls() -> None:
    """Show resolved config and state paths for this project."""
    state = _resolve_existing_state()
    console.print(f"home: {state.home}")
    console.print(f"default agent: {state.default_agent_path}")
    console.print(f"project agent: {state.agent_spec_path}")
    console.print(f"working dir: {state.working_dir}")
    console.print(f"state: {state.state_dir}")
    console.print(f"registrations: {state.registrations_path}")
    console.print(f"history: {state.history_path}")


@state_app.command("wipe")
def state_wipe(
    history: bool = typer.Option(False, "--history", help="Delete saved conversation history."),
    registrations: bool = typer.Option(False, "--registrations", help="Reset learned registrations."),
    all_: bool = typer.Option(False, "--all", help="Delete the entire project state directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete selected local state for this project."""
    if not any((history, registrations, all_)):
        console.print("[red]choose --history, --registrations, or --all[/red]")
        raise typer.Exit(1)

    state = _resolve_existing_state()
    target = state.state_dir if all_ else "selected state"
    if not yes and not typer.confirm(f"Wipe {target}?"):
        raise typer.Exit(1)

    if all_:
        if state.state_dir.exists():
            shutil.rmtree(state.state_dir)
        console.print("[green]state wiped[/green]")
        return

    if history:
        _history.clear(state.history_path)
        console.print("[green]history cleared[/green]")
    if registrations:
        if state.registrations_path.exists():
            Manifest().save(state.registrations_path)
            console.print("[green]registrations reset[/green]")
        else:
            console.print("[green]registrations already absent[/green]")


def _try_logfire() -> None:
    try:
        import logfire

        logfire.configure(send_to_logfire="if-token-present")
        logfire.instrument_pydantic_ai()
    except ImportError:
        pass


def _print_progress(line: str) -> None:
    console.print(Text("  " + line, style="dim"))


def _print_error(e: Exception) -> None:
    """Print a concise error, suppressing third-party/stdlib frames."""
    from pydantic_ai.exceptions import ModelHTTPError

    # Known model errors: just print the message
    if isinstance(e, ModelHTTPError):
        console.print(f"[red bold]error:[/red bold] {e}")
        if "Unable to calculate spend" in str(e):
            console.print(
                "[dim]hint: Disable 'Block if unable to calculate spend' in "
                "Logfire → Gateway → API Key settings for this provider.[/dim]"
            )
        return

    # For all other errors: print a short traceback showing only pa project frames.
    from rich.traceback import Traceback

    tb = Traceback.from_exception(
        type(e),
        e,
        e.__traceback__,
        show_locals=False,
        max_frames=4,
        suppress=_TRACEBACK_SUPPRESS,
        extra_lines=1,
    )
    console.print(tb)


if __name__ == "__main__":
    app()
