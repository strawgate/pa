from __future__ import annotations

from pathlib import Path
import shutil

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.traceback import install as _install_rich_tracebacks
import pydantic
import pydantic_ai
import pydantic_monty
import pydantic_ai_harness
import pydantic_core
import yaml
import httpx
from pa.runtime import build_agent
from pa import history as _history

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
console = Console()

_TEMPLATE = Path(__file__).parent / "agent_template.yaml"


def _ensure_config() -> None:
    """Create agent.yaml + pa/registrations.yaml in cwd if they don't exist."""
    target = Path("agent.yaml")
    if target.exists():
        return
    shutil.copyfile(_TEMPLATE, target)
    console.print(f"[green]wrote {target}[/green]")
    reg_dir = Path("pa")
    reg_dir.mkdir(exist_ok=True)
    reg_path = reg_dir / "registrations.yaml"
    reg_path.write_text("registrations: []\n")
    console.print(f"[green]wrote {reg_path}[/green]")


@app.command()
def init() -> None:
    """Create agent.yaml and pa/registrations.yaml in the current directory."""
    _ensure_config()


@app.command(name="clear-history")
def clear_history() -> None:
    """Delete the saved conversation history (pa/history.json)."""
    _history.clear()
    console.print("[green]history cleared[/green]")


@app.command()
def run(
    prompt: str, no_history: bool = typer.Option(False, "--no-history", help="Ignore saved history for this run.")
) -> None:
    """Run the agent once with the given prompt, resuming from saved history."""
    _try_logfire()
    _ensure_config()
    agent = build_agent()
    prior = [] if no_history else _history.load()
    if prior:
        console.print(f"[dim]resuming from {len(prior)} saved messages[/dim]")
    try:
        result = agent.run_sync(prompt, message_history=prior)
    except Exception as e:
        _print_error(e)
        raise typer.Exit(1)
    _history.save(result.all_messages())
    console.print(Panel(str(result.output), title="agent"))


@app.command()
def repl(no_history: bool = typer.Option(False, "--no-history", help="Start with a blank history.")) -> None:
    """Interactive REPL. History is loaded from and saved to pa/history.json."""
    _try_logfire()
    _ensure_config()
    agent = build_agent()
    console.print("[bold]pa repl[/bold] — /exit /list /clear")
    history = [] if no_history else _history.load()
    if history:
        console.print(f"[dim]resuming from {len(history)} saved messages[/dim]")
    while True:
        try:
            line = Prompt.ask("[cyan]>[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            _history.save(history)
            return
        cmd = line.strip()
        if cmd == "/exit":
            _history.save(history)
            return
        if cmd == "/list":
            from pa.registration_tools import list_registrations

            console.print(list_registrations())
            continue
        if cmd == "/clear":
            history = []
            _history.clear()
            console.print("[dim]history cleared[/dim]")
            continue
        try:
            result = agent.run_sync(line, message_history=history)
        except Exception as e:
            _print_error(e)
            continue
        history = result.all_messages()
        _history.save(history)
        console.print(Panel(str(result.output), title="agent"))


def _try_logfire() -> None:
    try:
        import logfire  # type: ignore

        logfire.configure(send_to_logfire="if-token-present")
        logfire.instrument_pydantic_ai()
    except ImportError:
        pass


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
