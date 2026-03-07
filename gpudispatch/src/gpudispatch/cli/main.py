"""CLI entry point for gpudispatch."""

from __future__ import annotations

from typing import Optional

import click

from gpudispatch.core import CommandResult
from gpudispatch.profiles import dispatcher_from_profile, list_profiles as available_profiles

PROFILE_CHOICES = tuple(sorted(available_profiles().keys()))


def _parse_env_items(env_items: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI options into an env dict."""
    parsed: dict[str, str] = {}
    for item in env_items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise click.BadParameter(
                f"Invalid --env value '{item}'. Use KEY=VALUE format."
            )
        parsed[key] = value
    return parsed


@click.group()
@click.version_option(package_name="gpudispatch")
def main() -> None:
    """gpudispatch - Universal GPU orchestration."""
    pass


@main.command()
def status() -> None:
    """Show GPU status."""
    from gpudispatch.utils.gpu import detect_gpus

    gpus = detect_gpus()
    if not gpus:
        click.echo("No GPUs detected.")
        return

    for gpu in gpus:
        click.echo(
            f"GPU {gpu.index}: {gpu.name} - "
            f"{gpu.memory_used_mb}MB / {gpu.memory_total_mb}MB "
            f"({gpu.utilization_percent}% utilization)"
        )


@main.command()
@click.argument("name")
def show(name: str) -> None:
    """Show experiment details."""
    from gpudispatch.experiments import load

    exp = load(name)
    if exp:
        click.echo(f"Experiment: {exp.name}")
        click.echo(f"Metric: {exp.metric}")
        click.echo(f"Maximize: {exp.maximize}")
        click.echo(f"Trials: {len(exp._trials)}")
    else:
        click.echo(f"Experiment '{name}' not found.", err=True)
        raise SystemExit(1)


@main.command("list")
def list_experiments() -> None:
    """List experiments."""
    from gpudispatch.experiments import list_experiments as list_exps

    experiments = list_exps()
    if not experiments:
        click.echo("No experiments found.")
        return

    for name in experiments:
        click.echo(name)


@main.command("profiles")
def profiles() -> None:
    """List available opinionated dispatcher profiles."""
    for name, description in available_profiles().items():
        click.echo(f"{name}: {description}")


@main.command("run-script")
@click.option(
    "--profile",
    type=click.Choice(PROFILE_CHOICES, case_sensitive=False),
    default="quickstart",
    show_default=True,
    help="Opinionated dispatcher profile.",
)
@click.option("--gpu", "gpu_count", type=int, default=1, show_default=True)
@click.option("--memory", type=str, default=None, help="Per-job memory requirement.")
@click.option("--priority", type=int, default=0, show_default=True)
@click.option("--name", type=str, default=None, help="Optional job name.")
@click.option("--interpreter", type=str, default=None, help="Interpreter (e.g. bash).")
@click.option("--cwd", type=str, default=None, help="Working directory for script.")
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Command timeout in seconds. Overrides profile default.",
)
@click.option(
    "--env",
    "env_items",
    multiple=True,
    help="Environment override in KEY=VALUE format. Repeat for multiple values.",
)
@click.argument("script_path")
@click.argument("script_args", nargs=-1)
def run_script(
    profile: str,
    gpu_count: int,
    memory: Optional[str],
    priority: int,
    name: Optional[str],
    interpreter: Optional[str],
    cwd: Optional[str],
    timeout: Optional[float],
    env_items: tuple[str, ...],
    script_path: str,
    script_args: tuple[str, ...],
) -> None:
    """Run an existing script under dispatcher control with profile defaults."""
    env = _parse_env_items(env_items)
    dispatcher = dispatcher_from_profile(profile)

    try:
        with dispatcher:
            job = dispatcher.submit_script(
                script_path=script_path,
                script_args=script_args,
                interpreter=interpreter,
                cwd=cwd,
                env=env,
                timeout=timeout,
                gpu=gpu_count,
                memory=memory,
                priority=priority,
                name=name,
            )
            result = dispatcher.wait(job)

    except Exception as exc:
        click.echo(f"Job failed: {exc}", err=True)
        raise SystemExit(1)

    if isinstance(result, CommandResult):
        if result.stdout:
            click.echo(result.stdout.rstrip("\n"))
        if result.stderr:
            click.echo(result.stderr.rstrip("\n"), err=True)

    click.echo(f"Job completed: {job.name} ({job.id})")


if __name__ == "__main__":
    main()
