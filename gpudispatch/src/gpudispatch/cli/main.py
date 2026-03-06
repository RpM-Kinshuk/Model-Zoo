"""CLI entry point for gpudispatch."""

from __future__ import annotations

import click


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


if __name__ == "__main__":
    main()
