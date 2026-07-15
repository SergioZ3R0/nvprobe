"""CLI entry point for nvProbe."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from nvprobe import __version__

app = typer.Typer(
    name="nvprobe",
    help="nvProbe — run CUDA workloads, generate reports, compare hardware.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="YAML config file defining the test matrix.",
        exists=True, dir_okay=False, readable=True,
    ),
    output: Path = typer.Option(
        Path("results"), "--output", "-o", help="Directory for raw results (JSON/CSV).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would run without executing.",
    ),
) -> None:
    """Run benchmarks defined in a config file."""
    from nvprobe.runner import run_benchmarks

    console.print(f"[bold green]nvprobe v{__version__}[/bold green]")
    console.print(f"Config:  {config}")
    console.print(f"Output:  {output}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    run_benchmarks(config, output, dry_run=dry_run)


@app.command()
def report(
    results: Path = typer.Option(
        Path("results"), "--results", "-r", help="Directory containing benchmark results.",
    ),
    output: Path = typer.Option(
        Path("reports"), "--output", "-o", help="Directory for generated HTML reports.",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Report title.",
    ),
) -> None:
    """Generate an HTML report from benchmark results."""
    from nvprobe.reporter import generate_report

    console.print(f"[bold green]Generating report from {results}[/bold green]")
    generate_report(results, output, title=title)


@app.command()
def compare(
    results_a: Path = typer.Option(..., "--a", help="First result set (baseline)."),
    results_b: Path = typer.Option(..., "--b", help="Second result set (comparison)."),
    output: Path = typer.Option(
        Path("reports"), "--output", "-o", help="Directory for comparison report.",
    ),
) -> None:
    """Compare two result sets side-by-side."""
    from nvprobe.reporter import generate_comparison

    console.print(f"[bold green]Comparing {results_a} vs {results_b}[/bold green]")
    generate_comparison(results_a, results_b, output)


@app.command()
def env() -> None:
    """Show detected GPU environment (driver, CUDA, GPUs)."""
    from nvprobe.runner import detect_environment

    info = detect_environment()
    for key, value in info.items():
        console.print(f"[bold]{key}[/bold]: {value}")


@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"nvprobe {__version__}")


if __name__ == "__main__":
    app()
