"""CLI entry point for nvProbe."""

from __future__ import annotations

import sys
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
    local: bool = typer.Option(
        False, "--local", "-l", help="Run locally on this machine (no Slurm).",
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
    console.print(f"Local:   {local}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    run_benchmarks(config, output, local=local, dry_run=dry_run)


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


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configs."),
) -> None:
    """Generate default config files in the current directory."""
    import shutil

    configs_dir = Path(__file__).parent / "configs"
    dest = Path("configs")

    if dest.exists() and not force:
        console.print(f"[yellow]configs/ already exists. Use --force to overwrite.[/yellow]")
        return

    dest.mkdir(parents=True, exist_ok=True)
    for src in configs_dir.glob("*.yaml"):
        dst = dest / src.name
        if dst.exists() and not force:
            console.print(f"  [dim]skip {dst.name}[/dim]")
        else:
            shutil.copy2(src, dst)
            console.print(f"  [green]{dst}[/green]")

    console.print(f"\n[green]Configs written to {dest}/[/green]")
    console.print("Edit [bold]configs/local.yaml[/bold] then run:")
    console.print("  nvprobe run --config configs/local.yaml --local")


@app.command()
def setup_tools(
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if already installed."),
) -> None:
    """Download and install HPL, HPCG, MLPerf locally to ~/.nvprobe/tools/."""
    import platform
    import shutil
    import subprocess
    import urllib.request

    tools_dir = Path.home() / ".nvprobe" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    arch = "x86_64" if platform.machine() == "x86_64" else "aarch64"

    # --- HPL ---
    hpl_bin = tools_dir / "xhpl"
    if hpl_bin.exists() and not force:
        console.print(f"[dim]HPL already installed: {hpl_bin}[/dim]")
    else:
        console.print("[bold]Installing HPL...[/bold]")
        try:
            url = f"https://github.com/SergioZ3R0/nvprobe-tools/releases/latest/download/xhpl-{arch}"
            urllib.request.urlretrieve(url, hpl_bin)
            hpl_bin.chmod(0o755)
            console.print(f"  [green]{hpl_bin}[/green]")
        except Exception as exc:
            console.print(f"  [yellow]HPL download failed: {exc}[/yellow]")
            console.print("  [dim]Build from source: https://www.netlib.org/benchmark/hpl/[/dim]")

    # --- HPCG ---
    hpcg_bin = tools_dir / "xhpcg"
    if hpcg_bin.exists() and not force:
        console.print(f"[dim]HPCG already installed: {hpcg_bin}[/dim]")
    else:
        console.print("[bold]Installing HPCG...[/bold]")
        try:
            url = f"https://github.com/SergioZ3R0/nvprobe-tools/releases/latest/download/xhpcg-{arch}"
            urllib.request.urlretrieve(url, hpcg_bin)
            hpcg_bin.chmod(0o755)
            console.print(f"  [green]{hpcg_bin}[/green]")
        except Exception as exc:
            console.print(f"  [yellow]HPCG download failed: {exc}[/yellow]")
            console.print("  [dim]Build from source: https://github.com/hpcg-benchmark/hpcg[/dim]")

    # --- MLPerf ---
    try:
        import mlperf_inference  # noqa: F401
        console.print("[dim]MLPerf already installed[/dim]")
    except ImportError:
        console.print("[bold]Installing MLPerf Inference...[/bold]")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "mlperf-inference"],
                check=True, capture_output=True,
            )
            console.print("  [green]mlperf-inference installed[/green]")
        except Exception as exc:
            console.print(f"  [yellow]MLPerf install failed: {exc}[/yellow]")

    # --- Update PATH ---
    path_add = str(tools_dir)
    console.print(f"\n[bold]Tools installed to: {tools_dir}[/bold]")
    console.print("Add to your shell profile:")
    console.print(f'  export PATH="{path_add}:$PATH"')
    console.print("Or run benchmarks with 'binary' param pointing to the full path.")
def slurm_cmd(
    config: Path = typer.Option(
        ..., "--config", "-c", help="YAML config file.",
        exists=True, dir_okay=False, readable=True,
    ),
    output: Path = typer.Option(
        Path("results"), "--output", "-o", help="Output directory.",
    ),
    action: str = typer.Option(
        "generate", "--action", "-a",
        help="Action: generate, submit, monitor, collect, or full (all steps).",
    ),
) -> None:
    """Manage Slurm jobs: generate scripts, submit, monitor, collect results."""
    from nvprobe.config import load_config
    from nvprobe.slurm import SlurmManager

    config_data = load_config(config)
    manager = SlurmManager(config_data, output)

    if action in ("generate", "full"):
        scripts = manager.generate_scripts()

    if action in ("submit", "full"):
        if action == "submit":
            scripts = list(manager.scripts_dir.glob("*.sh"))
        manager.submit_all(scripts)

    if action in ("monitor", "full"):
        manager.monitor()

    if action in ("collect", "full"):
        results = manager.collect_results()
        console.print(f"[green]Collected {len(results)} results[/green]")


if __name__ == "__main__":
    app()
