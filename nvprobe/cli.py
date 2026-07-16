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
        Path("nvprobe/results"), "--output", "-o", help="Directory for raw results (JSON/CSV).",
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
        Path("nvprobe/results"), "--results", "-r", help="Directory containing benchmark results.",
    ),
    output: Path = typer.Option(
        Path("nvprobe/reports"), "--output", "-o", help="Directory for generated HTML reports.",
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
        Path("nvprobe/reports"), "--output", "-o", help="Directory for comparison report.",
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


# ---------------------------------------------------------------------------
# Core logic (called by both standalone commands and `setup`)
# ---------------------------------------------------------------------------

def _do_init(force: bool = False) -> None:
    """Generate default config files under nvprobe/ working directory."""
    import shutil as _shutil

    configs_src = Path(__file__).parent / "configs"
    base = Path("nvprobe")
    dest = base / "configs"

    if dest.exists() and not force:
        console.print(f"[yellow]nvprobe/configs/ already exists. Use --force to overwrite.[/yellow]")
        return

    dest.mkdir(parents=True, exist_ok=True)
    for src in configs_src.glob("*.yaml"):
        dst = dest / src.name
        if dst.exists() and not force:
            console.print(f"  [dim]skip {dst.name}[/dim]")
        else:
            _shutil.copy2(src, dst)
            console.print(f"  [green]{dst}[/green]")

    # Create results and reports directories
    (base / "results").mkdir(exist_ok=True)
    (base / "reports").mkdir(exist_ok=True)

    console.print(f"\n[green]Project structure created in {base}/[/green]")
    console.print(f"  {base}/configs/   — YAML config files")
    console.print(f"  {base}/results/   — benchmark results (SQLite, CSV, JSON)")
    console.print(f"  {base}/reports/   — HTML reports + logo")
    console.print()
    console.print("Edit [bold]nvprobe/configs/local.yaml[/bold] then run:")
    console.print("  nvprobe run --config nvprobe/configs/local.yaml --local")


def _detect_cuda_major() -> str | None:
    """Detect CUDA major version via nvcc or nvidia-smi."""
    import subprocess
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, check=True,
            )
            for line in out.stdout.splitlines():
                if "release" in line:
                    ver = line.split("release")[-1].strip().rstrip(",").split(",")[0]
                    return ver.split(".")[0]
        except Exception:
            pass
    # Fallback: try nvidia-smi to get driver version, assume CUDA >= driver
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        ver = out.stdout.strip()
        if ver:
            major = ver.split(".")[0]
            # Rough CUDA version from driver: R535 → CUDA 12, R525 → CUDA 11, etc.
            # This is not exact but good enough for choosing tarball variant
            return "12" if int(major) >= 535 else "11"
    except Exception:
        pass
    return None


def _detect_mpi_variant() -> str:
    """Detect MPI implementation: 'mpich' (or ABI-compatible) vs 'openmpi'.

    Returns 'openmpi' if mpirun reports Open MPI, 'mpich' otherwise.
    """
    import os
    import shutil
    import subprocess
    mpi_bin = shutil.which("mpirun")
    if mpi_bin:
        try:
            out = subprocess.run(
                [mpi_bin, "--version"], capture_output=True, text=True, timeout=5,
            )
            combined = (out.stdout + out.stderr).lower()
            if "open mpi" in combined or "openmpi" in combined:
                return "openmpi"
        except Exception:
            pass
    # Also check common env vars for hints
    opal = os.environ.get("OPAL_PREFIX", "")
    if "openmpi" in opal.lower():
        return "openmpi"
    return "mpich"


def _do_setup_tools(force: bool = False, cuda_version: str | None = None, mpi_variant: str | None = None) -> None:
    """Download and install HPL, HPCG, MLPerf locally to ~/.nvprobe/tools/."""
    import os
    import platform
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import urllib.request

    # Auto-detect CUDA version if not provided
    if cuda_version is None:
        cuda_version = _detect_cuda_major() or "12"
    # Auto-detect MPI variant if not provided
    if mpi_variant is None:
        mpi_variant = _detect_mpi_variant()

    tools_dir = Path.home() / ".nvprobe" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    arch = "x86_64" if platform.machine() == "x86_64" else "aarch64"
    nvidia_version = "26.02.02"
    tarball_name = f"nvidia_hpc_benchmarks_{mpi_variant}-linux-{arch}-{nvidia_version}-archive.tar.xz"
    base_url = (
        "https://developer.download.nvidia.com/compute/nvidia-hpc-benchmarks"
        f"/redist/nvidia_hpc_benchmarks_{mpi_variant}/linux-{arch}"
    )

    benchmarks = {
        "HPL":  ("hpl-linux-{arch}/xhpl", "xhpl"),
        "HPCG": ("hpcg-linux-{arch}/xhpcg", "xhpcg"),
    }

    def _build_internal_path(cuda_dir: str, template: str) -> str:
        prefix = f"nvidia_hpc_benchmarks_{mpi_variant}-linux-{arch}-{nvidia_version}-archive"
        return f"{prefix}/{cuda_dir}/{template.format(arch=arch)}"

    for label, (internal_path, final_name) in benchmarks.items():
        target = tools_dir / final_name
        if target.exists() and not force:
            console.print(f"[dim]{label} already installed: {target}[/dim]")

    # Check if we need to download anything
    needed = {
        label: (internal_path, final_name)
        for label, (internal_path, final_name) in benchmarks.items()
        if not (tools_dir / final_name).exists() or force
    }
    if not needed:
        return

    console.print("[bold]Downloading NVIDIA HPC Benchmarks...[/bold]")
    tarball_url = f"{base_url}/{tarball_name}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tarball_path = Path(tmp) / tarball_name
            console.print(f"  {tarball_name} (~300 MB)...")
            urllib.request.urlretrieve(tarball_url, tarball_path)

            console.print("  Extracting...")
            with tarfile.open(tarball_path, "r:xz") as tar:
                # Discover available CUDA variants in the tarball
                cuda_dirs: list[str] = []
                for member in tar.getmembers():
                    parts = member.name.split("/")
                    if len(parts) >= 3 and parts[0].startswith("nvidia_hpc_benchmarks"):
                        dirname = parts[1]
                        if dirname.startswith("cuda") and dirname not in cuda_dirs:
                            cuda_dirs.append(dirname)
                cuda_dirs.sort(reverse=True)  # prefer newest

                if not cuda_dirs:
                    console.print("  [yellow]No CUDA variants found in tarball[/yellow]")
                    return

                # Pick the best CUDA variant: prefer exact match, then fallback
                cuda_target = f"cuda{cuda_version}"
                if cuda_target in cuda_dirs:
                    selected_cuda = cuda_target
                else:
                    # Fallback to the latest available variant
                    selected_cuda = cuda_dirs[0]
                    if cuda_target != selected_cuda:
                        console.print(
                            f"  [yellow]CUDA {cuda_version} variant not found in tarball, "
                            f"using {selected_cuda} instead[/yellow]"
                        )

                console.print(f"  [dim]Using CUDA variant: {selected_cuda}[/dim]")

                for label, (template, final_name) in needed.items():
                    internal_path = _build_internal_path(selected_cuda, template)
                    try:
                        member = tar.getmember(internal_path)
                        member.name = final_name
                        tar.extract(member, path=str(tools_dir))
                        (tools_dir / final_name).chmod(0o755)
                        console.print(f"  [green]{label}: {tools_dir / final_name}[/green]")
                    except KeyError:
                        console.print(f"  [yellow]{label} binary not found in tarball ({internal_path})[/yellow]")
    except Exception as exc:
        console.print(f"  [yellow]Download failed: {exc}[/yellow]")
        if "404" in str(exc) or "HTTP Error" in str(exc):
            console.print(f"  [dim]Check: {tarball_url}[/dim]")

    mlperf_cmd = shutil.which("cr") or shutil.which("cmx")
    if mlperf_cmd:
        console.print(f"[dim]MLPerf CLI found: {mlperf_cmd}[/dim]")
    else:
        # Also check ~/.local/bin
        local_bin = os.path.expanduser("~/.local/bin")
        for name in ("cr", "cmx"):
            path = os.path.join(local_bin, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                mlperf_cmd = path
                break
        if mlperf_cmd:
            console.print(f"[dim]MLPerf CLI found: {mlperf_cmd}[/dim]")
        else:
            console.print("[dim]MLPerf not installed (optional)[/dim]")
            console.print("  [dim]To enable: pip install --user cmx4mlperf[/dim]")

    path_add = str(tools_dir)
    console.print(f"\n[bold]Tools installed to: {tools_dir}[/bold]")
    console.print("Add to your shell profile:")
    console.print(f'  export PATH="{path_add}:$PATH"')
    console.print("Or run benchmarks with 'binary' param pointing to the full path.")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configs."),
) -> None:
    """Generate default config files in the current directory."""
    _do_init(force=force)


@app.command()
def setup_tools(
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if already installed."),
) -> None:
    """Download and install HPL, HPCG, MLPerf locally to ~/.nvprobe/tools/."""
    _do_setup_tools(force=force)  # cuda_version auto-detected inside


@app.command()
def setup(
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall."),
) -> None:
    """Full setup: install cupy[ctk], download HPL/HPCG, generate configs."""
    import shutil
    import subprocess

    console.print(f"[bold green]nvprobe v{__version__} — full setup[/bold green]\n")

    # --- Step 1: Detect CUDA and install cupy ---
    console.print("[bold]Step 1: Detect CUDA[/bold]")
    cuda_ver = None
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, check=True,
            )
            for line in out.stdout.splitlines():
                if "release" in line:
                    cuda_ver = line.split("release")[-1].strip().rstrip(",").split(",")[0]
                    break
        except Exception:
            pass

    if not cuda_ver:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            )
            console.print("  [yellow]nvcc not found (nvidia-smi present)[/yellow]")
            console.print("  [dim]Install CUDA toolkit for automatic cupy detection[/dim]")
        except Exception:
            console.print("  [yellow]No CUDA detected. Skipping cupy install.[/yellow]")
            console.print("  [dim]Install CUDA toolkit, then run: nvprobe setup-tools[/dim]")
            return

    if cuda_ver:
        cuda_major = cuda_ver.split(".")[0]
        py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
        cupy_pkg = f"cupy-cuda{cuda_major}x"
        console.print(f"  CUDA: {cuda_ver} | Python: {py_ver} | Package: {cupy_pkg}")

        # Check if installed cupy matches current CUDA version
        cupy_ok = False
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                if dist.metadata["Name"].startswith("cupy-cuda"):
                    installed_ver = dist.metadata["Name"]
                    console.print(f"  Installed: {installed_ver}")
                    if installed_ver == cupy_pkg:
                        cupy_ok = True
                    else:
                        console.print(f"  [yellow]CUDA version mismatch! {installed_ver} vs needed {cupy_pkg}[/yellow]")
                    break
        except Exception:
            pass

        if not cupy_ok:
            try:
                import cupy  # noqa: F401
                cupy_ok = True
            except ImportError:
                pass

        if cupy_ok:
            console.print("  [dim]cupy OK[/dim]")
        else:
            console.print(f"  Installing {cupy_pkg}[ctk]...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", f"{cupy_pkg}[ctk]"],
                    check=True,
                )
                console.print(f"  [green]{cupy_pkg}[ctk] installed[/green]")
            except Exception as exc:
                console.print(f"  [yellow]cupy install failed: {exc}[/yellow]")

    # --- Step 2: Download HPL/HPCG/MLPerf ---
    console.print("\n[bold]Step 2: Install HPL, HPCG, MLPerf[/bold]")
    _do_setup_tools(force=force, cuda_version=cuda_major if cuda_ver else None)

    # --- Step 3: Generate configs ---
    console.print("\n[bold]Step 3: Generate configs[/bold]")
    _do_init(force=force)


@app.command(name="slurm")
def slurm_cmd(
    config: Path = typer.Option(
        ..., "--config", "-c", help="YAML config file.",
        exists=True, dir_okay=False, readable=True,
    ),
    output: Path = typer.Option(
        Path("nvprobe/results"), "--output", "-o", help="Output directory.",
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
