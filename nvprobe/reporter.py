"""HTML report generator — self-contained reports with charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nvprobe.db import Database


def generate_report(
    results_dir: Path,
    output_dir: Path,
    title: str | None = None,
) -> Path:
    """Generate an HTML report from benchmark results in results_dir."""
    db_path = results_dir / "benchmarks.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No benchmark database found at {db_path}")

    db = Database(db_path)
    runs = db.get_runs()
    if not runs:
        raise ValueError("No runs found in database")

    latest_run = runs[0]
    results = db.get_results(latest_run["id"])
    env_info = json.loads(latest_run.get("environment", "{}"))
    db.close()

    report_title = title or f"nvProbe Report — {latest_run['name']}"

    html = _render_html(report_title, latest_run, results, env_info)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def generate_comparison(
    results_a: Path,
    results_b: Path,
    output_dir: Path,
) -> Path:
    """Generate a comparison HTML report between two result sets."""
    db_a = Database(results_a / "benchmarks.db")
    db_b = Database(results_b / "benchmarks.db")

    runs_a = db_a.get_runs()
    runs_b = db_b.get_runs()

    if not runs_a or not runs_b:
        raise ValueError("Both result sets must have at least one run")

    results_a_data = db_a.get_results(runs_a[0]["id"])
    results_b_data = db_b.get_results(runs_b[0]["id"])

    html = _render_comparison_html(runs_a[0], results_a_data, runs_b[0], results_b_data)

    db_a.close()
    db_b.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _render_html(
    title: str,
    run: dict[str, Any],
    results: list[dict[str, Any]],
    env_info: dict[str, Any],
) -> str:
    """Render the main report HTML."""
    gpus = env_info.get("gpus", [])
    gpu_rows = ""
    for g in gpus:
        gpu_rows += f"<tr><td>{g['index']}</td><td>{g['model']}</td><td>{g['memory_total_mb']} MB</td></tr>\n"

    bench_tables = _render_benchmark_tables(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
    --accent: #0F6CBD;
    --accent-dark: #0C4FA3;
    --accent-light: #EBF3FC;
    --bg: #F3F2F1;
    --surface: #FFFFFF;
    --sidebar-bg: #201F1E;
    --sidebar-width: 240px;
    --border: #E0E0E0;
    --text: #242424;
    --muted: #707070;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    color: var(--text);
    line-height: 1.5;
    background: var(--bg);
}}
.sidebar {{
    position: fixed; top: 0; left: 0; bottom: 0;
    width: var(--sidebar-width);
    background: var(--sidebar-bg);
    display: flex; flex-direction: column;
    overflow: hidden; z-index: 200;
}}
.sidebar-header {{
    padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}}
.sidebar-title {{
    font-size: 0.8rem; font-weight: 600;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase; letter-spacing: 0.04em;
}}
.sidebar-link {{
    display: block; padding: 0.5rem 1rem;
    color: rgba(255,255,255,0.7); text-decoration: none;
    font-size: 0.85rem; border-left: 3px solid transparent;
    transition: background 0.12s, color 0.12s;
}}
.sidebar-link:hover {{ background: rgba(255,255,255,0.07); color: #fff; }}
.sidebar-link.active {{ background: rgba(15,108,189,0.28); color: #fff; border-left-color: var(--accent); }}
.main-content {{
    margin-left: var(--sidebar-width);
    min-height: 100vh; padding: 2rem 2.5rem 4rem;
}}
h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
h2 {{ font-size: 1.3rem; margin: 2rem 0 0.8rem; color: var(--accent-dark); border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }}
h3 {{ font-size: 1.1rem; margin: 1.2rem 0 0.5rem; }}
.subtitle {{ color: var(--muted); margin-bottom: 1.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.5rem; background: var(--surface); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th {{ background: var(--accent); color: #fff; padding: 0.6rem 0.8rem; text-align: left; font-size: 0.85rem; }}
td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
tr:hover td {{ background: var(--accent-light); }}
.env-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; margin: 1rem 0; }}
.env-card {{ background: var(--surface); border-radius: 8px; padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.env-card .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }}
.env-card .value {{ font-size: 1.1rem; font-weight: 600; }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
.badge-ok {{ background: #D4EDDA; color: #155724; }}
.badge-fail {{ background: #F8D7DA; color: #721C24; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.75rem; text-align: center; }}
</style>
</head>
<body>
<nav class="sidebar">
    <div class="sidebar-header">
        <div class="sidebar-title">nvProbe</div>
    </div>
    <a class="sidebar-link active" href="#overview">Overview</a>
    <a class="sidebar-link" href="#environment">Environment</a>
    <a class="sidebar-link" href="#results">Results</a>
</nav>
<div class="main-content">
    <h1>{title}</h1>
    <p class="subtitle">Run: {run['name']} | {run['created_at']}</p>

    <h2 id="overview">Overview</h2>
    <div class="env-grid">
        <div class="env-card"><div class="label">GPUs</div><div class="value">{len(gpus)}</div></div>
        <div class="env-card"><div class="label">Driver</div><div class="value">{env_info.get('driver_version', 'N/A')}</div></div>
        <div class="env-card"><div class="label">CUDA</div><div class="value">{env_info.get('cuda_version', 'N/A')}</div></div>
        <div class="env-card"><div class="label">Total Results</div><div class="value">{len(results)}</div></div>
    </div>

    <h2 id="environment">Environment</h2>
    <table>
        <tr><th>GPU #</th><th>Model</th><th>Memory</th></tr>
        {gpu_rows}
    </table>

    <h2 id="results">Benchmark Results</h2>
    {bench_tables}

    <footer>Generated by nvProbe v0.1.0 | {run['created_at']}</footer>
</div>
</body>
</html>"""


def _render_benchmark_tables(results: list[dict[str, Any]]) -> str:
    """Group results by benchmark and render HTML tables."""
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["benchmark"], []).append(r)

    html = ""
    for bench_name, bench_results in grouped.items():
        html += f"<h3>{bench_name}</h3>\n<table>\n"
        html += "<tr><th>GPU</th><th>Model</th><th>Precision</th><th>Batch Size</th><th>Status</th><th>Time (s)</th><th>Metrics</th></tr>\n"
        for r in bench_results:
            status_badge = '<span class="badge badge-ok">OK</span>' if r["success"] else '<span class="badge badge-fail">FAIL</span>'
            metrics = r.get("metrics", "{}")
            if isinstance(metrics, str):
                try:
                    metrics_dict = json.loads(metrics)
                    metrics = ", ".join(f"{k}={v}" for k, v in metrics_dict.items())
                except json.JSONDecodeError:
                    pass
            html += f"<tr><td>{r['gpu_index']}</td><td>{r['gpu_model']}</td><td>{r['precision']}</td>"
            html += f"<td>{r['batch_size']}</td><td>{status_badge}</td><td>{r.get('elapsed_seconds', '')}</td>"
            html += f"<td>{metrics}</td></tr>\n"
        html += "</table>\n"
    return html


def _render_comparison_html(
    run_a: dict[str, Any],
    results_a: list[dict[str, Any]],
    run_b: dict[str, Any],
    results_b: list[dict[str, Any]],
) -> str:
    """Render comparison HTML between two runs."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nvProbe Comparison</title>
<style>
body {{ font-family: "Inter", sans-serif; margin: 2rem; background: #F3F2F1; color: #242424; }}
h1 {{ color: #0F6CBD; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: #fff; border-radius: 8px; overflow: hidden; }}
th {{ background: #0F6CBD; color: #fff; padding: 0.6rem; text-align: left; }}
td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid #E0E0E0; }}
</style>
</head>
<body>
<h1>nvProbe Comparison</h1>
<p>Baseline: {run_a['name']} ({run_a['created_at']})</p>
<p>Comparison: {run_b['name']} ({run_b['created_at']})</p>
<h2>Results A ({len(results_a)} entries)</h2>
<h2>Results B ({len(results_b)} entries)</h2>
<p><em>Detailed comparison tables coming in Fase 5.</em></p>
</body>
</html>"""
