"""HTML report generator — self-contained reports with matplotlib charts and corporate branding."""

from __future__ import annotations

import base64
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from nvprobe import __version__
from nvprobe.db import Database


# ── Color palette ──
COLORS = {
    "accent": "#0F6CBD",
    "accent_dark": "#0C4FA3",
    "accent_light": "#EBF3FC",
    "bg": "#F3F2F1",
    "surface": "#FFFFFF",
    "sidebar_bg": "#201F1E",
    "border": "#E0E0E0",
    "text": "#242424",
    "muted": "#707070",
    "success": "#28a745",
    "danger": "#dc3545",
}

SERIES_COLORS = [
    "#A7C7E7", "#F4C28F", "#C9A7EB", "#B8E096", "#A8E6CF",
    "#F7B7A3", "#D5C6E0", "#B8E0D2", "#F6C6EA", "#C7E8F3",
]


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _set_legend(ax, n_handles: int):
    """Place legend outside the plot when there are many entries."""
    kwargs = {"fontsize": 8}
    if n_handles > 6:
        kwargs["bbox_to_anchor"] = (1.05, 1)
        kwargs["loc"] = "upper left"
    ax.legend(**kwargs)


def _parse_metrics(metrics_raw: str | dict) -> dict[str, Any]:
    """Parse metrics from JSON string or dict."""
    if isinstance(metrics_raw, dict):
        return metrics_raw
    try:
        return json.loads(metrics_raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_val(v: Any, key: str = "mean") -> float:
    """Extract a scalar from value that may be a dict with stats or a bare number."""
    if isinstance(v, dict):
        return float(v.get(key, v.get("mean", 0)))
    return float(v)


def _chart_bandwidth(results: list[dict[str, Any]]) -> str:
    """Generate interactive bandwidth chart with Plotly (GPU + transfer type dropdowns)."""
    import json as _json
    import plotly.graph_objects as go
    import plotly.utils

    grouped: dict[str, dict] = {}
    for r in results:
        metrics = _parse_metrics(r.get("metrics", "{}"))
        key = f"GPU {r['gpu_index']} ({r['gpu_model']})"
        if "h2d" in metrics:
            grouped[key] = metrics

    if not grouped:
        return ""

    gpu_names = list(grouped.keys())
    sizes = list(list(grouped.values())[0].get("h2d", {}).keys())

    H2D_COLORS = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
                  '#c6dbef', '#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6']
    D2H_COLORS = ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b',
                  '#c7e9c0', '#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476']
    D2D_COLORS = ['#7f2704', '#a63603', '#d94801', '#e6550d', '#fd8d3c', '#fdbe85',
                  '#fdd0a2', '#7f2704', '#a63603', '#d94801', '#e6550d', '#fd8d3c']

    fig = go.Figure()
    trace_info: list[list[int | str]] = []

    for i, (gpu, data) in enumerate(grouped.items()):
        for ttype, cname, palette in [
            ("h2d", "H2D", H2D_COLORS),
            ("d2h", "D2H", D2H_COLORS),
            ("d2d", "D2D", D2D_COLORS),
        ]:
            vals = [_extract_val(v, "mean") for v in data.get(ttype, {}).values()]
            if not vals:
                continue
            fig.add_trace(go.Bar(
                x=list(sizes),
                y=vals,
                name=f"{gpu} {cname}",
                legendgroup=cname,
                marker_color=palette[i % len(palette)],
                hovertemplate="%{x} MB<br>%{y:.1f} GB/s<extra>" + cname + " — " + gpu + "</extra>",
            ))
            trace_info.append([i, ttype])

    n_gpus = len(gpu_names)
    n_traces = len(trace_info)

    fig.update_layout(
        barmode="group",
        title=dict(text="Memory Bandwidth by Buffer Size", x=0.5),
        xaxis_title="Buffer Size (MB)",
        yaxis_title="Bandwidth (GB/s)",
        hovermode="x unified",
        legend=dict(font=dict(size=10)),
        template="none",
        margin=dict(l=60, r=20, t=60, b=60),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
    )

    fig_json = _json.dumps(fig.to_plotly_json(), cls=plotly.utils.PlotlyJSONEncoder)
    trace_info_json = _json.dumps(trace_info)

    options_gpu = '<option value="-1">All GPUs</option>'
    options_gpu += "".join(f'<option value="{i}">GPU {i}</option>' for i in range(n_gpus))
    return f"""<div class="chart" style="padding:0;">
<summary style="padding:0.6rem 1rem;font-weight:600;font-size:0.9rem;background:var(--surface);cursor:pointer;user-select:none;"
  onclick="var d=this.parentElement;d.open=!d.open;if(d.open){{var el=document.getElementById('bw-chart');if(el&&el.layout)Plotly.Plots.resize(el);}}">
  Memory Bandwidth
</summary>
<div style="padding:0.75rem 1rem 0;">
  <label style="font-weight:600;font-size:0.85rem;margin-right:1rem;">
    GPU:
    <select id="bw-gpu" onchange="updateBW()" style="margin-left:0.3rem;padding:0.2rem 0.4rem;border:1px solid var(--border);border-radius:4px;">
      {options_gpu}
    </select>
  </label>
  <label style="font-weight:600;font-size:0.85rem;">
    Transfer:
    <select id="bw-transfer" onchange="updateBW()" style="margin-left:0.3rem;padding:0.2rem 0.4rem;border:1px solid var(--border);border-radius:4px;">
      <option value="all">All</option>
      <option value="h2d">H2D</option>
      <option value="d2h">D2H</option>
      <option value="d2d">D2D</option>
    </select>
  </label>
</div>
<div id="bw-chart" style="padding:0 0.5rem 0.5rem;"></div>
<script>
var bwData = {fig_json};
var bwMeta = {trace_info_json};
Plotly.newPlot('bw-chart', bwData.data, bwData.layout, {{responsive: true, displayModeBar: false}});
function updateBW() {{
  var gpu = parseInt(document.getElementById('bw-gpu').value);
  var transfer = document.getElementById('bw-transfer').value;
  var vis = bwMeta.map(function(m) {{
    return (gpu === -1 || m[0] === gpu) && (transfer === 'all' || m[1] === transfer);
  }});
  Plotly.restyle('bw-chart', 'visible', vis);
}}
</script>
</div>"""


def _chart_matmul(results: list[dict[str, Any]]) -> str:
    """Generate matmul GFLOPS scaling chart."""
    grouped: dict[str, dict] = {}
    for r in results:
        metrics = _parse_metrics(r.get("metrics", "{}"))
        matmul = metrics.get("matmul", {})
        if matmul:
            key = f"GPU {r['gpu_index']} ({r['gpu_model']}) — {r['precision']}"
            grouped[key] = matmul

    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, data) in enumerate(grouped.items()):
        sizes = [int(k) for k in data.keys()]
        gflops = [v.get("gflops", 0) for v in data.values()]
        ax.plot(sizes, gflops, "o-", label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)], linewidth=2, markersize=6)

    ax.set_xlabel("Matrix Size (N×N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Matrix Multiplication Performance")
    _set_legend(ax, len(grouped))
    ax.grid(alpha=0.3)

    return _fig_to_base64(fig)


def _chart_tiled_matmul(results: list[dict[str, Any]]) -> str:
    """Generate tiled matmul GFLOPS scaling chart."""
    grouped: dict[str, dict] = {}
    for r in results:
        metrics = _parse_metrics(r.get("metrics", "{}"))
        tiled = metrics.get("tiled_matmul", {})
        if tiled:
            key = f"GPU {r['gpu_index']} ({r['gpu_model']}) — {r['precision']}"
            grouped[key] = tiled

    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, data) in enumerate(grouped.items()):
        sizes = [int(k) for k in data.keys()]
        gflops = [v.get("gflops", 0) for v in data.values()]
        ax.plot(sizes, gflops, "^-", label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)], linewidth=2, markersize=6)

    ax.set_xlabel("Matrix Size (N×N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Tiled MatMul Performance (Shared Memory)")
    _set_legend(ax, len(grouped))
    ax.grid(alpha=0.3)

    return _fig_to_base64(fig)


def _chart_attention(results: list[dict[str, Any]]) -> str:
    """Generate attention TFLOPS scaling chart."""
    grouped: dict[str, dict] = {}
    for r in results:
        metrics = _parse_metrics(r.get("metrics", "{}"))
        attention = metrics.get("attention", {})
        if attention:
            key = f"GPU {r['gpu_index']} ({r['gpu_model']}) — {r['precision']}"
            grouped[key] = attention

    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, data) in enumerate(grouped.items()):
        sizes = [int(k) for k in data.keys()]
        tflops = [v.get("tflops", 0) for v in data.values()]
        ax.plot(sizes, tflops, "s-", label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)], linewidth=2, markersize=6)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("TFLOPS")
    ax.set_title("Scaled Dot-Product Attention Performance")
    _set_legend(ax, len(grouped))
    ax.grid(alpha=0.3)

    return _fig_to_base64(fig)


def _chart_hpl(results: list[dict[str, Any]]) -> str:
    """Generate HPL GFLOPS bar chart per problem size."""
    data: list[tuple[str, int, float]] = []
    for r in results:
        if not r.get("success"):
            continue
        if r["benchmark"] != "hpl":
            continue
        metrics = _parse_metrics(r.get("metrics", "{}"))
        gflops = metrics.get("gflops")
        size = metrics.get("problem_size")
        if gflops and size:
            label = f"GPU {r['gpu_index']} ({r['gpu_model']})"
            data.append((label, int(size), float(gflops)))

    if not data:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted(set(d[0] for d in data))
    by_label: dict[str, list[tuple[int, float]]] = {l: [] for l in labels}
    for label, size, gflops in data:
        by_label[label].append((size, gflops))

    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        pairs = sorted(by_label[label])
        sizes = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        x = [j + i * width for j in range(len(sizes))]
        ax.bar(x, vals, width, label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)])

    ax.set_xlabel("Problem Size (N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("HPL — High Performance Linpack")
    all_sizes = sorted(set(d[1] for d in data))
    ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(all_sizes))])
    ax.set_xticklabels([str(s) for s in all_sizes])
    _set_legend(ax, len(labels))
    ax.grid(axis="y", alpha=0.3)

    return _fig_to_base64(fig)


def _chart_hpcg(results: list[dict[str, Any]]) -> str:
    """Generate HPCG GFLOPS bar chart per grid size."""
    data: list[tuple[str, int, float]] = []
    for r in results:
        if not r.get("success"):
            continue
        if r["benchmark"] != "hpcg":
            continue
        metrics = _parse_metrics(r.get("metrics", "{}"))
        gflops = metrics.get("gflops")
        size = metrics.get("grid_size")
        if gflops and size:
            label = f"GPU {r['gpu_index']} ({r['gpu_model']})"
            data.append((label, int(size), float(gflops)))

    if not data:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted(set(d[0] for d in data))
    by_label: dict[str, list[tuple[int, float]]] = {l: [] for l in labels}
    for label, size, gflops in data:
        by_label[label].append((size, gflops))

    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        pairs = sorted(by_label[label])
        sizes = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        x = [j + i * width for j in range(len(sizes))]
        ax.bar(x, vals, width, label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)])

    ax.set_xlabel("Grid Size (N×N×N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("HPCG — High Performance Conjugate Gradients")
    all_sizes = sorted(set(d[1] for d in data))
    ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(all_sizes))])
    ax.set_xticklabels([str(s) for s in all_sizes])
    _set_legend(ax, len(labels))
    ax.grid(axis="y", alpha=0.3)

    return _fig_to_base64(fig)


def _chart_summary(results: list[dict[str, Any]]) -> str:
    """Generate a summary dashboard: pass/fail per benchmark + avg performance."""
    bench_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["benchmark"]
        if name not in bench_stats:
            bench_stats[name] = {"total": 0, "passed": 0, "gflops": []}
        bench_stats[name]["total"] += 1
        if r.get("success"):
            bench_stats[name]["passed"] += 1
            metrics = _parse_metrics(r.get("metrics", "{}"))
            gflops = metrics.get("gflops") or metrics.get("tflops")
            if gflops:
                bench_stats[name]["gflops"].append(float(gflops))

    if not bench_stats:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(bench_stats.keys())
    passed = [bench_stats[n]["passed"] for n in names]
    failed = [bench_stats[n]["total"] - bench_stats[n]["passed"] for n in names]

    x = range(len(names))
    w = 0.35
    ax1.bar([i - w/2 for i in x], passed, w, label="Passed", color="#28a745")
    ax1.bar([i + w/2 for i in x], failed, w, label="Failed", color="#dc3545")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Run Count")
    ax1.set_title("Pass / Fail by Benchmark")
    _set_legend(ax1, 2)
    ax1.grid(axis="y", alpha=0.3)

    perf_names = []
    perf_vals = []
    for n in names:
        vals = bench_stats[n]["gflops"]
        if vals:
            perf_names.append(n)
            perf_vals.append(sum(vals) / len(vals))
    if perf_names:
        bars = ax2.barh(perf_names, perf_vals, color=SERIES_COLORS[:len(perf_names)])
        for bar, val in zip(bars, perf_vals):
            ax2.text(bar.get_width() + max(perf_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)
        ax2.set_xlabel("Avg GFLOPS")
        ax2.set_title("Average Performance by Benchmark")
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No performance data", ha="center", va="center",
                transform=ax2.transAxes, color="gray")

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_report(
    results_dir: Path,
    output_dir: Path,
    title: str | None = None,
) -> Path:
    """Generate a full HTML report from benchmark results."""
    db_path = results_dir / "benchmarks.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No benchmark database found at {db_path}")

    with Database(db_path) as db:
        runs = db.get_runs()
        if not runs:
            raise ValueError("No runs found in database")

        latest_run = runs[0]
        results = db.get_results(latest_run["id"])
        env_info = json.loads(latest_run.get("environment") or "{}")

    report_title = title or f"nvProbe Report — {latest_run['name']}"

    # Generate charts
    charts: dict[str, str] = {}
    for name, func in [
        ("bandwidth", _chart_bandwidth),
        ("matmul", _chart_matmul),
        ("tiled_matmul", _chart_tiled_matmul),
        ("attention", _chart_attention),
        ("hpl", _chart_hpl),
        ("hpcg", _chart_hpcg),
        ("summary", _chart_summary),
    ]:
        try:
            charts[name] = func(results)
        except Exception:
            charts[name] = ""

    # Copy logo to reports directory
    logo_src = Path(__file__).parent / "nvprobe.svg"
    logo_dst = output_dir / "nvprobe.svg"
    if logo_src.exists():
        import shutil
        shutil.copy2(logo_src, logo_dst)

    html = _render_html(report_title, latest_run, results, env_info, charts, logo_src.exists())

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    # Also export CSV/JSON
    try:
        with Database(db_path) as db2:
            csv_path = db2.export_csv(latest_run["id"], output_dir / "results.csv")
            json_path = db2.export_json(latest_run["id"], output_dir / "results.json")
    except ValueError:
        pass

    return report_path


def generate_comparison(
    results_a: Path,
    results_b: Path,
    output_dir: Path,
) -> Path:
    """Generate a comparison HTML report between two result sets."""
    with Database(results_a / "benchmarks.db") as db_a, Database(results_b / "benchmarks.db") as db_b:
        runs_a = db_a.get_runs()
        runs_b = db_b.get_runs()

        if not runs_a or not runs_b:
            raise ValueError("Both result sets must have at least one run")

        results_a_data = db_a.get_results(runs_a[0]["id"])
        results_b_data = db_b.get_results(runs_b[0]["id"])

        env_a = json.loads(runs_a[0].get("environment") or "{}")
        env_b = json.loads(runs_b[0].get("environment") or "{}")

    html = _render_comparison_html(runs_a[0], results_a_data, env_a, runs_b[0], results_b_data, env_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _render_html(
    title: str,
    run: dict[str, Any],
    results: list[dict[str, Any]],
    env_info: dict[str, Any],
    charts: dict[str, str],
    has_logo: bool = True,
) -> str:
    """Render the main report HTML with sidebar, charts, and tables."""
    gpus = env_info.get("gpus", [])
    gpu_rows = ""
    for g in gpus:
        gpu_rows += f"<tr><td>{g['index']}</td><td>{g['model']}</td><td>{g['memory_total_mb']} MB</td></tr>\n"

    chart_html = ""
    _CHART_NAMES = [
        ("summary", "Summary Dashboard"),
        ("bandwidth", "Memory Bandwidth"),
        ("matmul", "Matrix Multiplication"),
        ("tiled_matmul", "Tiled MatMul (Shared Memory)"),
        ("attention", "Scaled Dot-Product Attention"),
        ("hpl", "HPL — High Performance Linpack"),
        ("hpcg", "HPCG — Conjugate Gradients"),
    ]
    for key, label in _CHART_NAMES:
        content = charts.get(key)
        if not content:
            continue
        if content.startswith("<div"):
            chart_html += content + "\n"
        else:
            chart_html += (
                f'<details class="chart">'
                f'<summary>{label}</summary>'
                f'<img src="{content}" alt="{label}">'
                f'</details>\n'
            )

    bench_tables = _render_benchmark_tables(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
    --accent: {COLORS['accent']};
    --accent-dark: {COLORS['accent_dark']};
    --accent-light: {COLORS['accent_light']};
    --bg: {COLORS['bg']};
    --surface: {COLORS['surface']};
    --sidebar-bg: {COLORS['sidebar_bg']};
    --sidebar-width: 240px;
    --border: {COLORS['border']};
    --text: {COLORS['text']};
    --muted: {COLORS['muted']};
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
[id] {{ scroll-margin-top: 1.5rem; }}
body {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    color: var(--text); line-height: 1.5; background: var(--bg);
}}
.sidebar {{
    position: fixed; top: 0; left: 0; bottom: 0;
    width: var(--sidebar-width); background: var(--sidebar-bg);
    display: flex; flex-direction: column; overflow: hidden; z-index: 200;
}}
.sidebar-header {{
    padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: center;
}}
.sidebar-title {{ font-size: 0.9rem; font-weight: 700; color: rgba(255,255,255,0.85); letter-spacing: 0.03em; }}
.sidebar-subtitle {{ font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 0.2rem; }}
.sidebar-link {{
    display: block; padding: 0.5rem 1rem; color: rgba(255,255,255,0.7);
    text-decoration: none; font-size: 0.85rem; border-left: 3px solid transparent;
    transition: background 0.12s, color 0.12s;
}}
.sidebar-link:hover {{ background: rgba(255,255,255,0.07); color: #fff; }}
.sidebar-link.active {{ background: rgba(15,108,189,0.28); color: #fff; border-left-color: var(--accent); }}
.sidebar-section {{ padding: 0.8rem 1rem 0.3rem; font-size: 0.7rem; font-weight: 700;
    color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.08em; }}
.main-content {{ margin-left: var(--sidebar-width); min-height: 100vh; padding: 2rem 2.5rem 4rem; }}
h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
h2 {{ font-size: 1.3rem; margin: 2rem 0 0.8rem; color: var(--accent-dark);
    border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }}
h3 {{ font-size: 1.05rem; margin: 1.2rem 0 0.5rem; }}
.subtitle {{ color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.5rem; background: var(--surface);
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th {{ background: var(--accent); color: #fff; padding: 0.6rem 0.8rem; text-align: left; font-size: 0.85rem; }}
td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
tr:hover td {{ background: var(--accent-light); }}
.env-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; margin: 1rem 0; }}
.env-card {{ background: var(--surface); border-radius: 8px; padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.env-card .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }}
.env-card .value {{ font-size: 1.1rem; font-weight: 600; }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
.badge-ok {{ background: #D4EDDA; color: #155724; }}
.badge-fail {{ background: #F8D7DA; color: #721C24; }}
.chart {{ margin: 0.5rem 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.chart summary {{ padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.9rem;
    background: var(--surface); user-select: none; }}
.chart summary:hover {{ background: var(--accent-light); }}
.chart[open] summary {{ border-bottom: 1px solid var(--border); }}
.chart img {{ max-width: 100%; display: block; padding: 1rem; box-sizing: border-box; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 0.75rem; text-align: center; }}
</style>
</head>
<body>
<nav class="sidebar">
    <div class="sidebar-header">
        {'<img src="nvprobe.svg" alt="nvProbe" style="width:80px;margin-bottom:0.5rem;">' if has_logo else ''}
        <div class="sidebar-title">nvProbe</div>
        <div class="sidebar-subtitle">GPU Benchmark Suite</div>
    </div>
    <div class="sidebar-section">Navigation</div>
    <a class="sidebar-link active" href="#overview">Overview</a>
    <a class="sidebar-link" href="#environment">Environment</a>
    <a class="sidebar-link" href="#charts">Charts</a>
    <a class="sidebar-link" href="#results">Results</a>
</nav>
<div class="main-content">
    <h1>{title}</h1>
    <p class="subtitle">Run: {run['name']} | {(run.get('created_at') or '')[:19]}</p>

    <h2 id="overview">Overview</h2>
    <div class="env-grid">
        <div class="env-card"><div class="label">GPUs</div><div class="value">{len(gpus)}</div></div>
        <div class="env-card"><div class="label">Driver</div><div class="value">{env_info.get('driver_version', 'N/A')}</div></div>
        <div class="env-card"><div class="label">CUDA</div><div class="value">{env_info.get('cuda_version', 'N/A')}</div></div>
        <div class="env-card"><div class="label">Results</div><div class="value">{len(results)}</div></div>
    </div>

    <h2 id="environment">Environment</h2>
    <table>
        <tr><th>GPU #</th><th>Model</th><th>Memory</th></tr>
        {gpu_rows}
    </table>

    <h2 id="charts">Performance Charts</h2>
    {chart_html if chart_html else '<p style="color:var(--muted)">No chart data available. Run benchmarks with CUDA-enabled GPUs to generate charts.</p>'}

    <h2 id="results">Detailed Results</h2>
    {bench_tables}

    <footer>Generated by nvProbe v{__version__} | {(run.get('created_at') or '')[:19]}</footer>
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
        html += "<tr><th>GPU</th><th>Model</th><th>Precision</th><th>Batch</th><th>Status</th><th>Time (s)</th><th>Metrics</th></tr>\n"
        for r in bench_results:
            status_badge = '<span class="badge badge-ok">OK</span>' if r["success"] else '<span class="badge badge-fail">FAIL</span>'
            metrics = _parse_metrics(r.get("metrics", "{}"))
            metrics_str = _format_metrics(metrics)
            html += f"<tr><td>{r['gpu_index']}</td><td>{r['gpu_model']}</td><td>{r['precision']}</td>"
            html += f"<td>{r['batch_size']}</td><td>{status_badge}</td><td>{r.get('elapsed_seconds', '')}</td>"
            html += f"<td style='font-size:0.8rem'>{metrics_str}</td></tr>\n"
        html += "</table>\n"
    return html


def _format_metrics(metrics: dict[str, Any], max_depth: int = 2) -> str:
    """Format nested metrics dict as readable string."""
    if not metrics:
        return ""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, dict) and max_depth > 0:
            inner = ", ".join(f"{ik}={_format_value(iv)}" for ik, iv in v.items())
            parts.append(f"{k}: {inner}")
        else:
            parts.append(f"{k}={_format_value(v)}")
    return "; ".join(parts)


def _format_value(v: Any) -> str:
    """Format a single metric value."""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _render_comparison_html(
    run_a: dict[str, Any],
    results_a: list[dict[str, Any]],
    env_a: dict[str, Any],
    run_b: dict[str, Any],
    results_b: list[dict[str, Any]],
    env_b: dict[str, Any],
) -> str:
    """Render comparison HTML between two runs with side-by-side charts."""
    # Build comparison chart
    all_labels = []
    all_vals_a = []
    all_vals_b = []

    for r in results_a:
        if not r.get("success"):
            continue
        metrics = _parse_metrics(r.get("metrics", "{}"))
        label = f"{r['benchmark']} GPU{r['gpu_index']} {r['precision']}"
        for kernel_data in metrics.values():
            if isinstance(kernel_data, dict):
                for v in kernel_data.values():
                    if isinstance(v, dict) and "gflops" in v:
                        all_labels.append(label)
                        all_vals_a.append(v["gflops"])

    for r in results_b:
        if not r.get("success"):
            continue
        metrics = _parse_metrics(r.get("metrics", "{}"))
        label = f"{r['benchmark']} GPU{r['gpu_index']} {r['precision']}"
        for kernel_data in metrics.values():
            if isinstance(kernel_data, dict):
                for v in kernel_data.values():
                    if isinstance(v, dict) and "gflops" in v:
                        all_vals_b.append(v["gflops"])

    chart_b64 = ""
    if all_labels and all_vals_a and all_vals_b:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(all_labels))
        w = 0.35
        ax.bar([i - w/2 for i in x], all_vals_a[:len(all_labels)], w, label=run_a["name"], color=SERIES_COLORS[0])
        ax.bar([i + w/2 for i in x], all_vals_b[:len(all_labels)], w, label=run_b["name"], color=SERIES_COLORS[1])
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("GFLOPS")
        ax.set_title("Performance Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        chart_b64 = _fig_to_base64(fig)

    chart_html = f'<details class="chart"><summary>Performance Comparison</summary><img src="{chart_b64}" alt="Comparison"></details>' if chart_b64 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nvProbe Comparison</title>
<style>
:root {{ --accent: {COLORS['accent']}; --bg: {COLORS['bg']}; --surface: {COLORS['surface']};
    --border: {COLORS['border']}; --text: {COLORS['text']}; --muted: {COLORS['muted']}; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: "Inter", sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
h2 {{ margin: 2rem 0 0.8rem; color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }}
.run-info {{ display: flex; gap: 2rem; margin: 1rem 0; }}
.run-card {{ background: var(--surface); padding: 1rem; border-radius: 8px; flex: 1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.run-card h3 {{ margin-bottom: 0.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--surface);
    border-radius: 8px; overflow: hidden; }}
th {{ background: var(--accent); color: #fff; padding: 0.6rem; text-align: left; font-size: 0.85rem; }}
td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
.chart {{ margin: 0.5rem 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.chart summary {{ padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.9rem;
    background: var(--surface); user-select: none; }}
.chart summary:hover {{ background: var(--accent-light); }}
.chart[open] summary {{ border-bottom: 1px solid var(--border); }}
.chart img {{ max-width: 100%; display: block; padding: 1rem; box-sizing: border-box; }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
.badge-ok {{ background: #D4EDDA; color: #155724; }}
.badge-fail {{ background: #F8D7DA; color: #721C24; }}
</style>
</head>
<body>
<h1>nvProbe Comparison Report</h1>
<div class="run-info">
    <div class="run-card">
        <h3>Baseline: {run_a['name']}</h3>
        <p>{(run_a.get('created_at') or '')[:19]}</p>
        <p>{len(results_a)} results</p>
    </div>
    <div class="run-card">
        <h3>Comparison: {run_b['name']}</h3>
        <p>{(run_b.get('created_at') or '')[:19]}</p>
        <p>{len(results_b)} results</p>
    </div>
</div>

<h2>Performance Comparison</h2>
{chart_html if chart_html else '<p style="color:var(--muted)">No comparable data found between runs.</p>'}

<h2>Results A — {run_a['name']}</h2>
{_render_benchmark_tables(results_a)}

<h2>Results B — {run_b['name']}</h2>
{_render_benchmark_tables(results_b)}
</body>
</html>"""
