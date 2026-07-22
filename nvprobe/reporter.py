"""HTML report generator — self-contained reports with Chart.js interactive charts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from nvprobe import __version__
from nvprobe.db import Database


# ── Design tokens (same as landing page) ──
ACCENT = "#39FF88"
ACCENT_WARN = "#FFB020"
ACCENT_FAIL = "#FF5C5C"
BG = "#0B0E10"
SURFACE = "#14181B"
TEXT = "#E8ECEF"
MUTED = "#7C8791"
BORDER = "rgba(255,255,255,0.08)"
SIDEBAR_BG = "#0B0E10"
RADIUS = "8px"

SERIES_COLORS = [
    ACCENT,
    "#5CB8FF",
    ACCENT_WARN,
    "#FF7CB8",
    "#B87CFF",
    "#7CFFF0",
]


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


def _moving_average(vals: list[float], window: int) -> list[float]:
    """Compute simple moving average (centered)."""
    if len(vals) < window or window < 2:
        return list(vals)
    half = window // 2
    result = []
    for i in range(len(vals)):
        start = max(0, i - half)
        end = min(len(vals), i + half + 1)
        result.append(sum(vals[start:end]) / (end - start))
    return result


# ── Helper: render a Chart.js chart ──

_CHART_COUNTER = 0


def _chart_canvas(
    chart_type: str,
    data: dict,
    options: dict,
    title: str,
    *,
    open: bool = True,
    aspect_ratio: float = 1.6,
) -> str:
    """Return a <details>/<canvas>/<script> block rendering the given Chart.js config."""
    global _CHART_COUNTER
    _CHART_COUNTER += 1
    chart_id = f"c{_CHART_COUNTER}"

    data_json = json.dumps(data)
    options.setdefault("responsive", True)
    options.setdefault("maintainAspectRatio", True)
    options["aspectRatio"] = aspect_ratio
    opts_json = json.dumps(options)

    return f"""<details class="chart" {"open" if open else ""}>
<summary>{title}</summary>
<div class="chart-inner">
  <canvas id="{chart_id}"></canvas>
</div>
</details>
<script>
(function(){{
  var el = document.getElementById('{chart_id}');
  if (!el || !window.Chart) return;
  new Chart(el.getContext('2d'), {{
    type: '{chart_type}',
    data: {data_json},
    options: {opts_json}
  }});
}})();
</script>"""


def _chart_default_opts(title_label: str = "") -> dict:
    """Return default Chart.js options for the dark theme."""
    return {
        "responsive": True,
        "maintainAspectRatio": True,
        "aspectRatio": 1.6,
        "plugins": {
            "legend": {
                "labels": {
                    "color": MUTED,
                    "font": {"family": "IBM Plex Mono, JetBrains Mono, Fira Code, monospace", "size": 11},
                    "boxWidth": 14,
                    "padding": 14,
                }
            },
            "tooltip": {
                "enabled": True,
                "backgroundColor": SURFACE,
                "titleFont": {"family": "IBM Plex Mono, JetBrains Mono, Fira Code, monospace", "size": 12},
                "bodyFont": {"family": "IBM Plex Mono, JetBrains Mono, Fira Code, monospace", "size": 11},
                "borderColor": BORDER,
                "borderWidth": 1,
                "padding": 10,
                "cornerRadius": 6,
                "titleColor": TEXT,
                "bodyColor": TEXT,
            },
        },
        "scales": {
            "x": {
                "grid": {"color": "rgba(255,255,255,0.04)"},
                "ticks": {
                    "color": MUTED,
                    "font": {"family": "IBM Plex Mono, JetBrains Mono, Fira Code, monospace", "size": 10},
                },
            },
            "y": {
                "grid": {"color": "rgba(255,255,255,0.04)"},
                "ticks": {
                    "color": MUTED,
                    "font": {"family": "IBM Plex Mono, JetBrains Mono, Fira Code, monospace", "size": 10},
                },
            },
        },
    }


# ── Chart generators ──


def _chart_bandwidth(results: list[dict[str, Any]]) -> str:
    """Bandwidth bar chart with GPU + transfer type dropdowns."""
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
    n_sizes = len(sizes)
    if n_sizes == 0:
        return ""

    datasets = []
    trace_info: list[list[int | str]] = []
    for i, (gpu, data) in enumerate(grouped.items()):
        for ti, (ttype, cname) in enumerate([("h2d", "H2D"), ("d2h", "D2H"), ("d2d", "D2D")]):
            vals = [_extract_val(v, "mean") for v in data.get(ttype, {}).values()]
            if not vals:
                continue
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            datasets.append({
                "label": f"{gpu} {cname}",
                "data": vals,
                "backgroundColor": color + "99",
                "borderColor": color,
                "borderWidth": 1,
                "borderRadius": 2,
                "hidden": ttype != "h2d",
                "gpu_idx": i,
                "ttype": ttype,
            })
            trace_info.append([i, ttype])

    n_traces = len(trace_info)

    def _vis_gpu(gpu_idx: int):
        return [gi == gpu_idx for gi, _ in trace_info]

    def _vis_transfer(ttype: str):
        return [tt == ttype for _, tt in trace_info]

    # Build HTML + JS with filter logic
    chart_id = "ch-bw"
    datasets_json = json.dumps(datasets)

    filter_script = f"""
<script>
(function() {{
  var chartInstance = null;
  var bwData = {datasets_json};

  function renderBandwidth(filterGpu, filterTransfer) {{
    var ds = bwData.map(function(d, i) {{
      var show = true;
      if (filterGpu !== -1 && d.gpu_idx !== filterGpu) show = false;
      if (filterTransfer && d.ttype !== filterTransfer) show = false;
      return Object.assign({{}}, d, {{hidden: !show}});
    }});
    var ctx = document.getElementById('{chart_id}');
    if (!ctx || !window.Chart) return;
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx.getContext('2d'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(sizes)},
        datasets: ds,
      }},
      options: Object.assign({{}}, {json.dumps(_chart_default_opts())}, {{
        plugins: {{
          legend: {{ labels: {{ color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}}, boxWidth: 14, padding: 14 }} }},
          tooltip: {{
            enabled: true,
            backgroundColor: '{SURFACE}',
            titleFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 12}},
            bodyFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}},
            borderColor: '{BORDER}',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 6,
            titleColor: '{TEXT}',
            bodyColor: '{TEXT}',
            callbacks: {{
              label: function(ctx) {{ return ctx.parsed.y.toFixed(1) + ' GB/s'; }}
            }}
          }}
        }},
        scales: {{
          x: {{ stacked: false, grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
          y: {{ beginAtZero: true, grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
        }},
      }})
    }});
  }}

  renderBandwidth(-1, 'h2d');
  window.bwFilterGpu = function(idx) {{ renderBandwidth(idx, document.getElementById('bw-transfer').value); }};
  window.bwFilterTransfer = function(ttype) {{ renderBandwidth(parseInt(document.getElementById('bw-gpu').value), ttype); }};
}})();
</script>"""

    gpu_opts = "".join(
        f'<option value="{i}">GPU {i}</option>' for i in range(len(gpu_names))
    )
    transfer_opts = "".join(
        f'<option value="{t}" {"selected" if t == "h2d" else ""}>{l}</option>'
        for t, l in [("h2d", "H2D"), ("d2h", "D2H"), ("d2d", "D2D")]
    )

    return f"""<details class="chart" open>
<summary>Memory Bandwidth</summary>
<div class="chart-toolbar">
  <label>GPU: <select id="bw-gpu" onchange="bwFilterGpu(parseInt(this.value))">
    <option value="-1">All GPUs</option>
    {gpu_opts}
  </select></label>
  <label>Transfer: <select id="bw-transfer" onchange="bwFilterTransfer(this.value)">
    {transfer_opts}
  </select></label>
</div>
<div class="chart-inner">
  <canvas id="{chart_id}"></canvas>
</div>
</details>
{filter_script}"""


def _chart_line_with_filters(
    results: list[dict[str, Any]],
    benchmark_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    value_key: str = "gflops",
    unit: str = "GFLOPS",
) -> str:
    """Line chart with GPU + Precision dropdowns, smoothing, and glow."""
    raw: list[tuple[int, str, dict]] = []
    for r in results:
        metrics = _parse_metrics(r.get("metrics", "{}"))
        data = metrics.get(benchmark_key, {})
        if data:
            raw.append((r["gpu_index"], r["precision"], data))

    if not raw:
        return ""

    all_sizes = sorted(set(int(k) for _, _, d in raw for k in d.keys()))
    gpu_indices = sorted(set(gi for gi, _, _ in raw))
    precisions = sorted(set(p for _, p, _ in raw))

    datasets = []
    trace_meta: list[list[int | str]] = []
    for i, (gpu_idx, prec, data) in enumerate(raw):
        sizes = sorted(int(k) for k in data.keys())
        vals = [data[str(s)].get(value_key, 0) for s in sizes]
        smooth = _moving_average(vals, max(3, len(vals) // 8))
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        datasets.append({
            "label": f"GPU {gpu_idx} ({prec})",
            "data": smooth,
            "borderColor": color,
            "backgroundColor": color + "0f",
            "fill": True,
            "tension": 0.3,
            "pointRadius": 3,
            "pointBackgroundColor": color,
            "pointBorderColor": color,
            "pointHoverRadius": 5,
            "borderWidth": 2,
            "gpu_idx": gpu_idx,
            "precision": prec,
        })
        trace_meta.append([gpu_idx, prec])

    chart_id = f"ch-{benchmark_key}"
    datasets_json = json.dumps(datasets)
    sizes_json = json.dumps(all_sizes)
    xlabel_js = json.dumps(xlabel)
    ylabel_js = json.dumps(ylabel)
    unit_js = json.dumps(unit)

    filter_script = f"""
<script>
(function() {{
  var chartInstance = null;
  var lineData = {datasets_json};

  function renderLine(filterGpu, filterPrec) {{
    var ds = lineData.map(function(d) {{
      var show = true;
      if (filterGpu !== -1 && d.gpu_idx !== filterGpu) show = false;
      if (filterPrec !== 'all' && d.precision !== filterPrec) show = false;
      return Object.assign({{}}, d, {{hidden: !show}});
    }});
    var ctx = document.getElementById('{chart_id}');
    if (!ctx || !window.Chart) return;
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx.getContext('2d'), {{
      type: 'line',
      data: {{ labels: {sizes_json}, datasets: ds }},
      options: Object.assign({{}}, {json.dumps(_chart_default_opts())}, {{
        plugins: {{
          legend: {{ labels: {{ color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}}, boxWidth: 14, padding: 14 }} }},
          tooltip: {{
            enabled: true,
            backgroundColor: '{SURFACE}',
            titleFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 12}},
            bodyFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}},
            borderColor: '{BORDER}',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 6,
            titleColor: '{TEXT}',
            bodyColor: '{TEXT}',
            callbacks: {{
              label: function(ctx) {{ return ctx.parsed.y.toFixed(1) + ' {unit_js}'; }}
            }}
          }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: {xlabel_js}, color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}} }}, grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
          y: {{ title: {{ display: true, text: {ylabel_js}, color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}} }}, beginAtZero: true, grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
        }},
      }})
    }});
  }}

  renderLine(-1, 'all');
  window['filterGpu_{chart_id}'] = function(idx) {{ renderLine(parseInt(idx), document.getElementById('{chart_id}-prec').value); }};
  window['filterPrec_{chart_id}'] = function(p) {{ renderLine(parseInt(document.getElementById('{chart_id}-gpu').value), p); }};
}})();
</script>"""

    gpu_opts = "".join(
        f'<option value="{gi}">GPU {gi}</option>' for gi in gpu_indices
    )
    prec_opts = "".join(
        f'<option value="{p}" {"selected" if p == precisions[0] else ""}>{p}</option>' for p in precisions
    )

    return f"""<details class="chart" open>
<summary>{title}</summary>
<div class="chart-toolbar">
  <label>GPU: <select id="{chart_id}-gpu" onchange="filterGpu_{chart_id}(this.value)">
    <option value="-1">All GPUs</option>
    {gpu_opts}
  </select></label>
  <label>Precision: <select id="{chart_id}-prec" onchange="filterPrec_{chart_id}(this.value)">
    <option value="all">All</option>
    {prec_opts}
  </select></label>
</div>
<div class="chart-inner">
  <canvas id="{chart_id}"></canvas>
</div>
</details>
{filter_script}"""


def _chart_matmul(results: list[dict[str, Any]]) -> str:
    return _chart_line_with_filters(results, "matmul", "Matrix Multiplication Performance",
                                    "Matrix Size (N×N)", "GFLOPS", "gflops", "GFLOPS")


def _chart_tiled_matmul(results: list[dict[str, Any]]) -> str:
    return _chart_line_with_filters(results, "tiled_matmul", "Tiled MatMul (Shared Memory)",
                                    "Matrix Size (N×N)", "GFLOPS", "gflops", "GFLOPS")


def _chart_attention(results: list[dict[str, Any]]) -> str:
    return _chart_line_with_filters(results, "attention", "Scaled Dot-Product Attention",
                                    "Sequence Length", "TFLOPS", "tflops", "TFLOPS")


def _chart_hpl(results: list[dict[str, Any]]) -> str:
    """HPL grouped bar chart per problem size."""
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

    labels = sorted(set(d[0] for d in data))
    by_label: dict[str, list[tuple[int, float]]] = {l: [] for l in labels}
    for label, size, gflops in data:
        by_label[label].append((size, gflops))

    all_sizes = sorted(set(d[1] for d in data))
    datasets = []
    for i, label in enumerate(labels):
        pairs = sorted(by_label[label])
        size_map = {s: v for s, v in pairs}
        vals = [size_map.get(s, 0) for s in all_sizes]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        datasets.append({
            "label": label,
            "data": vals,
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
            "borderRadius": 2,
        })

    return _chart_canvas(
        "bar",
        {"labels": [str(s) for s in all_sizes], "datasets": datasets},
        {
            **_chart_default_opts(),
            "plugins": {
                **_chart_default_opts()["plugins"],
                "legend": {"labels": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 11}}},
            },
            "scales": {
                "x": {"grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "GFLOPS", "color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 11}}, "grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
            },
        },
        title="HPL — High Performance Linpack",
        aspect_ratio=1.8,
    )


def _chart_hpcg(results: list[dict[str, Any]]) -> str:
    """HPCG grouped bar chart per grid size."""
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

    labels = sorted(set(d[0] for d in data))
    by_label: dict[str, list[tuple[int, float]]] = {l: [] for l in labels}
    for label, size, gflops in data:
        by_label[label].append((size, gflops))

    all_sizes = sorted(set(d[1] for d in data))
    datasets = []
    for i, label in enumerate(labels):
        pairs = sorted(by_label[label])
        size_map = {s: v for s, v in pairs}
        vals = [size_map.get(s, 0) for s in all_sizes]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        datasets.append({
            "label": label,
            "data": vals,
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
            "borderRadius": 2,
        })

    return _chart_canvas(
        "bar",
        {"labels": [str(s) for s in all_sizes], "datasets": datasets},
        {
            **_chart_default_opts(),
            "scales": {
                "x": {"grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "GFLOPS", "color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 11}}, "grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
            },
        },
        title="HPCG — Conjugate Gradients",
        aspect_ratio=1.8,
    )


def _chart_mlperf(results: list[dict[str, Any]]) -> str:
    """MLPerf horizontal bar chart with accuracy annotation."""
    data: list[tuple[str, float, float | None]] = []
    for r in results:
        if not r.get("success"):
            continue
        if r["benchmark"] != "mlperf":
            continue
        metrics = _parse_metrics(r.get("metrics", "{}"))
        qps = metrics.get("queries_per_second")
        if qps:
            label = f"GPU {r['gpu_index']} ({r['gpu_model']})"
            scenario = metrics.get("scenario", "?")
            model = metrics.get("model", "?")
            acc = metrics.get("accuracy")
            data.append((f"{label} {model} {scenario}", float(qps), float(acc) if acc else None))

    if not data:
        return ""

    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    annotations = []
    for _, _, acc in data:
        annotations.append(f"{acc:.1f}%" if acc is not None else "")

    datasets = [{
        "label": "Throughput (qps)",
        "data": vals,
        "backgroundColor": [ACCENT + "99" for _ in vals],
        "borderColor": ACCENT,
        "borderWidth": 1,
        "borderRadius": 2,
    }]

    chart_id = "ch-mlperf"
    labels_json = json.dumps(labels)
    vals_json = json.dumps(vals)
    annot_json = json.dumps(annotations)

    script = f"""
<script>
(function() {{
  var ctx = document.getElementById('{chart_id}');
  if (!ctx || !window.Chart) return;
  new Chart(ctx.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: {labels_json},
      datasets: [{{
        label: 'Throughput (qps)',
        data: {vals_json},
        backgroundColor: '{ACCENT}99',
        borderColor: '{ACCENT}',
        borderWidth: 1,
        borderRadius: 2,
      }}]
    }},
    options: Object.assign({{}}, {json.dumps(_chart_default_opts())}, {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.2,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          enabled: true,
          backgroundColor: '{SURFACE}',
          titleFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 12}},
          bodyFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}},
          borderColor: '{BORDER}',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 6,
          callbacks: {{
            afterBody: function(items) {{
              var idx = items[0].dataIndex;
              var annots = {annot_json};
              return annots[idx] ? 'Accuracy: ' + annots[idx] : '';
            }}
          }}
        }},
      }},
      scales: {{
        x: {{ beginAtZero: true, title: {{display: true, text: 'Throughput (queries / second)', color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}}, grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
        y: {{ grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{TEXT}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
      }},
    }})
  }});
}})();
</script>"""

    return f"""<details class="chart" open>
<summary>MLPerf Inference — Throughput & Accuracy</summary>
<div class="chart-inner">
  <canvas id="{chart_id}"></canvas>
</div>
</details>
{script}"""


def _chart_summary(results: list[dict[str, Any]]) -> str:
    """Summary dashboard: pass/fail per benchmark + avg performance."""
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

    names = list(bench_stats.keys())
    passed = [bench_stats[n]["passed"] for n in names]
    failed = [bench_stats[n]["total"] - bench_stats[n]["passed"] for n in names]

    perf_names = []
    perf_vals = []
    for n in names:
        vals = bench_stats[n]["gflops"]
        if vals:
            perf_names.append(n)
            perf_vals.append(sum(vals) / len(vals))

    # Pass/fail bar chart
    pass_fail_data = {
        "labels": names,
        "datasets": [
            {
                "label": "Passed",
                "data": passed,
                "backgroundColor": ACCENT + "99",
                "borderColor": ACCENT,
                "borderWidth": 1,
                "borderRadius": 2,
            },
            {
                "label": "Failed",
                "data": failed,
                "backgroundColor": ACCENT_FAIL + "99",
                "borderColor": ACCENT_FAIL,
                "borderWidth": 1,
                "borderRadius": 2,
            },
        ],
    }

    pass_fail_html = _chart_canvas(
        "bar",
        pass_fail_data,
        {
            **_chart_default_opts(),
            "plugins": {
                **_chart_default_opts()["plugins"],
                "legend": {"labels": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 11}}},
            },
            "scales": {
                "x": {"grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Run Count", "color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 11}}, "grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
            },
        },
        title="Pass / Fail by Benchmark",
        aspect_ratio=1.8,
    )

    if not perf_names:
        return pass_fail_html + '<p class="no-data">No performance data available.</p>'

    # Avg performance horizontal bar
    perf_data = {
        "labels": perf_names,
        "datasets": [{
            "label": "Avg GFLOPS",
            "data": perf_vals,
            "backgroundColor": [SERIES_COLORS[i % len(SERIES_COLORS)] + "99" for i in range(len(perf_names))],
            "borderColor": [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(perf_names))],
            "borderWidth": 1,
            "borderRadius": 2,
        }],
    }

    perf_html = _chart_canvas(
        "bar",
        perf_data,
        {
            **_chart_default_opts(),
            "indexAxis": "y",
            "plugins": {
                **_chart_default_opts()["plugins"],
                "legend": {"display": False},
            },
            "scales": {
                "x": {"beginAtZero": True, "title": {"display": True, "text": "Avg GFLOPS", "color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 11}}, "grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": MUTED, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
                "y": {"grid": {"color": "rgba(255,255,255,0.04)"}, "ticks": {"color": TEXT, "font": {"family": "'IBM Plex Mono', monospace", "size": 10}}},
            },
        },
        title="Average Performance by Benchmark",
        aspect_ratio=1.8,
    )

    return pass_fail_html + "\n" + perf_html


# ── Main report HTML template ──


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
        ("summary", _chart_summary),
        ("bandwidth", _chart_bandwidth),
        ("matmul", _chart_matmul),
        ("tiled_matmul", _chart_tiled_matmul),
        ("attention", _chart_attention),
        ("hpl", _chart_hpl),
        ("hpcg", _chart_hpcg),
        ("mlperf", _chart_mlperf),
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
        ("mlperf", "MLPerf Inference — Throughput"),
    ]
    for key, _label in _CHART_NAMES:
        content = charts.get(key)
        if content:
            chart_html += content + "\n"

    bench_tables = _render_benchmark_tables(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
    --accent: {ACCENT};
    --accent-warn: {ACCENT_WARN};
    --accent-fail: {ACCENT_FAIL};
    --bg: {BG};
    --surface: {SURFACE};
    --text: {TEXT};
    --muted: {MUTED};
    --border: {BORDER};
    --sidebar-width: 220px;
    --radius: {RADIUS};
    --font-mono: 'IBM Plex Mono', 'JetBrains Mono', 'Fira Code', monospace;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
[id] {{ scroll-margin-top: 1.5rem; }}
body {{
    font-family: var(--font-sans);
    color: var(--text);
    line-height: 1.5;
    background: var(--bg);
}}
.sidebar {{
    position: fixed; top: 0; left: 0; bottom: 0;
    width: var(--sidebar-width); background: {SIDEBAR_BG};
    display: flex; flex-direction: column; overflow: hidden; z-index: 200;
    border-right: 1px solid var(--border);
}}
.sidebar-header {{
    padding: 1.2rem 1rem; border-bottom: 1px solid var(--border); text-align: center;
}}
.sidebar-header svg,
.sidebar-header img {{ width: 72px; height: 72px; margin-bottom: 0.5rem; display: block; margin-left: auto; margin-right: auto; }}
.sidebar-title {{ font-size: 0.95rem; font-weight: 700; color: var(--accent); font-family: var(--font-mono); letter-spacing: 0.02em; }}
.sidebar-subtitle {{ font-size: 0.65rem; color: var(--muted); margin-top: 0.15rem; font-family: var(--font-mono); }}
.sidebar-section {{ padding: 0.8rem 1rem 0.3rem; font-size: 0.65rem; font-weight: 700;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-family: var(--font-mono); }}
.sidebar-link {{
    display: block; padding: 0.45rem 1rem; color: var(--muted);
    text-decoration: none; font-size: 0.82rem; font-family: var(--font-mono);
    border-left: 3px solid transparent;
    transition: background 0.12s, color 0.12s;
}}
.sidebar-link:hover {{ background: rgba(255,255,255,0.05); color: var(--text); }}
.sidebar-link.active {{ background: rgba(57,255,136,0.1); color: var(--accent); border-left-color: var(--accent); }}
.main-content {{ margin-left: var(--sidebar-width); min-height: 100vh; padding: 2rem 2.5rem 4rem; }}
h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; font-family: var(--font-mono); font-weight: 700; }}
h2 {{ font-size: 1.2rem; margin: 2rem 0 0.8rem; color: var(--accent); font-family: var(--font-mono);
    border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
h3 {{ font-size: 1rem; margin: 1.2rem 0 0.5rem; font-family: var(--font-mono); color: var(--accent); }}
.subtitle {{ color: var(--muted); margin-bottom: 1.5rem; font-size: 0.85rem; font-family: var(--font-mono); }}
table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.5rem; background: var(--surface);
    border-radius: var(--radius); overflow: hidden; }}
th {{ background: var(--surface); color: var(--accent); padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem;
    font-family: var(--font-mono); font-weight: 600; border-bottom: 1px solid var(--border); }}
td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); font-size: 0.82rem; font-family: var(--font-mono); }}
tr:hover td {{ background: rgba(57,255,136,0.04); }}
.env-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; margin: 1rem 0; }}
.env-card {{ background: var(--surface); border-radius: var(--radius); padding: 0.8rem 1rem;
    border: 1px solid var(--border); }}
.env-card .label {{ font-size: 0.65rem; color: var(--muted); text-transform: uppercase; font-family: var(--font-mono); letter-spacing: 0.04em; }}
.env-card .value {{ font-size: 1.1rem; font-weight: 700; color: var(--accent); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.72rem; font-weight: 600; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.badge-ok {{ background: rgba(57,255,136,0.12); color: var(--accent); }}
.badge-fail {{ background: rgba(255,92,92,0.12); color: var(--accent-fail); }}
.chart {{ margin: 0.5rem 0; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; background: var(--surface); }}
.chart summary {{ padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.85rem;
    font-family: var(--font-mono); color: var(--text); user-select: none; }}
.chart summary:hover {{ background: rgba(57,255,136,0.04); }}
.chart[open] summary {{ border-bottom: 1px solid var(--border); }}
.chart-toolbar {{
    display: flex; gap: 1rem; padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border); background: rgba(0,0,0,0.15);
    flex-wrap: wrap;
}}
.chart-toolbar label {{
    font-size: 0.78rem; color: var(--muted); font-family: var(--font-mono); display: flex; align-items: center; gap: 0.3rem;
}}
.chart-toolbar select {{
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.2rem 0.4rem; font-size: 0.78rem; font-family: var(--font-mono);
    cursor: pointer;
}}
.chart-toolbar select:hover {{ border-color: var(--accent); }}
.chart-inner {{ padding: 1rem; position: relative; }}
.chart-inner canvas {{ max-width: 100%; }}
.chart img {{ max-width: 100%; display: block; padding: 1rem; box-sizing: border-box; }}
.metrics-details summary {{ font-size:0.78rem;cursor:pointer;color:var(--accent);font-family:var(--font-mono); }}
.metrics-details[open] summary {{ margin-bottom:0.3rem; }}
.metrics-content {{ font-size:0.72rem;word-break:break-all;max-height:200px;overflow-y:auto;font-family:var(--font-mono);color:var(--muted); }}
.no-data {{ color: var(--muted); font-size: 0.85rem; font-family: var(--font-mono); padding: 1rem; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 0.72rem; font-family: var(--font-mono); text-align: center; }}
@media (max-width: 768px) {{
    .sidebar {{ width: 180px; }}
    .main-content {{ margin-left: 180px; padding: 1.5rem; }}
    .env-grid {{ grid-template-columns: 1fr 1fr; }}
}}
@media (max-width: 520px) {{
    .sidebar {{ width: 100%; position: static; border-right: none; border-bottom: 1px solid var(--border); }}
    .main-content {{ margin-left: 0; padding: 1rem; }}
    .env-grid {{ grid-template-columns: 1fr; }}
    .chart-toolbar {{ flex-direction: column; gap: 0.4rem; }}
}}
</style>
</head>
<body>
<nav class="sidebar">
    <div class="sidebar-header">
        {'<img src="nvprobe.svg" alt="nvProbe" style="width:72px;margin-bottom:0.5rem;">' if has_logo else ''}
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
    {chart_html if chart_html else '<p class="no-data">No chart data available. Run benchmarks with CUDA-enabled GPUs to generate charts.</p>'}

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
            if len(metrics_str) > 100:
                metrics_cell = (
                    f'<details class="metrics-details">'
                    f'<summary>View Raw Metrics</summary>'
                    f'<div class="metrics-content">{metrics_str}</div>'
                    f'</details>'
                )
            else:
                metrics_cell = metrics_str
            html += f"<tr><td>{r['gpu_index']}</td><td>{r['gpu_model']}</td><td>{r['precision']}</td>"
            html += f"<td>{r['batch_size']}</td><td>{status_badge}</td><td>{r.get('elapsed_seconds', '')}</td>"
            html += f"<td style='font-size:0.78rem'>{metrics_cell}</td></tr>\n"
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
    # Build comparison chart data
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

    chart_html = ""
    if all_labels and all_vals_a and all_vals_b:
        labels_json = json.dumps(all_labels)
        vals_a_json = json.dumps(all_vals_a[:len(all_labels)])
        vals_b_json = json.dumps(all_vals_b[:len(all_labels)])
        name_a = run_a["name"]
        name_b = run_b["name"]

        chart_html = f"""<details class="chart" open>
<summary>Performance Comparison</summary>
<div class="chart-inner">
  <canvas id="ch-compare"></canvas>
</div>
</details>
<script>
(function() {{
  var ctx = document.getElementById('ch-compare');
  if (!ctx || !window.Chart) return;
  new Chart(ctx.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: {labels_json},
      datasets: [
        {{
          label: '{name_a}',
          data: {vals_a_json},
          backgroundColor: '{ACCENT}99',
          borderColor: '{ACCENT}',
          borderWidth: 1,
          borderRadius: 2,
        }},
        {{
          label: '{name_b}',
          data: {vals_b_json},
          backgroundColor: '{ACCENT_WARN}99',
          borderColor: '{ACCENT_WARN}',
          borderWidth: 1,
          borderRadius: 2,
        }},
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.8,
      plugins: {{
        legend: {{ labels: {{ color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}}, boxWidth: 14, padding: 14 }} }},
        tooltip: {{
          enabled: true,
          backgroundColor: '{SURFACE}',
          titleFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 12}},
          bodyFont: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}},
          borderColor: '{BORDER}',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 6,
          titleColor: '{TEXT}',
          bodyColor: '{TEXT}',
        }},
      }},
      scales: {{
        x: {{ grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 9}}, maxRotation: 45}} }},
        y: {{ beginAtZero: true, title: {{display: true, text: 'GFLOPS', color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 11}}}}, grid: {{color: 'rgba(255,255,255,0.04)'}}, ticks: {{color: '{MUTED}', font: {{family: 'IBM Plex Mono, JetBrains Mono, Fira Code, monospace', size: 10}}}} }},
      }},
    }}
  }});
}})();
</script>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nvProbe Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
    --accent: {ACCENT};
    --accent-warn: {ACCENT_WARN};
    --accent-fail: {ACCENT_FAIL};
    --bg: {BG};
    --surface: {SURFACE};
    --text: {TEXT};
    --muted: {MUTED};
    --border: {BORDER};
    --radius: {RADIUS};
    --font-mono: 'IBM Plex Mono', 'JetBrains Mono', 'Fira Code', monospace;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: var(--font-sans);
    background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.5;
}}
h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; font-family: var(--font-mono); color: var(--accent); }}
h2 {{ font-size: 1.2rem; margin: 2rem 0 0.8rem; color: var(--accent); font-family: var(--font-mono);
    border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
.run-info {{ display: flex; gap: 1.5rem; margin: 1rem 0; flex-wrap: wrap; }}
.run-card {{ background: var(--surface); padding: 1rem; border-radius: var(--radius); flex: 1; min-width: 200px;
    border: 1px solid var(--border); }}
.run-card h3 {{ margin-bottom: 0.3rem; font-family: var(--font-mono); font-size: 1rem; }}
.run-card p {{ color: var(--muted); font-size: 0.82rem; font-family: var(--font-mono); }}
table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.5rem; background: var(--surface);
    border-radius: var(--radius); overflow: hidden; }}
th {{ background: var(--surface); color: var(--accent); padding: 0.6rem 0.8rem; text-align: left; font-size: 0.8rem;
    font-family: var(--font-mono); font-weight: 600; border-bottom: 1px solid var(--border); }}
td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); font-size: 0.82rem; font-family: var(--font-mono); }}
tr:hover td {{ background: rgba(57,255,136,0.04); }}
.chart {{ margin: 0.5rem 0; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; background: var(--surface); }}
.chart summary {{ padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.85rem;
    font-family: var(--font-mono); color: var(--text); user-select: none; }}
.chart summary:hover {{ background: rgba(57,255,136,0.04); }}
.chart[open] summary {{ border-bottom: 1px solid var(--border); }}
.chart-inner {{ padding: 1rem; }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.72rem; font-weight: 600; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.badge-ok {{ background: rgba(57,255,136,0.12); color: var(--accent); }}
.badge-fail {{ background: rgba(255,92,92,0.12); color: var(--accent-fail); }}
@media (max-width: 640px) {{
    body {{ padding: 1rem; }}
    .run-info {{ flex-direction: column; }}
}}
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
{chart_html if chart_html else '<p style="color:var(--muted);font-family:var(--font-mono)">No comparable data found between runs.</p>'}

<h2>Results A — {run_a['name']}</h2>
{_render_benchmark_tables(results_a)}

<h2>Results B — {run_b['name']}</h2>
{_render_benchmark_tables(results_b)}
</body>
</html>"""
