from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA_PATH = "data/physics_data.txt"
DEFAULT_OUTPUT_PATH = "results/physics/dashboard/contact_force_dashboard.html"
DEFAULT_ACTIVE_THRESHOLD = 0.1
DEFAULT_TRANSPOSE_INPUT = True
DEFAULT_GRID_ROWS: int | None = None
DEFAULT_GRID_COLS: int | None = None


def _fmt_js_number(value: float) -> str:
    """Compact numeric representation for embedded JavaScript arrays."""
    return f"{float(value):.8g}"


def _write_js_array(handle, name: str, values: np.ndarray, per_line: int = 10) -> None:
    flat = np.asarray(values, dtype=float).ravel()
    handle.write(f"const {name} = [\n")
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        suffix = "," if start + per_line < flat.size else ""
        handle.write("  " + ", ".join(_fmt_js_number(v) for v in chunk) + suffix + "\n")
    handle.write("];\n")


def _percentile_range(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _infer_grid_shape(node_count: int) -> tuple[int, int]:
    """Pick the closest rectangular grid for a flattened node vector."""
    best_rows, best_cols = 1, node_count
    best_gap = node_count - 1
    for candidate in range(1, int(np.sqrt(node_count)) + 1):
        if node_count % candidate == 0:
            other = node_count // candidate
            gap = abs(other - candidate)
            if gap < best_gap:
                best_rows, best_cols = candidate, other
                best_gap = gap
    return best_rows, best_cols


def _html_head(title: str, rows: int, cols: int, grid_rows: int, grid_cols: int) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-strong: #eef2f6;
      --ink: #15191f;
      --muted: #627084;
      --line: #d8dee8;
      --teal: #0f766e;
      --blue: #2563eb;
      --gold: #b7791f;
      --red: #c2410c;
      --shadow: 0 18px 45px rgba(20, 31, 45, 0.10);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .app-shell {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
    }}

    .topbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(420px, 0.95fr);
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }}

    .eyebrow {{
      margin: 0 0 4px;
      color: var(--teal);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }}

    h1, h2, h3, p {{
      margin: 0;
    }}

    h1 {{
      font-size: 32px;
      line-height: 1.05;
      letter-spacing: 0;
    }}

    h2 {{
      font-size: 16px;
      line-height: 1.2;
      letter-spacing: 0;
    }}

    h3 {{
      font-size: 14px;
      line-height: 1.25;
      letter-spacing: 0;
    }}

    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }}

    .top-controls {{
      display: grid;
      grid-template-columns: 110px minmax(160px, 1fr) 140px 140px 150px;
      gap: 10px;
      align-items: end;
    }}

    .control {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}

    .control span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}

    input[type="number"],
    select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
    }}

    input[type="range"] {{
      width: 100%;
      accent-color: var(--teal);
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}

    .metric {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(20, 31, 45, 0.05);
      min-width: 0;
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
    }}

    .metric strong {{
      display: block;
      font-size: 19px;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }}

    .dashboard-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(340px, 0.8fr);
      gap: 14px;
      align-items: start;
    }}

    .side-grid {{
      display: grid;
      gap: 14px;
    }}

    .bottom-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.72fr);
      gap: 14px;
      margin-top: 14px;
      align-items: start;
    }}

    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 13px 14px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: var(--surface-strong);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}

    .canvas-wrap {{
      padding: 12px 14px 14px;
    }}

    .heatmap-frame {{
      position: relative;
      height: 430px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #0d1218;
    }}

    #heatmapCanvas {{
      display: block;
      width: 100%;
      height: 100%;
      image-rendering: auto;
      cursor: crosshair;
    }}

    .marker {{
      position: absolute;
      pointer-events: none;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 0 0 1px rgba(15, 118, 110, 0.65);
    }}

    .marker.row {{
      left: 0;
      width: 100%;
      height: 2px;
    }}

    .marker.col {{
      top: 0;
      height: 100%;
      width: 2px;
    }}

    .axis-row {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      padding-top: 8px;
    }}

    .plot {{
      display: block;
      width: 100%;
      height: 240px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      cursor: crosshair;
    }}

    .field-frame {{
      height: 320px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #0d1218;
    }}

    .field-canvas {{
      display: block;
      width: 100%;
      height: 100%;
      image-rendering: pixelated;
      cursor: crosshair;
    }}

    .mini-select {{
      min-height: 30px;
      width: 118px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      padding: 0 8px;
      font: inherit;
      font-size: 12px;
    }}

    .selected-stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 14px 14px;
      border-top: 1px solid var(--line);
    }}

    .stat-line {{
      min-width: 0;
    }}

    .stat-line span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }}

    .stat-line strong {{
      display: block;
      font-size: 16px;
      overflow-wrap: anywhere;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    th,
    td {{
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}

    th:first-child,
    td:first-child {{
      text-align: left;
    }}

    thead th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      background: var(--surface-strong);
    }}

    tbody tr:last-child td {{
      border-bottom: 0;
    }}

    .table-wrap {{
      max-height: 314px;
      overflow: auto;
    }}

    @media (max-width: 1100px) {{
      .topbar,
      .dashboard-grid,
      .bottom-grid {{
        grid-template-columns: 1fr;
      }}

      .metrics {{
        grid-template-columns: repeat(3, minmax(130px, 1fr));
      }}
    }}

    @media (max-width: 720px) {{
      .app-shell {{
        padding: 12px;
      }}

      .top-controls,
      .metrics {{
        grid-template-columns: 1fr;
      }}

      h1 {{
        font-size: 26px;
      }}

      .heatmap-frame {{
        height: 360px;
      }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">physics_data.txt</p>
        <h1>Forces de contact</h1>
        <p class="subtitle">{rows} snapshots x {cols} nodes, grille {grid_rows} x {grid_cols}</p>
      </div>
      <div class="top-controls">
        <label class="control">
          <span>Snapshot</span>
          <input id="snapshotInput" type="number" min="0" max="{rows - 1}" step="1" />
        </label>
        <label class="control">
          <span>Index</span>
          <input id="snapshotSlider" type="range" min="0" max="{rows - 1}" step="1" />
        </label>
        <label class="control">
          <span>Grille</span>
          <select id="gridShapeSelect">
            <option value="{grid_rows}x{grid_cols}">{grid_rows} x {grid_cols}</option>
            <option value="{grid_cols}x{grid_rows}">{grid_cols} x {grid_rows}</option>
          </select>
        </label>
        <label class="control">
          <span>Ordre</span>
          <select id="nodeOrderSelect">
            <option value="row-major">row-major</option>
            <option value="column-major">column-major</option>
          </select>
        </label>
        <label class="control">
          <span>Echelle</span>
          <select id="scaleSelect">
            <option value="log">log10(abs)</option>
            <option value="sqrt">racine</option>
            <option value="linear">lineaire</option>
          </select>
        </label>
      </div>
    </header>

    <section class="metrics">
      <div class="metric"><span>Snapshots</span><strong id="metricRows"></strong></div>
      <div class="metric"><span>Nodes</span><strong id="metricCols"></strong></div>
      <div class="metric"><span>Actifs</span><strong id="metricActive"></strong></div>
      <div class="metric"><span>Force max</span><strong id="metricGlobalMax"></strong></div>
      <div class="metric"><span>Snapshot courant</span><strong id="metricSnapshot"></strong></div>
      <div class="metric"><span>Noeud courant</span><strong id="metricPoint"></strong></div>
    </section>

    <main class="dashboard-grid">
      <section class="panel">
        <div class="panel-head">
          <h2>Carte snapshots x nodes</h2>
          <span id="scaleBadge" class="badge"></span>
        </div>
        <div class="canvas-wrap">
          <div class="heatmap-frame">
            <canvas id="heatmapCanvas"></canvas>
            <div id="rowMarker" class="marker row"></div>
            <div id="colMarker" class="marker col"></div>
          </div>
          <div class="axis-row">
            <span>Snapshot 0</span>
            <span id="selectedCoords"></span>
            <span>Snapshot {rows - 1}</span>
          </div>
        </div>
      </section>

      <div class="side-grid">
        <section class="panel">
          <div class="panel-head">
            <h2>Carte 2D du snapshot</h2>
            <span id="profileBadge" class="badge"></span>
          </div>
          <div class="canvas-wrap">
            <div class="field-frame">
              <canvas id="profileCanvas" class="field-canvas"></canvas>
            </div>
          </div>
          <div class="selected-stats">
            <div class="stat-line"><span>Norme L2</span><strong id="statNorm"></strong></div>
            <div class="stat-line"><span>Moyenne</span><strong id="statMean"></strong></div>
            <div class="stat-line"><span>Max snapshot</span><strong id="statMax"></strong></div>
            <div class="stat-line"><span>Force au noeud</span><strong id="statPointForce"></strong></div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Noeuds dominants</h2>
            <span class="badge">top 10</span>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Noeud</th><th>(x, y)</th><th>Force</th><th>Part</th></tr>
              </thead>
              <tbody id="topPointsBody"></tbody>
            </table>
          </div>
        </section>
      </div>
    </main>

    <section class="bottom-grid">
      <section class="panel">
        <div class="panel-head">
          <h2>Normes par snapshot</h2>
          <span class="badge">log10</span>
        </div>
        <div class="canvas-wrap">
          <canvas id="normCanvas" class="plot"></canvas>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Enveloppe par noeud</h2>
          <select id="envelopeSelect" class="mini-select" aria-label="Statistique enveloppe">
            <option value="max">max</option>
            <option value="mean">moyenne</option>
          </select>
        </div>
        <div class="canvas-wrap">
          <div class="field-frame">
            <canvas id="pointCanvas" class="field-canvas"></canvas>
          </div>
        </div>
      </section>
    </section>
  </div>

  <script>
"""


def _html_tail() -> str:
    return r"""
const state = {
  row: DEFAULT_ROW,
  point: DEFAULT_POINT,
  scale: "log",
  gridRows: GRID_ROWS,
  gridCols: GRID_COLS,
  nodeOrder: "row-major",
  envelope: "max"
};

const els = {
  snapshotInput: document.getElementById("snapshotInput"),
  snapshotSlider: document.getElementById("snapshotSlider"),
  scaleSelect: document.getElementById("scaleSelect"),
  gridShapeSelect: document.getElementById("gridShapeSelect"),
  nodeOrderSelect: document.getElementById("nodeOrderSelect"),
  envelopeSelect: document.getElementById("envelopeSelect"),
  heatmapCanvas: document.getElementById("heatmapCanvas"),
  rowMarker: document.getElementById("rowMarker"),
  colMarker: document.getElementById("colMarker"),
  selectedCoords: document.getElementById("selectedCoords"),
  profileCanvas: document.getElementById("profileCanvas"),
  normCanvas: document.getElementById("normCanvas"),
  pointCanvas: document.getElementById("pointCanvas"),
  topPointsBody: document.getElementById("topPointsBody"),
  scaleBadge: document.getElementById("scaleBadge"),
  profileBadge: document.getElementById("profileBadge"),
  metricRows: document.getElementById("metricRows"),
  metricCols: document.getElementById("metricCols"),
  metricActive: document.getElementById("metricActive"),
  metricGlobalMax: document.getElementById("metricGlobalMax"),
  metricSnapshot: document.getElementById("metricSnapshot"),
  metricPoint: document.getElementById("metricPoint"),
  statNorm: document.getElementById("statNorm"),
  statMean: document.getElementById("statMean"),
  statMax: document.getElementById("statMax"),
  statPointForce: document.getElementById("statPointForce")
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function dataAt(row, col) {
  return DATA[row * COLS + col];
}

function fmt(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs < 0.001 || abs >= 10000) return value.toExponential(3);
  return value.toLocaleString("fr-FR", { maximumFractionDigits: 3 });
}

function fmtInt(value) {
  return Math.round(value).toLocaleString("fr-FR");
}

function transformValue(value) {
  if (state.scale === "linear") return value;
  if (state.scale === "sqrt") return Math.sqrt(Math.max(0, Math.abs(value)));
  return Math.log10(Math.abs(value) + Number.MIN_VALUE);
}

function activeRange() {
  if (state.scale === "linear") return SCALE_RANGES.linear;
  if (state.scale === "sqrt") return SCALE_RANGES.sqrt;
  return SCALE_RANGES.log;
}

function scaleLabel() {
  if (state.scale === "linear") return "force";
  if (state.scale === "sqrt") return "sqrt(abs(force))";
  return "log10(abs(force))";
}

const COLOR_STOPS = [
  [0.00, 12, 18, 28],
  [0.18, 32, 82, 149],
  [0.38, 20, 130, 137],
  [0.58, 64, 168, 95],
  [0.78, 218, 177, 55],
  [1.00, 248, 244, 220]
];

function colorFor(t) {
  const x = clamp(t, 0, 1);
  for (let i = 1; i < COLOR_STOPS.length; i++) {
    const prev = COLOR_STOPS[i - 1];
    const next = COLOR_STOPS[i];
    if (x <= next[0]) {
      const local = (x - prev[0]) / (next[0] - prev[0] || 1);
      return [
        Math.round(prev[1] + (next[1] - prev[1]) * local),
        Math.round(prev[2] + (next[2] - prev[2]) * local),
        Math.round(prev[3] + (next[3] - prev[3]) * local)
      ];
    }
  }
  return [248, 244, 220];
}

function gridToNodeIndex(x, y) {
  if (state.nodeOrder === "column-major") {
    return clamp(x * state.gridRows + y, 0, COLS - 1);
  }
  return clamp(y * state.gridCols + x, 0, COLS - 1);
}

function nodeIndexToGrid(index) {
  const node = clamp(index, 0, COLS - 1);
  if (state.nodeOrder === "column-major") {
    return {
      x: Math.floor(node / state.gridRows),
      y: node % state.gridRows
    };
  }
  return {
    x: node % state.gridCols,
    y: Math.floor(node / state.gridCols)
  };
}

function drawFieldMap(canvas, values, options = {}) {
  canvas.width = state.gridCols;
  canvas.height = state.gridRows;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(state.gridCols, state.gridRows);
  const range = options.range || activeRange();
  const lo = range[0];
  const span = range[1] - range[0] || 1;

  for (let y = 0; y < state.gridRows; y++) {
    for (let x = 0; x < state.gridCols; x++) {
      const node = gridToNodeIndex(x, y);
      const t = (transformValue(values[node]) - lo) / span;
      const color = colorFor(t);
      const offset = (y * state.gridCols + x) * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);

  const selected = nodeIndexToGrid(state.point);
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 1;
  ctx.strokeRect(selected.x + 0.5, selected.y + 0.5, 1, 1);
  ctx.strokeStyle = "rgba(194,65,12,0.95)";
  ctx.strokeRect(selected.x - 1.5, selected.y - 1.5, 4, 4);
}

function drawHeatmap() {
  const canvas = els.heatmapCanvas;
  canvas.width = COLS;
  canvas.height = ROWS;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(COLS, ROWS);
  const range = activeRange();
  const lo = range[0];
  const span = range[1] - range[0] || 1;

  for (let i = 0; i < DATA.length; i++) {
    const t = (transformValue(DATA[i]) - lo) / span;
    const color = colorFor(t);
    const offset = i * 4;
    image.data[offset] = color[0];
    image.data[offset + 1] = color[1];
    image.data[offset + 2] = color[2];
    image.data[offset + 3] = 255;
  }

  ctx.putImageData(image, 0, 0);
  updateMarkers();
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawAxes(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    const y = (height * i) / 5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawLinePlot(canvas, values, options) {
  const { ctx, width, height } = resizeCanvas(canvas);
  const pad = { left: 42, right: 18, top: 16, bottom: 28 };
  const plotW = Math.max(1, width - pad.left - pad.right);
  const plotH = Math.max(1, height - pad.top - pad.bottom);
  const color = options.color || "#0f766e";
  const transform = options.transform || ((v) => v);
  const series = values.map(transform);
  let yMin = options.yMin;
  let yMax = options.yMax;
  if (yMin === undefined || yMax === undefined) {
    yMin = Math.min(...series);
    yMax = Math.max(...series);
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
    yMin = 0;
    yMax = 1;
  }

  drawAxes(ctx, width, height);
  ctx.strokeStyle = "#9aa7b8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);

  ctx.beginPath();
  series.forEach((value, index) => {
    const x = pad.left + (index / Math.max(1, series.length - 1)) * plotW;
    const y = pad.top + (1 - (value - yMin) / (yMax - yMin || 1)) * plotH;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  if (options.markerIndex !== undefined) {
    const markerX = pad.left + (options.markerIndex / Math.max(1, series.length - 1)) * plotW;
    ctx.strokeStyle = options.markerColor || "#c2410c";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(markerX, pad.top);
    ctx.lineTo(markerX, pad.top + plotH);
    ctx.stroke();
  }

  if (options.pointIndex !== undefined) {
    const pointX = pad.left + (options.pointIndex / Math.max(1, series.length - 1)) * plotW;
    const value = series[options.pointIndex];
    const pointY = pad.top + (1 - (value - yMin) / (yMax - yMin || 1)) * plotH;
    ctx.fillStyle = options.markerColor || "#c2410c";
    ctx.beginPath();
    ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#627084";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(options.leftLabel || "", 8, pad.top + 11);
  ctx.fillText(options.bottomLabel || "", pad.left, height - 8);

  return { pad, plotW, plotH };
}

function drawMultiLinePlot(canvas, seriesList) {
  const { ctx, width, height } = resizeCanvas(canvas);
  const pad = { left: 42, right: 18, top: 16, bottom: 28 };
  const plotW = Math.max(1, width - pad.left - pad.right);
  const plotH = Math.max(1, height - pad.top - pad.bottom);
  const allValues = seriesList.flatMap((s) => s.values.map(s.transform || ((v) => v)));
  let yMin = Math.min(...allValues);
  let yMax = Math.max(...allValues);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
    yMin = 0;
    yMax = 1;
  }

  drawAxes(ctx, width, height);
  ctx.strokeStyle = "#9aa7b8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);

  seriesList.forEach((series) => {
    const transform = series.transform || ((v) => v);
    ctx.beginPath();
    series.values.forEach((raw, index) => {
      const value = transform(raw);
      const x = pad.left + (index / Math.max(1, series.values.length - 1)) * plotW;
      const y = pad.top + (1 - (value - yMin) / (yMax - yMin || 1)) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.width || 2;
    ctx.stroke();
  });

  ctx.fillStyle = "#0f766e";
  ctx.fillRect(pad.left, 10, 10, 3);
  ctx.fillStyle = "#627084";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("max", pad.left + 15, 14);
  ctx.fillStyle = "#b7791f";
  ctx.fillRect(pad.left + 58, 10, 10, 3);
  ctx.fillStyle = "#627084";
  ctx.fillText("moyenne", pad.left + 73, 14);
}

function currentRowValues() {
  const offset = state.row * COLS;
  return DATA.slice(offset, offset + COLS);
}

function updateMarkers() {
  els.rowMarker.style.top = `${((state.row + 0.5) / ROWS) * 100}%`;
  els.colMarker.style.left = `${((state.point + 0.5) / COLS) * 100}%`;
}

function updateMetrics() {
  const values = currentRowValues();
  const pointValue = dataAt(state.row, state.point);
  const gridPoint = nodeIndexToGrid(state.point);
  els.metricRows.textContent = fmtInt(ROWS);
  els.metricCols.textContent = fmtInt(COLS);
  els.metricActive.textContent = fmtInt(ACTIVE_COUNT);
  els.metricGlobalMax.textContent = fmt(GLOBAL_MAX);
  els.metricSnapshot.textContent = `#${state.row}`;
  els.metricPoint.textContent = `#${state.point}`;
  els.statNorm.textContent = fmt(ROW_NORMS[state.row]);
  els.statMean.textContent = fmt(ROW_MEANS[state.row]);
  els.statMax.textContent = fmt(Math.max(...values));
  els.statPointForce.textContent = fmt(pointValue);
  els.selectedCoords.textContent = `snapshot ${state.row} / node ${state.point} / (${gridPoint.x}, ${gridPoint.y}) / ${fmt(pointValue)}`;
  els.profileBadge.textContent = `${state.gridRows} x ${state.gridCols} | node ${state.point}`;
  els.scaleBadge.textContent = scaleLabel();
}

function updateTopTable() {
  const values = currentRowValues();
  const total = values.reduce((acc, v) => acc + Math.abs(v), 0) || 1;
  const top = values
    .map((value, point) => ({ point, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  els.topPointsBody.innerHTML = top.map((item) => {
    const share = (Math.abs(item.value) / total) * 100;
    const isSelected = item.point === state.point;
    const coord = nodeIndexToGrid(item.point);
    return `<tr style="${isSelected ? "background:#fff7ed" : ""}">
      <td>#${item.point}</td>
      <td>(${coord.x}, ${coord.y})</td>
      <td>${fmt(item.value)}</td>
      <td>${share.toFixed(2)}%</td>
    </tr>`;
  }).join("");
}

function drawProfile() {
  const values = currentRowValues();
  drawFieldMap(els.profileCanvas, values, { range: activeRange() });
}

function drawNorms() {
  const logs = ROW_NORMS.map((v) => Math.log10(Math.abs(v) + Number.MIN_VALUE));
  drawLinePlot(els.normCanvas, ROW_NORMS, {
    color: "#2563eb",
    transform: (v) => Math.log10(Math.abs(v) + Number.MIN_VALUE),
    yMin: Math.min(...logs),
    yMax: Math.max(...logs),
    markerIndex: state.row,
    markerColor: "#c2410c",
    leftLabel: "log10(norme)",
    bottomLabel: "snapshot"
  });
}

function drawPointEnvelope() {
  const values = state.envelope === "mean" ? POINT_MEAN : POINT_MAX;
  drawFieldMap(els.pointCanvas, values, { range: activeRange() });
}

function updateAll(redrawHeatmap = false) {
  els.snapshotInput.value = state.row;
  els.snapshotSlider.value = state.row;
  els.scaleSelect.value = state.scale;
  els.gridShapeSelect.value = `${state.gridRows}x${state.gridCols}`;
  els.nodeOrderSelect.value = state.nodeOrder;
  els.envelopeSelect.value = state.envelope;
  updateMetrics();
  updateTopTable();
  updateMarkers();
  drawProfile();
  drawNorms();
  drawPointEnvelope();
  if (redrawHeatmap) drawHeatmap();
}

function setRow(row) {
  state.row = clamp(Number.parseInt(row, 10) || 0, 0, ROWS - 1);
  const values = currentRowValues();
  state.point = values.reduce((best, value, index) => {
    return Math.abs(value) > Math.abs(values[best]) ? index : best;
  }, state.point);
  updateAll(false);
}

function setPoint(point) {
  state.point = clamp(Number.parseInt(point, 10) || 0, 0, COLS - 1);
  updateAll(false);
}

function setGridShape(value) {
  const parts = value.split("x").map((item) => Number.parseInt(item, 10));
  if (parts.length === 2 && parts[0] * parts[1] === COLS) {
    state.gridRows = parts[0];
    state.gridCols = parts[1];
  }
  updateAll(false);
}

function canvasPosition(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
    y: clamp((event.clientY - rect.top) / rect.height, 0, 1)
  };
}

els.snapshotSlider.addEventListener("input", (event) => setRow(event.target.value));
els.snapshotInput.addEventListener("change", (event) => setRow(event.target.value));
els.scaleSelect.addEventListener("change", (event) => {
  state.scale = event.target.value;
  updateAll(true);
});

els.gridShapeSelect.addEventListener("change", (event) => setGridShape(event.target.value));
els.nodeOrderSelect.addEventListener("change", (event) => {
  state.nodeOrder = event.target.value;
  updateAll(false);
});
els.envelopeSelect.addEventListener("change", (event) => {
  state.envelope = event.target.value;
  updateAll(false);
});

els.heatmapCanvas.addEventListener("click", (event) => {
  const pos = canvasPosition(event, els.heatmapCanvas);
  state.row = clamp(Math.floor(pos.y * ROWS), 0, ROWS - 1);
  state.point = clamp(Math.floor(pos.x * COLS), 0, COLS - 1);
  updateAll(false);
});

els.profileCanvas.addEventListener("click", (event) => {
  const pos = canvasPosition(event, els.profileCanvas);
  const x = clamp(Math.floor(pos.x * state.gridCols), 0, state.gridCols - 1);
  const y = clamp(Math.floor(pos.y * state.gridRows), 0, state.gridRows - 1);
  setPoint(gridToNodeIndex(x, y));
});

els.pointCanvas.addEventListener("click", (event) => {
  const pos = canvasPosition(event, els.pointCanvas);
  const x = clamp(Math.floor(pos.x * state.gridCols), 0, state.gridCols - 1);
  const y = clamp(Math.floor(pos.y * state.gridRows), 0, state.gridRows - 1);
  setPoint(gridToNodeIndex(x, y));
});

els.normCanvas.addEventListener("click", (event) => {
  const pos = canvasPosition(event, els.normCanvas);
  setRow(Math.round(pos.x * (ROWS - 1)));
});

window.addEventListener("resize", () => updateAll(false));

drawHeatmap();
updateAll(false);
</script>
</body>
</html>
"""


def build_dashboard(
    data_path: Path,
    output_path: Path,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    transpose_input: bool = DEFAULT_TRANSPOSE_INPUT,
    grid_rows: int | None = DEFAULT_GRID_ROWS,
    grid_cols: int | None = DEFAULT_GRID_COLS,
) -> None:
    raw = np.loadtxt(data_path)
    matrix = raw.T if transpose_input else raw
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {matrix.shape}")

    rows, cols = matrix.shape
    if grid_rows is None or grid_cols is None:
        inferred_rows, inferred_cols = _infer_grid_shape(cols)
        grid_rows = inferred_rows if grid_rows is None else grid_rows
        grid_cols = inferred_cols if grid_cols is None else grid_cols
    if grid_rows * grid_cols != cols:
        raise ValueError(
            f"grid_rows * grid_cols must equal the node count {cols}, "
            f"got {grid_rows} * {grid_cols}"
        )

    row_norms = np.linalg.norm(matrix, axis=1)
    row_means = np.mean(matrix, axis=1)
    point_max = np.max(np.abs(matrix), axis=0)
    point_mean = np.mean(np.abs(matrix), axis=0)
    active_count = int(np.count_nonzero(row_norms > active_threshold))

    default_row = int(np.argmax(row_norms))
    default_point = int(np.argmax(np.abs(matrix[default_row])))

    abs_matrix = np.abs(matrix)
    log_values = np.log10(abs_matrix + np.finfo(float).tiny)
    sqrt_values = np.sqrt(abs_matrix)
    linear_lo, linear_hi = _percentile_range(matrix, 0.5, 99.5)
    log_lo, log_hi = _percentile_range(log_values, 1.0, 99.5)
    sqrt_lo, sqrt_hi = _percentile_range(sqrt_values, 0.5, 99.5)

    title = "Contact Force Dashboard"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(_html_head(title, rows, cols, grid_rows, grid_cols))
        handle.write(f"const ROWS = {rows};\n")
        handle.write(f"const COLS = {cols};\n")
        handle.write(f"const GRID_ROWS = {grid_rows};\n")
        handle.write(f"const GRID_COLS = {grid_cols};\n")
        handle.write(f"const ACTIVE_COUNT = {active_count};\n")
        handle.write(f"const ACTIVE_THRESHOLD = {_fmt_js_number(active_threshold)};\n")
        handle.write(f"const GLOBAL_MIN = {_fmt_js_number(float(np.min(matrix)))};\n")
        handle.write(f"const GLOBAL_MAX = {_fmt_js_number(float(np.max(matrix)))};\n")
        handle.write(f"const DEFAULT_ROW = {default_row};\n")
        handle.write(f"const DEFAULT_POINT = {default_point};\n")
        handle.write("const SCALE_RANGES = {\n")
        handle.write(
            "  linear: "
            f"[{_fmt_js_number(linear_lo)}, {_fmt_js_number(linear_hi)}],\n"
        )
        handle.write(
            "  log: "
            f"[{_fmt_js_number(log_lo)}, {_fmt_js_number(log_hi)}],\n"
        )
        handle.write(
            "  sqrt: "
            f"[{_fmt_js_number(sqrt_lo)}, {_fmt_js_number(sqrt_hi)}]\n"
        )
        handle.write("};\n")
        _write_js_array(handle, "DATA", matrix, per_line=8)
        _write_js_array(handle, "ROW_NORMS", row_norms, per_line=8)
        _write_js_array(handle, "ROW_MEANS", row_means, per_line=8)
        _write_js_array(handle, "POINT_MAX", point_max, per_line=8)
        _write_js_array(handle, "POINT_MEAN", point_mean, per_line=8)
        handle.write(_html_tail())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained contact-force dashboard."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Input matrix file.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="Output HTML file.")
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=DEFAULT_ACTIVE_THRESHOLD,
        help="L2-norm threshold used for the active snapshot count.",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Use the input matrix as snapshots x nodes instead of transposing it.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=DEFAULT_GRID_ROWS,
        help="Number of rows in the 2-D node grid. Defaults to inferred factor.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=DEFAULT_GRID_COLS,
        help="Number of columns in the 2-D node grid. Defaults to inferred factor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dashboard(
        data_path=Path(args.data),
        output_path=Path(args.out),
        active_threshold=args.active_threshold,
        transpose_input=not args.no_transpose,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    print(f"Dashboard written to {args.out}")


if __name__ == "__main__":
    main()
