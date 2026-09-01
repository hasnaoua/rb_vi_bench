from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASIS_PATH = "results/lambda/cpg/cpg_basis.npy"
DEFAULT_SELECTED_PATH = "results/lambda/cpg/cpg_selected_indices.txt"
DEFAULT_HTML_PATH = "results/lambda/cpg/cpg_output_snapshots_cylinder_88x88.html"
DEFAULT_PNG_PATH = "results/lambda/cpg/cpg_output_snapshots_cylinder_88x88.png"
DEFAULT_GRID_SHAPE = (88, 88)


def _fmt_js_number(value: float) -> str:
    if np.isnan(value):
        return "null"
    return f"{float(value):.6g}"


def _read_selected_indices(path: Path) -> list[int]:
    if not path.exists():
        return []

    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            labels.append(int(line))
    return labels


def _pad_matrix(matrix: np.ndarray, target_nodes: int) -> np.ndarray:
    if matrix.shape[1] > target_nodes:
        raise ValueError(
            f"Cannot fit {matrix.shape[1]} nodes into a grid with {target_nodes} cells"
        )
    if matrix.shape[1] == target_nodes:
        return matrix

    padded = np.full((matrix.shape[0], target_nodes), np.nan, dtype=float)
    padded[:, : matrix.shape[1]] = matrix
    return padded


def _normalise_global(padded: np.ndarray) -> tuple[np.ndarray, float, float]:
    log_values = np.log10(np.abs(padded) + np.finfo(float).tiny)
    finite = log_values[np.isfinite(log_values)]
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return np.clip((log_values - vmin) / (vmax - vmin), 0.0, 1.0), float(vmin), float(vmax)


def _normalise_local(padded: np.ndarray) -> np.ndarray:
    log_values = np.log10(np.abs(padded) + np.finfo(float).tiny)
    mins = np.nanmin(log_values, axis=1, keepdims=True)
    maxs = np.nanmax(log_values, axis=1, keepdims=True)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    return np.clip((log_values - mins) / spans, 0.0, 1.0)


def _write_js_array(handle, name: str, values: np.ndarray, per_line: int = 12) -> None:
    flat = np.asarray(values, dtype=float).ravel()
    handle.write(f"const {name} = [\n")
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        suffix = "," if start + per_line < flat.size else ""
        handle.write("  " + ", ".join(_fmt_js_number(v) for v in chunk) + suffix + "\n")
    handle.write("];\n")


def _html_head(
    snapshot_count: int,
    grid_rows: int,
    grid_cols: int,
    original_nodes: int,
    padded_nodes: int,
    labels: list[int],
    log_min: float,
    log_max: float,
    raw_max: float,
) -> str:
    label_text = "[" + ", ".join(str(int(v)) for v in labels) + "]"
    missing = padded_nodes - original_nodes
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CPG Cylinder 88x88</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --surface: #ffffff;
      --ink: #141820;
      --muted: #617084;
      --line: #d8dee8;
      --teal: #0f766e;
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

    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px;
    }}

    .topbar {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(520px, 1.15fr);
      gap: 18px;
      align-items: end;
      margin-bottom: 14px;
    }}

    h1, p {{
      margin: 0;
    }}

    h1 {{
      font-size: 30px;
      line-height: 1.08;
      letter-spacing: 0;
    }}

    .subtitle {{
      margin-top: 7px;
      color: var(--muted);
    }}

    .controls {{
      display: grid;
      grid-template-columns: 120px 1fr 1fr 1fr;
      gap: 10px;
      align-items: end;
    }}

    label {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}

    label span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}

    input {{
      width: 100%;
      accent-color: var(--teal);
    }}

    input[type="number"] {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
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
      font-weight: 800;
      margin-bottom: 5px;
    }}

    .metric strong {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }}

    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
    }}

    .panel-head strong {{
      color: var(--muted);
      font-size: 12px;
    }}

    .canvas-wrap {{
      padding: 12px;
    }}

    canvas {{
      display: block;
      width: 100%;
      height: 680px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0d1218;
      cursor: crosshair;
    }}

    @media (max-width: 900px) {{
      .topbar,
      .controls,
      .metrics {{
        grid-template-columns: 1fr;
      }}

      canvas {{
        height: 460px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div>
        <h1>Snapshots CPG sur cylindre</h1>
        <p class="subtitle">{snapshot_count} snapshots, grille {grid_rows} x {grid_cols}; {original_nodes} valeurs + {missing} cases masquees</p>
      </div>
      <div class="controls">
        <label>
          <span>Snapshot</span>
          <input id="snapshotInput" type="number" min="0" max="{snapshot_count - 1}" step="1" />
        </label>
        <label>
          <span>Index</span>
          <input id="snapshotSlider" type="range" min="0" max="{snapshot_count - 1}" step="1" />
        </label>
        <label>
          <span>Rotation</span>
          <input id="yawSlider" type="range" min="-180" max="180" step="1" value="-42" />
        </label>
        <label>
          <span>Relief radial</span>
          <input id="reliefSlider" type="range" min="0" max="70" step="1" value="18" />
        </label>
      </div>
    </header>

    <section class="metrics">
      <div class="metric"><span>Snapshot CPG</span><strong id="metricSnapshot"></strong></div>
      <div class="metric"><span>Snapshot original</span><strong id="metricLabel"></strong></div>
      <div class="metric"><span>Noeud max</span><strong id="metricMaxNode"></strong></div>
      <div class="metric"><span>Force max</span><strong id="metricMaxForce"></strong></div>
      <div class="metric"><span>Cases masquees</span><strong>{missing}</strong></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Carte de force sur cylindre</h2>
        <strong id="rangeLabel"></strong>
      </div>
      <div class="canvas-wrap">
        <canvas id="surfaceCanvas"></canvas>
      </div>
    </section>
  </div>

  <script>
const SNAPSHOT_COUNT = {snapshot_count};
const GRID_ROWS = {grid_rows};
const GRID_COLS = {grid_cols};
const NODE_COUNT = GRID_ROWS * GRID_COLS;
const ORIGINAL_NODES = {original_nodes};
const LOG_MIN = {_fmt_js_number(log_min)};
const LOG_MAX = {_fmt_js_number(log_max)};
const RAW_MAX = {_fmt_js_number(raw_max)};
const LABELS = {label_text};
"""


def _html_tail() -> str:
    return r"""

const state = {
  snapshot: 0,
  yaw: -42,
  relief: 0.18
};

const els = {
  canvas: document.getElementById("surfaceCanvas"),
  snapshotInput: document.getElementById("snapshotInput"),
  snapshotSlider: document.getElementById("snapshotSlider"),
  yawSlider: document.getElementById("yawSlider"),
  reliefSlider: document.getElementById("reliefSlider"),
  metricSnapshot: document.getElementById("metricSnapshot"),
  metricLabel: document.getElementById("metricLabel"),
  metricMaxNode: document.getElementById("metricMaxNode"),
  metricMaxForce: document.getElementById("metricMaxForce"),
  rangeLabel: document.getElementById("rangeLabel")
};

const COLOR_STOPS = [
  [0.00, 12, 18, 28],
  [0.18, 32, 82, 149],
  [0.38, 20, 130, 137],
  [0.58, 64, 168, 95],
  [0.78, 218, 177, 55],
  [1.00, 248, 244, 220]
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function fmt(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs < 0.001 || abs >= 10000) return value.toExponential(3);
  return value.toLocaleString("fr-FR", { maximumFractionDigits: 3 });
}

function colorFor(t) {
  const x = clamp(t, 0, 1);
  for (let i = 1; i < COLOR_STOPS.length; i++) {
    const prev = COLOR_STOPS[i - 1];
    const next = COLOR_STOPS[i];
    if (x <= next[0]) {
      const local = (x - prev[0]) / (next[0] - prev[0] || 1);
      return `rgb(${Math.round(prev[1] + (next[1] - prev[1]) * local)}, ${Math.round(prev[2] + (next[2] - prev[2]) * local)}, ${Math.round(prev[3] + (next[3] - prev[3]) * local)})`;
    }
  }
  return "rgb(248,244,220)";
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  els.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = els.canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function snapshotOffset() {
  return state.snapshot * NODE_COUNT;
}

function valueAt(snapshot, node) {
  return Z_DATA[snapshot * NODE_COUNT + node];
}

function heightAt(snapshot, node) {
  return H_DATA[snapshot * NODE_COUNT + node];
}

function rawAt(snapshot, node) {
  const z = valueAt(snapshot, node);
  const logValue = LOG_MIN + z * (LOG_MAX - LOG_MIN);
  return Math.pow(10, logValue);
}

function isValid(node) {
  return node < ORIGINAL_NODES && Number.isFinite(valueAt(state.snapshot, node));
}

function projectCylinder(row, col, colorValue, heightValue, width, height, scale, yawRad, pitchRad) {
  const theta = (col / GRID_COLS) * Math.PI * 2;
  const axial = (row / Math.max(1, GRID_ROWS - 1) - 0.5) * 2.15;
  const radius = 1.0 + state.relief * heightValue;
  let x = Math.cos(theta) * radius;
  let y = axial;
  let z = Math.sin(theta) * radius;

  const cosYaw = Math.cos(yawRad);
  const sinYaw = Math.sin(yawRad);
  const xr = x * cosYaw - z * sinYaw;
  const zr = x * sinYaw + z * cosYaw;

  const cosPitch = Math.cos(pitchRad);
  const sinPitch = Math.sin(pitchRad);
  const yr = y * cosPitch - zr * sinPitch;
  const depth = y * sinPitch + zr * cosPitch;

  return {
    x: width / 2 + xr * scale,
    y: height * 0.53 + yr * scale,
    depth
  };
}

function currentColorValues() {
  return Z_DATA.slice(snapshotOffset(), snapshotOffset() + NODE_COUNT);
}

function currentHeightValues() {
  return H_DATA.slice(snapshotOffset(), snapshotOffset() + NODE_COUNT);
}

function draw() {
  const { ctx, width, height } = resizeCanvas();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0d1218";
  ctx.fillRect(0, 0, width, height);

  const colors = currentColorValues();
  const heights = currentHeightValues();
  const yawRad = (state.yaw * Math.PI) / 180;
  const pitchRad = (18 * Math.PI) / 180;
  const scale = Math.min(width * 0.31, height * 0.34);
  const points = new Array(NODE_COUNT);

  for (let row = 0; row < GRID_ROWS; row++) {
    for (let col = 0; col < GRID_COLS; col++) {
      const node = row * GRID_COLS + col;
      if (!isValid(node)) continue;
      points[node] = projectCylinder(
        row,
        col,
        colors[node],
        heights[node],
        width,
        height,
        scale,
        yawRad,
        pitchRad
      );
    }
  }

  const cells = [];
  for (let row = 0; row < GRID_ROWS - 1; row++) {
    for (let col = 0; col < GRID_COLS; col++) {
      const nextCol = (col + 1) % GRID_COLS;
      const i0 = row * GRID_COLS + col;
      const i1 = row * GRID_COLS + nextCol;
      const i2 = (row + 1) * GRID_COLS + nextCol;
      const i3 = (row + 1) * GRID_COLS + col;
      if (!points[i0] || !points[i1] || !points[i2] || !points[i3]) continue;
      const avg = (colors[i0] + colors[i1] + colors[i2] + colors[i3]) / 4;
      const depth = (points[i0].depth + points[i1].depth + points[i2].depth + points[i3].depth) / 4;
      cells.push({ i0, i1, i2, i3, avg, depth });
    }
  }

  cells.sort((a, b) => a.depth - b.depth);
  ctx.lineWidth = 0.25;
  ctx.strokeStyle = "rgba(255,255,255,0.055)";

  for (const cell of cells) {
    const p0 = points[cell.i0];
    const p1 = points[cell.i1];
    const p2 = points[cell.i2];
    const p3 = points[cell.i3];
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.lineTo(p3.x, p3.y);
    ctx.closePath();
    ctx.fillStyle = colorFor(cell.avg);
    ctx.fill();
    ctx.stroke();
  }

  const validColors = colors.slice(0, ORIGINAL_NODES);
  const maxNode = validColors.reduce((best, value, index) => value > validColors[best] ? index : best, 0);
  const maxPoint = points[maxNode];
  if (maxPoint) {
    ctx.fillStyle = "rgba(255,255,255,0.96)";
    ctx.strokeStyle = "rgba(194,65,12,0.96)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(maxPoint.x, maxPoint.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  const gx = maxNode % GRID_COLS;
  const gy = Math.floor(maxNode / GRID_COLS);
  els.metricSnapshot.textContent = `#${state.snapshot + 1} / ${SNAPSHOT_COUNT}`;
  els.metricLabel.textContent = `snap #${LABELS[state.snapshot] ?? state.snapshot}`;
  els.metricMaxNode.textContent = `#${maxNode} (${gx}, ${gy})`;
  els.metricMaxForce.textContent = fmt(rawAt(state.snapshot, maxNode));
  els.rangeLabel.textContent = `couleur log globale ${LOG_MIN.toFixed(2)} -> ${LOG_MAX.toFixed(2)}, max ${fmt(RAW_MAX)}`;

  els.snapshotInput.value = state.snapshot;
  els.snapshotSlider.value = state.snapshot;
  els.yawSlider.value = state.yaw;
  els.reliefSlider.value = Math.round(state.relief * 100);
}

function setSnapshot(value) {
  state.snapshot = clamp(Number.parseInt(value, 10) || 0, 0, SNAPSHOT_COUNT - 1);
  draw();
}

els.snapshotInput.addEventListener("change", (event) => setSnapshot(event.target.value));
els.snapshotSlider.addEventListener("input", (event) => setSnapshot(event.target.value));
els.yawSlider.addEventListener("input", (event) => {
  state.yaw = Number.parseFloat(event.target.value);
  draw();
});
els.reliefSlider.addEventListener("input", (event) => {
  state.relief = Number.parseFloat(event.target.value) / 100;
  draw();
});
window.addEventListener("resize", draw);

draw();
</script>
</body>
</html>
"""


def write_interactive_html(
    basis: np.ndarray,
    labels: list[int],
    output_path: Path,
    grid_shape: tuple[int, int],
) -> None:
    target_nodes = grid_shape[0] * grid_shape[1]
    padded = _pad_matrix(basis, target_nodes)
    colors, log_min, log_max = _normalise_global(padded)
    heights = _normalise_local(padded)

    label_values = labels if len(labels) == basis.shape[0] else list(range(basis.shape[0]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            _html_head(
                snapshot_count=basis.shape[0],
                grid_rows=grid_shape[0],
                grid_cols=grid_shape[1],
                original_nodes=basis.shape[1],
                padded_nodes=target_nodes,
                labels=label_values,
                log_min=log_min,
                log_max=log_max,
                raw_max=float(np.nanmax(basis)),
            )
        )
        _write_js_array(handle, "Z_DATA", colors, per_line=12)
        _write_js_array(handle, "H_DATA", heights, per_line=12)
        handle.write(_html_tail())


def write_static_png(
    basis: np.ndarray,
    labels: list[int],
    output_path: Path,
    grid_shape: tuple[int, int],
    max_surfaces: int = 12,
) -> None:
    target_nodes = grid_shape[0] * grid_shape[1]
    padded = _pad_matrix(basis, target_nodes)
    colors, log_min, log_max = _normalise_global(padded)
    heights = _normalise_local(padded)

    grid_rows, grid_cols = grid_shape
    theta = np.linspace(0.0, 2.0 * np.pi, grid_cols, endpoint=False)
    axial = np.linspace(-1.0, 1.0, grid_rows)
    theta_grid, axial_grid = np.meshgrid(theta, axial)

    n_surfaces = min(max_surfaces, len(basis))
    ncols = min(4, n_surfaces)
    nrows = int(np.ceil(n_surfaces / ncols))
    fig = plt.figure(figsize=(4.1 * ncols, 3.4 * nrows), constrained_layout=True)

    for idx in range(n_surfaces):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        h_grid = heights[idx].reshape(grid_shape)
        c_grid = colors[idx].reshape(grid_shape)
        radius = 1.0 + 0.18 * h_grid
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        z_grid = axial_grid * 1.5

        invalid = np.isnan(c_grid)
        x_grid = np.where(invalid, np.nan, x_grid)
        y_grid = np.where(invalid, np.nan, y_grid)
        z_grid = np.where(invalid, np.nan, z_grid)

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            facecolors=plt.cm.viridis(np.nan_to_num(c_grid, nan=0.0)),
            linewidth=0,
            antialiased=False,
            rstride=2,
            cstride=2,
            shade=False,
        )
        label = labels[idx] if idx < len(labels) else idx
        ax.set_title(f"basis {idx + 1} | snap #{label}", fontsize=9)
        ax.view_init(elev=22, azim=-56)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])  # type: ignore[attr-defined]  # Axes3D wrapper, not typed as a method
        ax.set_box_aspect((1, 1, 1.6))

    fig.suptitle(
        f"CPG output snapshots on 88 x 88 cylinder "
        f"({target_nodes - basis.shape[1]} masked cells, color log {log_min:.2f} -> {log_max:.2f})"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise CPG output snapshots on a padded cylindrical grid."
    )
    parser.add_argument("--basis", default=DEFAULT_BASIS_PATH, help="Input CPG basis .npy file.")
    parser.add_argument(
        "--selected",
        default=DEFAULT_SELECTED_PATH,
        help="Selected snapshot index text file.",
    )
    parser.add_argument("--html", default=DEFAULT_HTML_PATH, help="Output interactive HTML.")
    parser.add_argument("--png", default=DEFAULT_PNG_PATH, help="Output static PNG.")
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_SHAPE[0])
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_SHAPE[1])
    parser.add_argument("--max-surfaces", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basis = np.load(args.basis)
    if basis.ndim != 2:
        raise ValueError(f"Expected basis to be 2-D, got shape {basis.shape}")

    grid_shape = (args.grid_rows, args.grid_cols)
    labels = _read_selected_indices(Path(args.selected))
    write_interactive_html(basis, labels, Path(args.html), grid_shape)
    write_static_png(
        basis,
        labels,
        Path(args.png),
        grid_shape,
        max_surfaces=args.max_surfaces,
    )
    missing = grid_shape[0] * grid_shape[1] - basis.shape[1]
    print(f"Interactive cylinder HTML written to {args.html}")
    print(f"Static cylinder PNG written to {args.png}")
    print(f"Grid {grid_shape[0]} x {grid_shape[1]} with {missing} masked cells")


if __name__ == "__main__":
    main()
