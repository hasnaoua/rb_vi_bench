from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ARCHIVE = Path("data/FEM_SOLS.zip")
DEFAULT_DIRECTORY = Path("data/FEM_SOLS")
DEFAULT_OUTPUT_DIR = Path("results/lambda/dataset")
RADIUS_PATTERN = re.compile(
    r"^rad=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@dataclass(frozen=True)
class SnapshotRecord:
    radius: float
    radius_label: str
    source: str
    original_shape: tuple[int, ...]
    values: np.ndarray


def _parse_radius_from_parts(parts: tuple[str, ...]) -> tuple[float, str] | None:
    for part in parts:
        match = RADIUS_PATTERN.match(part)
        if match:
            label = match.group(1)
            return float(label), label
    return None


def _load_npy_bytes(data: bytes) -> np.ndarray:
    return np.load(BytesIO(data), allow_pickle=False)


def _flatten_snapshot(array: np.ndarray, source: str) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(array, dtype=float)
    original_shape = tuple(values.shape)
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError(f"{source} is empty")
    if not np.all(np.isfinite(values)):
        bad_count = int(values.size - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"{source} contains {bad_count} non-finite values")
    return values, original_shape


def _choose_lambda_file(files: dict[str, str | Path]) -> tuple[str, str | Path]:
    if "npy" in files:
        return "npy", files["npy"]
    if "txt" in files:
        return "txt", files["txt"]
    raise ValueError("Missing lambda.npy/lambda.txt entry")


def load_records_from_zip(path: Path) -> list[SnapshotRecord]:
    groups: dict[float, dict[str, str | Path | float]] = {}

    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not (name.endswith("/lambda.npy") or name.endswith("/lambda.txt")):
                continue

            radius_info = _parse_radius_from_parts(PurePosixPath(name).parts)
            if radius_info is None:
                continue

            radius, label = radius_info
            suffix = "npy" if name.endswith(".npy") else "txt"
            group = groups.setdefault(radius, {"label": label})
            group[suffix] = name

        records: list[SnapshotRecord] = []
        for radius in sorted(groups):
            group = groups[radius]
            suffix, member = _choose_lambda_file(group)  # type: ignore[arg-type]
            assert isinstance(member, str)
            raw = archive.read(member)
            if suffix == "npy":
                array = _load_npy_bytes(raw)
            else:
                array = np.loadtxt(BytesIO(raw), dtype=float)

            values, original_shape = _flatten_snapshot(array, member)
            records.append(
                SnapshotRecord(
                    radius=radius,
                    radius_label=str(group["label"]),
                    source=f"{path}!{member}",
                    original_shape=original_shape,
                    values=values,
                )
            )

    return records


def load_records_from_directory(path: Path) -> list[SnapshotRecord]:
    groups: dict[float, dict[str, str | Path | float]] = {}

    for file_path in path.rglob("lambda.*"):
        if file_path.suffix not in {".npy", ".txt"}:
            continue

        radius_info = _parse_radius_from_parts(file_path.parts)
        if radius_info is None:
            continue

        radius, label = radius_info
        suffix = file_path.suffix.lstrip(".")
        group = groups.setdefault(radius, {"label": label})
        group[suffix] = file_path

    records: list[SnapshotRecord] = []
    for radius in sorted(groups):
        group = groups[radius]
        suffix, file_path = _choose_lambda_file(group)  # type: ignore[arg-type]
        assert isinstance(file_path, Path)
        if suffix == "npy":
            array = np.load(file_path, allow_pickle=False)
        else:
            array = np.loadtxt(file_path, dtype=float)

        values, original_shape = _flatten_snapshot(array, str(file_path))
        records.append(
            SnapshotRecord(
                radius=radius,
                radius_label=str(group["label"]),
                source=str(file_path),
                original_shape=original_shape,
                values=values,
            )
        )

    return records


def load_records(input_path: Path) -> list[SnapshotRecord]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        records = load_records_from_zip(input_path)
    elif input_path.is_dir():
        records = load_records_from_directory(input_path)
    else:
        raise FileNotFoundError(
            f"Expected a .zip archive or extracted FEM_SOLS directory, got {input_path}"
        )

    if not records:
        raise ValueError(f"No rad=*/lambda.npy or rad=*/lambda.txt files found in {input_path}")

    snapshot_size = records[0].values.size
    mismatches = [
        (record.source, record.values.size)
        for record in records
        if record.values.size != snapshot_size
    ]
    if mismatches:
        details = ", ".join(f"{source}: {size}" for source, size in mismatches[:5])
        raise ValueError(
            "All lambda snapshots must have the same flattened size. "
            f"Expected {snapshot_size}; mismatches: {details}"
        )

    return records


def records_to_arrays(records: list[SnapshotRecord]) -> tuple[np.ndarray, np.ndarray]:
    radii = np.array([record.radius for record in records], dtype=float)
    snapshots = np.vstack([record.values for record in records])
    return snapshots, radii


def _component_header(component_count: int) -> str:
    columns = ["radius"] + [f"lambda_{i}" for i in range(component_count)]
    return ",".join(columns)


def save_merged_outputs(
    records: list[SnapshotRecord],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots, radii = records_to_arrays(records)

    paths = {
        "dataset_npz": output_dir / f"{prefix}_dataset.npz",
        "snapshots_npy": output_dir / f"{prefix}_snapshots.npy",
        "radii_npy": output_dir / f"{prefix}_radii.npy",
        "snapshots_txt": output_dir / f"{prefix}_snapshots.txt",
        "radii_txt": output_dir / f"{prefix}_radii.txt",
        "table_txt": output_dir / f"{prefix}_table.txt",
        "sources_csv": output_dir / f"{prefix}_sources.csv",
        "metadata_txt": output_dir / f"{prefix}_metadata.txt",
    }

    np.savez(
        paths["dataset_npz"],
        snapshots=snapshots,
        radii=radii,
        radius_labels=np.array([record.radius_label for record in records]),
        sources=np.array([record.source for record in records]),
    )
    np.save(paths["snapshots_npy"], snapshots)
    np.save(paths["radii_npy"], radii)
    np.savetxt(paths["snapshots_txt"], snapshots, fmt="%.18e")
    np.savetxt(paths["radii_txt"], radii, fmt="%.12g")
    np.savetxt(
        paths["table_txt"],
        np.column_stack([radii, snapshots]),
        fmt="%.18e",
        delimiter=",",
        header=_component_header(snapshots.shape[1]),
        comments="",
    )

    with paths["sources_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "radius", "radius_folder", "original_shape", "source"])
        for row, record in enumerate(records):
            writer.writerow(
                [
                    row,
                    f"{record.radius:.12g}",
                    f"rad={record.radius_label}",
                    "x".join(str(v) for v in record.original_shape),
                    record.source,
                ]
            )

    metadata = [
        "Merged lambda snapshot dataset",
        "",
        f"snapshot_count: {snapshots.shape[0]}",
        f"components_per_snapshot: {snapshots.shape[1]}",
        f"snapshot_matrix_shape: {snapshots.shape}",
        f"radii_min: {float(np.min(radii)):.12g}",
        f"radii_max: {float(np.max(radii)):.12g}",
        "",
        "Files:",
        f"- {paths['snapshots_npy'].name}: rows are flattened lambda snapshots",
        f"- {paths['radii_txt'].name}: radius value for each snapshot row",
        f"- {paths['table_txt'].name}: first column is radius, remaining columns are lambda values",
        f"- {paths['dataset_npz'].name}: snapshots, radii, radius_labels, sources",
    ]
    paths["metadata_txt"].write_text("\n".join(metadata) + "\n", encoding="utf-8")

    return paths


def _nice_indices(length: int, max_count: int) -> np.ndarray:
    if length <= max_count:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_count, dtype=int))


def plot_heatmap(snapshots: np.ndarray, radii: np.ndarray, output_path: Path) -> None:
    height = max(5.0, min(12.0, 2.2 + 0.16 * len(radii)))
    fig, ax = plt.subplots(figsize=(12, height))
    image = ax.imshow(snapshots, aspect="auto", origin="lower", cmap="viridis")

    y_indices = _nice_indices(len(radii), 12)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f"{radii[i]:.3g}" for i in y_indices])
    ax.set_xlabel("lambda component index")
    ax.set_ylabel("radius")
    ax.set_title(f"Lambda snapshots heatmap ({snapshots.shape[0]} radii x {snapshots.shape[1]} components)")

    cbar = fig.colorbar(image, ax=ax, pad=0.012)
    cbar.set_label("lambda value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_curves(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    max_curves: int,
) -> None:
    selected = _nice_indices(len(radii), max(1, max_curves))
    x = np.arange(snapshots.shape[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    for row in selected:
        ax.plot(x, snapshots[row], linewidth=1.5, label=f"r={radii[row]:.3g}")

    ax.set_xlabel("lambda component index")
    ax.set_ylabel("lambda value")
    ax.set_title("Selected lambda snapshot profiles")
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_norms(snapshots: np.ndarray, radii: np.ndarray, output_path: Path) -> None:
    l2_norms = np.linalg.norm(snapshots, axis=1)
    max_values = np.max(snapshots, axis=1)
    mean_values = np.mean(snapshots, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(radii, l2_norms, marker="o", linewidth=1.5)
    axes[0].set_ylabel("L2 norm")
    axes[1].plot(radii, max_values, marker="o", linewidth=1.5, color="#0f766e")
    axes[1].set_ylabel("max")
    axes[2].plot(radii, mean_values, marker="o", linewidth=1.5, color="#c2410c")
    axes[2].set_ylabel("mean")
    axes[2].set_xlabel("radius")

    for ax in axes:
        ax.grid(True, alpha=0.28)

    fig.suptitle("Lambda snapshot summary versus radius")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_interactive_html(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
) -> None:
    payload = {
        "radii": radii.tolist(),
        "snapshots": snapshots.tolist(),
    }
    max_index = snapshots.shape[0] - 1
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lambda Snapshots by Radius</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --surface: #ffffff;
      --ink: #17202a;
      --muted: #5f6f7d;
      --line: #d9e0e7;
      --accent: #0f766e;
      --accent-2: #c2410c;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .shell {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 20px;
    }}

    header {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) minmax(360px, 1.1fr);
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

    .subtle {{
      margin-top: 7px;
      color: var(--muted);
    }}

    .controls {{
      display: grid;
      grid-template-columns: 118px 1fr 120px;
      gap: 10px;
      align-items: end;
    }}

    label {{
      display: grid;
      gap: 6px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}

    input {{
      width: 100%;
      accent-color: var(--accent);
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

    .check {{
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }}

    .check input {{
      width: 16px;
      height: 16px;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}

    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 11px 12px;
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

    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
      gap: 14px;
      align-items: start;
    }}

    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 13px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}

    .canvas-wrap {{
      padding: 12px;
    }}

    canvas {{
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}

    #profileCanvas {{
      height: 420px;
    }}

    #heatmapCanvas {{
      height: 420px;
      cursor: crosshair;
    }}

    @media (max-width: 900px) {{
      header,
      .controls,
      .metrics,
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>Lambda Snapshots by Radius</h1>
        <p class="subtle">{snapshots.shape[0]} snapshots, {snapshots.shape[1]} lambda components per snapshot</p>
      </div>
      <div class="controls">
        <label>
          Row
          <input id="rowInput" type="number" min="0" max="{max_index}" step="1" value="0" />
        </label>
        <label>
          Snapshot
          <input id="rowSlider" type="range" min="0" max="{max_index}" step="1" value="0" />
        </label>
        <label class="check">
          <input id="localScale" type="checkbox" checked />
          local scale
        </label>
      </div>
    </header>

    <section class="metrics">
      <div class="metric"><span>Radius</span><strong id="radiusMetric"></strong></div>
      <div class="metric"><span>Row</span><strong id="rowMetric"></strong></div>
      <div class="metric"><span>Max</span><strong id="maxMetric"></strong></div>
      <div class="metric"><span>Mean</span><strong id="meanMetric"></strong></div>
      <div class="metric"><span>L2 Norm</span><strong id="normMetric"></strong></div>
    </section>

    <main class="grid">
      <section class="panel">
        <div class="panel-head">
          <span>Selected snapshot profile</span>
          <span id="profileScale"></span>
        </div>
        <div class="canvas-wrap"><canvas id="profileCanvas"></canvas></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <span>All snapshots heatmap</span>
          <span id="hoverMetric"></span>
        </div>
        <div class="canvas-wrap"><canvas id="heatmapCanvas"></canvas></div>
      </section>
    </main>
  </div>

  <script id="snapshotPayload" type="application/json">{json.dumps(payload)}</script>
  <script>
    const payload = JSON.parse(document.getElementById("snapshotPayload").textContent);
    const radii = payload.radii;
    const snapshots = payload.snapshots;
    const rows = snapshots.length;
    const cols = snapshots[0].length;
    const rowInput = document.getElementById("rowInput");
    const rowSlider = document.getElementById("rowSlider");
    const localScale = document.getElementById("localScale");
    const profileCanvas = document.getElementById("profileCanvas");
    const heatmapCanvas = document.getElementById("heatmapCanvas");
    const profileScale = document.getElementById("profileScale");
    const hoverMetric = document.getElementById("hoverMetric");

    const allValues = snapshots.flat();
    const globalMin = Math.min(...allValues);
    const globalMax = Math.max(...allValues);

    function resizeCanvas(canvas) {{
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {{ ctx, width: rect.width, height: rect.height }};
    }}

    function lerp(a, b, t) {{
      return a + (b - a) * t;
    }}

    function heatColor(value) {{
      const t = Math.max(0, Math.min(1, (value - globalMin) / (globalMax - globalMin || 1)));
      const stops = [
        [35, 42, 93],
        [22, 111, 136],
        [30, 157, 122],
        [163, 204, 74],
        [245, 223, 66],
      ];
      const scaled = t * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(scaled));
      const f = scaled - i;
      const a = stops[i];
      const b = stops[i + 1];
      return `rgb(${{Math.round(lerp(a[0], b[0], f))}}, ${{Math.round(lerp(a[1], b[1], f))}}, ${{Math.round(lerp(a[2], b[2], f))}})`;
    }}

    function fmt(value) {{
      return Number(value).toPrecision(5);
    }}

    function selectedRow() {{
      return Math.max(0, Math.min(rows - 1, Number(rowSlider.value) || 0));
    }}

    function drawAxes(ctx, left, top, width, height, xLabel, yLabel) {{
      ctx.strokeStyle = "#cdd6df";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, top + height);
      ctx.lineTo(left + width, top + height);
      ctx.stroke();
      ctx.fillStyle = "#5f6f7d";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(xLabel, left + width - 118, top + height + 33);
      ctx.save();
      ctx.translate(left - 34, top + 110);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }}

    function drawProfile() {{
      const row = selectedRow();
      const data = snapshots[row];
      const {{ ctx, width, height }} = resizeCanvas(profileCanvas);
      ctx.clearRect(0, 0, width, height);

      const margin = {{ left: 58, right: 20, top: 22, bottom: 48 }};
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const minValue = localScale.checked ? Math.min(...data) : globalMin;
      const maxValue = localScale.checked ? Math.max(...data) : globalMax;
      const span = maxValue - minValue || 1;

      drawAxes(ctx, margin.left, margin.top, plotW, plotH, "component index", "lambda");

      ctx.strokeStyle = "rgba(15, 118, 110, 0.18)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{
        const y = margin.top + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotW, y);
        ctx.stroke();
      }}

      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((value, i) => {{
        const x = margin.left + (i / Math.max(1, cols - 1)) * plotW;
        const y = margin.top + (1 - (value - minValue) / span) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      ctx.fillStyle = "#17202a";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(`min ${{fmt(minValue)}}`, margin.left + 4, margin.top + plotH - 8);
      ctx.fillText(`max ${{fmt(maxValue)}}`, margin.left + 4, margin.top + 14);
      profileScale.textContent = localScale.checked ? "local y-scale" : "global y-scale";
    }}

    function drawHeatmap() {{
      const row = selectedRow();
      const {{ ctx, width, height }} = resizeCanvas(heatmapCanvas);
      ctx.clearRect(0, 0, width, height);

      const margin = {{ left: 46, right: 12, top: 18, bottom: 38 }};
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const cellW = plotW / cols;
      const cellH = plotH / rows;

      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          ctx.fillStyle = heatColor(snapshots[r][c]);
          ctx.fillRect(
            margin.left + c * cellW,
            margin.top + (rows - 1 - r) * cellH,
            Math.ceil(cellW) + 0.5,
            Math.ceil(cellH) + 0.5
          );
        }}
      }}

      const y = margin.top + (rows - 1 - row) * cellH;
      ctx.strokeStyle = "#c2410c";
      ctx.lineWidth = 2;
      ctx.strokeRect(margin.left, y, plotW, cellH);

      ctx.fillStyle = "#5f6f7d";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("radius", 4, margin.top + 10);
      ctx.fillText(`${{radii[0].toPrecision(3)}}`, 5, margin.top + plotH + 2);
      ctx.fillText(`${{radii[rows - 1].toPrecision(3)}}`, 5, margin.top + 12);
      ctx.fillText("component index", margin.left + plotW - 112, margin.top + plotH + 28);
    }}

    function updateMetrics() {{
      const row = selectedRow();
      const data = snapshots[row];
      const sum = data.reduce((a, b) => a + b, 0);
      const norm = Math.sqrt(data.reduce((a, b) => a + b * b, 0));
      document.getElementById("radiusMetric").textContent = radii[row].toPrecision(5);
      document.getElementById("rowMetric").textContent = `${{row}} / ${{rows - 1}}`;
      document.getElementById("maxMetric").textContent = fmt(Math.max(...data));
      document.getElementById("meanMetric").textContent = fmt(sum / data.length);
      document.getElementById("normMetric").textContent = fmt(norm);
    }}

    function setRow(row) {{
      const next = Math.max(0, Math.min(rows - 1, Math.round(row)));
      rowSlider.value = String(next);
      rowInput.value = String(next);
      updateMetrics();
      drawProfile();
      drawHeatmap();
    }}

    rowSlider.addEventListener("input", () => setRow(Number(rowSlider.value)));
    rowInput.addEventListener("change", () => setRow(Number(rowInput.value)));
    localScale.addEventListener("change", drawProfile);
    window.addEventListener("resize", () => setRow(selectedRow()));

    heatmapCanvas.addEventListener("mousemove", (event) => {{
      const rect = heatmapCanvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const margin = {{ left: 46, right: 12, top: 18, bottom: 38 }};
      const plotW = rect.width - margin.left - margin.right;
      const plotH = rect.height - margin.top - margin.bottom;
      const c = Math.floor(((x - margin.left) / plotW) * cols);
      const rFromTop = Math.floor(((y - margin.top) / plotH) * rows);
      const r = rows - 1 - rFromTop;
      if (r >= 0 && r < rows && c >= 0 && c < cols) {{
        hoverMetric.textContent = `r=${{radii[r].toPrecision(4)}}, c=${{c}}, v=${{fmt(snapshots[r][c])}}`;
      }} else {{
        hoverMetric.textContent = "";
      }}
    }});

    heatmapCanvas.addEventListener("click", (event) => {{
      const rect = heatmapCanvas.getBoundingClientRect();
      const y = event.clientY - rect.top;
      const margin = {{ top: 18, bottom: 38 }};
      const plotH = rect.height - margin.top - margin.bottom;
      const rFromTop = Math.floor(((y - margin.top) / plotH) * rows);
      const r = rows - 1 - rFromTop;
      if (r >= 0 && r < rows) setRow(r);
    }});

    setRow(0);
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def make_visualizations(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_dir: Path,
    prefix: str,
    max_curves: int,
    write_html: bool,
) -> dict[str, Path]:
    paths = {
        "heatmap_png": output_dir / f"{prefix}_heatmap.png",
        "curves_png": output_dir / f"{prefix}_profiles.png",
        "summary_png": output_dir / f"{prefix}_summary_vs_radius.png",
    }
    plot_heatmap(snapshots, radii, paths["heatmap_png"])
    plot_curves(snapshots, radii, paths["curves_png"], max_curves=max_curves)
    plot_norms(snapshots, radii, paths["summary_png"])

    if write_html:
        paths["html"] = output_dir / f"{prefix}_viewer.html"
        _write_interactive_html(snapshots, radii, paths["html"])

    return paths


def default_input_path() -> Path:
    if DEFAULT_ARCHIVE.exists():
        return DEFAULT_ARCHIVE
    return DEFAULT_DIRECTORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge FEM_SOLS/rad=*/lambda.npy snapshots into one row-wise dataset "
            "and create quick visualizations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Path to FEM_SOLS zip archive or extracted FEM_SOLS directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where merged data and plots will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="lambda",
        help="Filename prefix for generated files.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=12,
        help="Maximum number of snapshots to overlay in the profile plot.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write merged data files.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip the interactive HTML viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    snapshots, radii = records_to_arrays(records)

    data_paths = save_merged_outputs(records, args.output_dir, args.prefix)
    plot_paths: dict[str, Path] = {}
    if not args.no_plots:
        plot_paths = make_visualizations(
            snapshots,
            radii,
            args.output_dir,
            args.prefix,
            max_curves=args.max_curves,
            write_html=not args.no_html,
        )

    print(f"Loaded {snapshots.shape[0]} lambda snapshots from {args.input}")
    print(f"Snapshot matrix shape: {snapshots.shape}")
    print(f"Radius range: {radii[0]:.12g} to {radii[-1]:.12g}")
    print(f"Merged data written to: {args.output_dir}")
    for label, path in {**data_paths, **plot_paths}.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
