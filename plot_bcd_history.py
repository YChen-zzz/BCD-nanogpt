#!/usr/bin/env python3
"""Plot BCD hyperparameter search history as an SVG.

This script intentionally uses only the Python standard library so it can run
on training machines that do not have matplotlib installed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HISTORY = (
    "/models/share/chenyupeng/chenyupeng/pretraining_record/"
    "BCD_optimizer/adamw/1c130m/bcd_history.json"
)


@dataclass(frozen=True)
class Block:
    round_id: int
    param_name: str
    start: int
    end: int
    rows: list[dict[str, Any]]


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.0e}"
        return f"{value:g}"
    return str(value)


def format_loss(value: float) -> str:
    return f"{value:.4f}"


def group_consecutive_blocks(rows: list[dict[str, Any]]) -> list[Block]:
    blocks: list[Block] = []
    start = 0
    while start < len(rows):
        round_id = int(rows[start]["round"])
        param_name = str(rows[start]["param_name"])
        end = start + 1
        while (
            end < len(rows)
            and int(rows[end]["round"]) == round_id
            and str(rows[end]["param_name"]) == param_name
        ):
            end += 1
        blocks.append(Block(round_id, param_name, start, end, rows[start:end]))
        start = end
    return blocks


def nice_ticks(y_min: float, y_max: float, count: int = 6) -> list[float]:
    if y_min == y_max:
        return [y_min]
    raw_step = (y_max - y_min) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    first = math.floor(y_min / step) * step
    ticks = []
    value = first
    while value <= y_max + step * 0.5:
        if value >= y_min - step * 0.5:
            ticks.append(value)
        value += step
    return ticks


def write_summary_csv(rows: list[dict[str, Any]], blocks: list[Block], path: Path, threshold: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "block_index",
                "round",
                "param_name",
                "old_param_value",
                "old_loss",
                "best_param_value_in_grid",
                "best_val_loss_in_grid",
                "accepted_param_value_if_delta_gt_threshold",
                "accepted_loss_if_delta_gt_threshold",
                "loss_delta",
                "threshold",
            ]
        )
        for block_index, block in enumerate(blocks, start=1):
            best_row = min(block.rows, key=lambda row: float(row["val_loss"]))
            old_loss = float(block.rows[0]["old_loss"])
            loss_delta = old_loss - float(best_row["val_loss"])
            accepted_value = (
                best_row["param_value"]
                if loss_delta > threshold
                else block.rows[0]["old_param_value"]
            )
            accepted_loss = float(best_row["val_loss"]) if loss_delta > threshold else old_loss
            writer.writerow(
                [
                    block_index,
                    block.round_id,
                    block.param_name,
                    format_value(block.rows[0]["old_param_value"]),
                    old_loss,
                    format_value(best_row["param_value"]),
                    float(best_row["val_loss"]),
                    format_value(accepted_value),
                    accepted_loss,
                    loss_delta,
                    threshold,
                ]
            )


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    fill: str = "#17202a",
    anchor: str = "middle",
    weight: str = "400",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:g} {x:g} {y:g})"' if rotate is not None else ""
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{html.escape(text)}</text>'
    )


def make_svg(rows: list[dict[str, Any]], blocks: list[Block], title: str) -> str:
    n = len(rows)
    width = max(1280, n * 34 + 260)
    height = 760
    left = 92
    right = 34
    top = 78
    bottom = 178
    plot_w = width - left - right
    plot_h = height - top - bottom

    losses = [float(row["val_loss"]) for row in rows] + [float(row["old_loss"]) for row in rows]
    y_min = min(losses)
    y_max = max(losses)
    pad = max((y_max - y_min) * 0.08, 0.005)
    y_min -= pad
    y_max += pad

    def x_pos(index: float) -> float:
        if n <= 1:
            return left + plot_w / 2
        return left + index * plot_w / (n - 1)

    def y_pos(loss: float) -> float:
        return top + (y_max - loss) * plot_h / (y_max - y_min)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 34, title, size=22, anchor="start", weight="700"),
        svg_text(left, 58, f"{n} grid runs, {len(blocks)} BCD blocks", size=13, fill="#566573", anchor="start"),
    ]

    # Alternating block bands should sit behind grid lines and points.
    for block_i, block in enumerate(blocks):
        x_start = x_pos(block.start - 0.5 if block.start > 0 else 0)
        x_end = x_pos(block.end - 0.5 if block.end < n else n - 1)
        if block_i % 2 == 1:
            parts.append(
                f'<rect x="{x_start:g}" y="{top}" width="{max(0, x_end - x_start):g}" height="{plot_h}" '
                'fill="#f7f9fb"/>'
            )

    # Grid and y-axis.
    for tick in nice_ticks(y_min, y_max):
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:g}" x2="{width - right}" y2="{y:g}" stroke="#e8edf2" stroke-width="1"/>')
        parts.append(svg_text(left - 10, y + 4, format_loss(tick), size=12, fill="#566573", anchor="end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#2c3e50" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#2c3e50" stroke-width="1.2"/>')
    parts.append(svg_text(22, top + plot_h / 2, "loss", size=14, fill="#2c3e50", rotate=-90))

    # Vertical block bands and labels.
    for block_i, block in enumerate(blocks):
        if block.start > 0:
            sep = x_pos(block.start - 0.5)
            parts.append(f'<line x1="{sep:g}" y1="{top}" x2="{sep:g}" y2="{height - bottom}" stroke="#d5dde5" stroke-width="1"/>')
        center = (x_pos(block.start) + x_pos(block.end - 1)) / 2
        parts.append(svg_text(center, height - 122, f"R{block.round_id}", size=11, fill="#566573"))
        parts.append(svg_text(center, height - 104, block.param_name, size=12, fill="#17202a", weight="700"))

    # Old-loss dashed line.
    old_points = " ".join(f"{x_pos(i):g},{y_pos(float(row['old_loss'])):g}" for i, row in enumerate(rows))
    parts.append(
        f'<polyline points="{old_points}" fill="none" stroke="#6c757d" stroke-width="2.2" '
        'stroke-dasharray="7 5" stroke-linejoin="round"/>'
    )

    # Val-loss scatter and candidate value labels.
    best_indices = {block.start + min(range(len(block.rows)), key=lambda i: float(block.rows[i]["val_loss"])) for block in blocks}
    global_best_index = min(range(n), key=lambda i: float(rows[i]["val_loss"]))
    for i, row in enumerate(rows):
        x = x_pos(i)
        y = y_pos(float(row["val_loss"]))
        is_block_best = i in best_indices
        is_global_best = i == global_best_index
        radius = 5.8 if is_block_best else 4.7
        stroke = "#d35400" if is_block_best else "#1f77b4"
        fill = "#d35400" if is_global_best else "#ffffff"
        width_px = 2.4 if is_block_best else 1.8
        tooltip = (
            f"round={row['round']}, param={row['param_name']}, value={format_value(row['param_value'])}, "
            f"val_loss={format_loss(float(row['val_loss']))}, old_loss={format_loss(float(row['old_loss']))}"
        )
        parts.append(
            f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" fill="{fill}" stroke="{stroke}" stroke-width="{width_px:g}">'
            f"<title>{html.escape(tooltip)}</title></circle>"
        )
        parts.append(svg_text(x, height - 78, format_value(row["param_value"]), size=10, fill="#566573", rotate=-45))

    # Legend.
    legend_x = width - right - 330
    legend_y = 34
    parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}" stroke="#6c757d" stroke-width="2.2" stroke-dasharray="7 5"/>')
    parts.append(svg_text(legend_x + 52, legend_y + 4, "old_loss before each block", size=12, anchor="start", fill="#566573"))
    parts.append(f'<circle cx="{legend_x + 8}" cy="{legend_y + 25}" r="5" fill="#ffffff" stroke="#1f77b4" stroke-width="1.8"/>')
    parts.append(svg_text(legend_x + 52, legend_y + 29, "val_loss candidate", size=12, anchor="start", fill="#566573"))
    parts.append(f'<circle cx="{legend_x + 8}" cy="{legend_y + 50}" r="5.8" fill="#ffffff" stroke="#d35400" stroke-width="2.4"/>')
    parts.append(svg_text(legend_x + 52, legend_y + 54, "best candidate in block", size=12, anchor="start", fill="#566573"))
    parts.append(f'<circle cx="{legend_x + 8}" cy="{legend_y + 75}" r="5.8" fill="#d35400" stroke="#d35400" stroke-width="2.4"/>')
    parts.append(svg_text(legend_x + 52, legend_y + 79, "global best val_loss", size=12, anchor="start", fill="#566573"))

    best = rows[global_best_index]
    summary = (
        f"best: R{best['round']} {best['param_name']}={format_value(best['param_value'])}, "
        f"val_loss={format_loss(float(best['val_loss']))}"
    )
    parts.append(svg_text(left, height - 28, summary, size=13, anchor="start", fill="#2c3e50", weight="700"))
    parts.append(svg_text(width - right, height - 28, "x-axis: each candidate run; block labels show round / param_name", size=12, anchor="end", fill="#566573"))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default=DEFAULT_HISTORY, help="Path to bcd_history.json")
    parser.add_argument("--output", default=None, help="Output SVG path. Defaults to HISTORY_DIR/bcd_history_plot.svg")
    parser.add_argument("--summary-csv", default=None, help="Optional CSV summary path. Defaults to HISTORY_DIR/bcd_history_summary.csv")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=3e-3,
        help="BCD acceptance threshold used for the summary CSV. The plot still shows all raw grid losses.",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    with history_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{history_path} must contain a non-empty JSON list")

    blocks = group_consecutive_blocks(rows)
    output_path = Path(args.output) if args.output else history_path.with_name("bcd_history_plot.svg")
    csv_path = Path(args.summary_csv) if args.summary_csv else history_path.with_name("bcd_history_summary.csv")
    title = args.title or f"BCD search history: {history_path.parent.name}"

    output_path.write_text(make_svg(rows, blocks, title), encoding="utf-8")
    write_summary_csv(rows, blocks, csv_path, args.convergence_threshold)
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
