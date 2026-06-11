#!/usr/bin/env python3
"""Analyze near-best BCD configs across multiple settings.

The script reads each setting's bcd_history.json, keeps all runs with
val_loss <= best_loss + threshold, and writes:
  - a structured JSON report
  - one CSV row per near-best run
  - one CSV row per setting/param/value summary
  - one CSV row per cross-setting common param value
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "configs/adamw/near_best_analysis.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.0e}"
        return f"{value:g}"
    return str(value)


def canonical_value_key(value: Any) -> str:
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"number:{format_value(float(value))}"
    return f"str:{value}"


def display_value_from_key(value_key: str) -> str:
    _, _, value = value_key.partition(":")
    return value


def tau_label(threshold: float) -> str:
    return format_value(threshold).replace("+", "").replace("-", "m")


def finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def resolve_setting(setting: Any, config_dir: Path) -> dict[str, Any]:
    if isinstance(setting, str):
        return {"name": setting, "output_dir": Path(setting)}
    if not isinstance(setting, dict):
        raise ValueError(f"Each setting must be a string or mapping, got {setting!r}")

    name = setting.get("name")
    if not name:
        raise ValueError(f"Setting is missing name: {setting!r}")

    resolved = dict(setting)
    if "search_config" in setting:
        search_config = Path(setting["search_config"])
        if not search_config.is_absolute():
            search_config = (config_dir / search_config).resolve()
        search_data = load_yaml(search_config)
        if "output_dir" not in search_data:
            raise ValueError(f"search_config has no output_dir: {search_config}")
        resolved["search_config"] = str(search_config)
        resolved.setdefault("output_dir", search_data["output_dir"])

    if "output_dir" not in resolved:
        raise ValueError(f"Setting must define output_dir or search_config: {setting!r}")

    resolved["name"] = str(name)
    resolved["output_dir"] = str(Path(resolved["output_dir"]))
    return resolved


def load_history(setting: dict[str, Any]) -> list[dict[str, Any]]:
    history_path = Path(setting["output_dir"]) / "bcd_history.json"
    rows = load_json(history_path)
    if not isinstance(rows, list):
        raise ValueError(f"bcd_history.json must contain a list: {history_path}")

    valid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"History row {index} is not a mapping: {history_path}")
        if "all_params" not in row or not isinstance(row["all_params"], dict):
            raise ValueError(f"History row {index} has no all_params mapping: {history_path}")
        val_loss = finite_float(row.get("val_loss"), field=f"{history_path} row {index} val_loss")
        copied = dict(row)
        copied["val_loss"] = val_loss
        copied["_history_index"] = index
        valid_rows.append(copied)

    if not valid_rows:
        raise ValueError(f"No valid history rows: {history_path}")
    return valid_rows


def best_loss_for_setting(setting: dict[str, Any], rows: list[dict[str, Any]], source: str) -> float:
    if source == "bcd_history_min":
        return min(float(row["val_loss"]) for row in rows)
    if source == "final_result":
        final_path = Path(setting["output_dir"]) / "final_result.json"
        final_result = load_json(final_path)
        return finite_float(final_result.get("best_val_loss"), field=f"{final_path} best_val_loss")
    raise ValueError(f"Unknown best_loss_source: {source}")


def collect_param_order(config: dict[str, Any], settings: list[dict[str, Any]]) -> list[str]:
    configured = config.get("param_order")
    if configured:
        return [str(item) for item in configured]

    for setting in settings:
        search_config = setting.get("search_config")
        if search_config:
            search_data = load_yaml(Path(search_config))
            order = search_data.get("bcd_order")
            if order:
                return [str(item) for item in order]

    return []


def sorted_param_names(rows_by_setting: dict[str, list[dict[str, Any]]], param_order: list[str]) -> list[str]:
    names = set(param_order)
    for rows in rows_by_setting.values():
        for row in rows:
            names.update(str(key) for key in row["all_params"])
    return list(param_order) + sorted(names - set(param_order))


def run_record(row: dict[str, Any], best_loss: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "history_index": row["_history_index"],
        "val_loss": float(row["val_loss"]),
        "delta": float(row["val_loss"]) - best_loss,
        "round": row.get("round"),
        "param_name": row.get("param_name"),
        "param_value": row.get("param_value"),
        "run_dir": row.get("run_dir"),
        "config": row["all_params"],
    }


def summarize_values(
    near_runs: list[dict[str, Any]],
    param_names: list[str],
    best_loss: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for param in param_names:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in near_runs:
            if param in run["config"]:
                groups[canonical_value_key(run["config"][param])].append(run)

        param_summary: dict[str, dict[str, Any]] = {}
        for value_key, runs in sorted(groups.items(), key=lambda item: display_value_from_key(item[0])):
            losses = [float(run["val_loss"]) for run in runs]
            deltas = [loss - best_loss for loss in losses]
            best_run = min(runs, key=lambda run: float(run["val_loss"]))
            param_summary[display_value_from_key(value_key)] = {
                "count": len(runs),
                "best_loss": min(losses),
                "best_delta": min(deltas),
                "worst_loss": max(losses),
                "worst_delta": max(deltas),
                "example_run_dir": best_run.get("run_dir"),
            }
        summary[param] = param_summary
    return summary


def build_cross_setting_search(
    report_settings: dict[str, dict[str, Any]],
    param_names: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    setting_names = list(report_settings)

    for param in param_names:
        value_to_setting_runs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        for setting_name, setting_report in report_settings.items():
            runs_by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for run in setting_report["near_best_runs"]:
                config = run["config"]
                if param in config:
                    runs_by_value[canonical_value_key(config[param])].append(run)
            for value_key, runs in runs_by_value.items():
                value_to_setting_runs[value_key][setting_name] = runs

        common_values: dict[str, dict[str, Any]] = {}
        for value_key, per_setting_runs in sorted(
            value_to_setting_runs.items(), key=lambda item: display_value_from_key(item[0])
        ):
            if set(per_setting_runs) != set(setting_names):
                continue
            per_setting: dict[str, dict[str, Any]] = {}
            deltas = []
            for setting_name in setting_names:
                runs = per_setting_runs[setting_name]
                witness = min(runs, key=lambda run: float(run["val_loss"]))
                deltas.append(float(witness["delta"]))
                per_setting[setting_name] = {
                    "count": len(runs),
                    "best_loss": float(witness["val_loss"]),
                    "best_delta": float(witness["delta"]),
                    "witness_run_dir": witness.get("run_dir"),
                    "witness_config": witness["config"],
                }
            common_values[display_value_from_key(value_key)] = {
                "settings_covered": setting_names,
                "num_settings": len(setting_names),
                "worst_delta": max(deltas),
                "mean_delta": sum(deltas) / len(deltas),
                "per_setting": per_setting,
            }

        result[param] = {"common_values": common_values}
    return result


def write_near_best_csv(path: Path, report: dict[str, Any], param_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "setting",
        "rank",
        "val_loss",
        "delta",
        "history_index",
        "round",
        "param_name",
        "param_value",
    ] + param_names + ["run_dir"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for setting_name, setting_report in report["settings"].items():
            for run in setting_report["near_best_runs"]:
                row = {
                    "setting": setting_name,
                    "rank": run["rank"],
                    "val_loss": run["val_loss"],
                    "delta": run["delta"],
                    "history_index": run["history_index"],
                    "round": run["round"],
                    "param_name": run["param_name"],
                    "param_value": format_value(run["param_value"]),
                    "run_dir": run["run_dir"],
                }
                for param in param_names:
                    row[param] = format_value(run["config"].get(param, ""))
                writer.writerow(row)


def write_value_summary_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "setting",
        "param",
        "value",
        "count",
        "best_loss",
        "best_delta",
        "worst_loss",
        "worst_delta",
        "example_run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for setting_name, setting_report in report["settings"].items():
            for param, values in setting_report["value_summary"].items():
                for value, stats in values.items():
                    row = {"setting": setting_name, "param": param, "value": value}
                    row.update(stats)
                    writer.writerow(row)


def write_cross_setting_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "param",
        "value",
        "num_settings",
        "settings_covered",
        "worst_delta",
        "mean_delta",
        "per_setting_best_delta",
        "witness_run_dirs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for param, param_report in report["cross_setting_search"].items():
            for value, stats in param_report["common_values"].items():
                per_setting = stats["per_setting"]
                writer.writerow(
                    {
                        "param": param,
                        "value": value,
                        "num_settings": stats["num_settings"],
                        "settings_covered": ";".join(stats["settings_covered"]),
                        "worst_delta": stats["worst_delta"],
                        "mean_delta": stats["mean_delta"],
                        "per_setting_best_delta": ";".join(
                            f"{name}:{data['best_delta']}" for name, data in per_setting.items()
                        ),
                        "witness_run_dirs": ";".join(
                            f"{name}:{data['witness_run_dir']}" for name, data in per_setting.items()
                        ),
                    }
                )


def write_setting_overview_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "setting",
        "best_loss",
        "cutoff_loss",
        "num_total_runs",
        "num_near_best_runs",
        "output_dir",
        "history_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for setting_name, setting_report in report["settings"].items():
            row = {"setting": setting_name}
            row.update({key: setting_report[key] for key in columns if key != "setting"})
            writer.writerow(row)


def analyze(config: dict[str, Any], config_path: Path, threshold_override: float | None, output_dir_override: str | None) -> dict[str, Any]:
    threshold = finite_float(
        threshold_override if threshold_override is not None else config.get("threshold", 0.0064),
        field="threshold",
    )
    best_loss_source = str(config.get("best_loss_source", "bcd_history_min"))
    raw_settings = config.get("settings")
    if not raw_settings:
        raise ValueError("Config must define settings")

    settings = [resolve_setting(setting, config_path.parent) for setting in raw_settings]
    rows_by_setting = {setting["name"]: load_history(setting) for setting in settings}
    param_names = sorted_param_names(rows_by_setting, collect_param_order(config, settings))

    report_settings: dict[str, dict[str, Any]] = {}
    for setting in settings:
        setting_name = setting["name"]
        rows = rows_by_setting[setting_name]
        best_loss = best_loss_for_setting(setting, rows, best_loss_source)
        cutoff = best_loss + threshold
        near_rows = sorted(
            [row for row in rows if float(row["val_loss"]) <= cutoff],
            key=lambda row: (float(row["val_loss"]), int(row["_history_index"])),
        )
        near_runs = [run_record(row, best_loss, rank) for rank, row in enumerate(near_rows, start=1)]
        report_settings[setting_name] = {
            "output_dir": setting["output_dir"],
            "history_path": str(Path(setting["output_dir"]) / "bcd_history.json"),
            "best_loss": best_loss,
            "cutoff_loss": cutoff,
            "num_total_runs": len(rows),
            "num_near_best_runs": len(near_runs),
            "near_best_runs": near_runs,
            "value_summary": summarize_values(near_runs, param_names, best_loss),
        }

    report = {
        "optimizer": config.get("optimizer"),
        "threshold": threshold,
        "best_loss_source": best_loss_source,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings_order": [setting["name"] for setting in settings],
        "param_order": param_names,
        "settings": report_settings,
    }
    report["cross_setting_search"] = build_cross_setting_search(report_settings, param_names)

    output_dir_value = output_dir_override or config.get("output_dir")
    if not output_dir_value:
        first_output_dir = Path(settings[0]["output_dir"])
        output_dir_value = str(first_output_dir.parent / "analysis")
    output_dir = Path(output_dir_value)

    prefix = str(config.get("filename_prefix", "near_best_configs"))
    label = tau_label(threshold)
    outputs = {
        "json": str(output_dir / f"{prefix}_tau_{label}.json"),
        "near_best_configs_csv": str(output_dir / f"{prefix}_tau_{label}.csv"),
        "setting_overview_csv": str(output_dir / f"near_best_setting_overview_tau_{label}.csv"),
        "value_summary_csv": str(output_dir / f"near_best_value_summary_tau_{label}.csv"),
        "cross_setting_csv": str(output_dir / f"cross_setting_common_values_tau_{label}.csv"),
    }
    report["outputs"] = outputs

    write_json(Path(outputs["json"]), report)
    write_near_best_csv(Path(outputs["near_best_configs_csv"]), report, param_names)
    write_setting_overview_csv(Path(outputs["setting_overview_csv"]), report)
    write_value_summary_csv(Path(outputs["value_summary_csv"]), report)
    write_cross_setting_csv(Path(outputs["cross_setting_csv"]), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze near-best BCD configs across settings.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML analysis config.")
    parser.add_argument("--threshold", type=float, default=None, help="Override near-best loss threshold.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    report = analyze(config, config_path, args.threshold, args.output_dir)

    print(f"wrote: {report['outputs']['json']}")
    print(f"wrote: {report['outputs']['near_best_configs_csv']}")
    print(f"wrote: {report['outputs']['value_summary_csv']}")
    print(f"wrote: {report['outputs']['cross_setting_csv']}")


if __name__ == "__main__":
    main()
