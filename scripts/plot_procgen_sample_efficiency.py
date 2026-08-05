import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Expert": "#000000",
    "Ours": "#003366",
    "DPT": "#2ca02c",
    "BC+PPO": "#9467bd",
}

MARKERS = {
    "Ours": "o",
    "DPT": "s",
    "BC+PPO": "v",
}


def stderr(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=0) / np.sqrt(len(values)))


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def normalize_returns(values, expert_return):
    values = np.asarray(values, dtype=float)
    if expert_return is None or not np.isfinite(expert_return) or abs(expert_return) < 1e-8:
        return values
    if expert_return < 0:
        denom = np.maximum(np.abs(values), 1e-8)
        return abs(expert_return) / denom
    return values / expert_return


def load_curve(paths, metric, expert_return=None):
    curves = []
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        steps = np.asarray([float(row["env_interactions"]) for row in rows], dtype=float)
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        curves.append((steps, values))
    if not curves:
        return None
    min_len = min(len(values) for _, values in curves)
    steps = curves[0][0][:min_len]
    raw_values = np.asarray([values[:min_len] for _, values in curves], dtype=float)
    values = normalize_returns(raw_values, expert_return) if metric == "mean_return" else raw_values
    return {
        "steps": steps,
        "mean": values.mean(axis=0),
        "stderr": values.std(axis=0) / np.sqrt(values.shape[0]),
        "raw_mean": raw_values.mean(axis=0),
        "raw_stderr": raw_values.std(axis=0) / np.sqrt(raw_values.shape[0]),
        "seed_values": raw_values,
    }


def load_single_point(paths, metric, expert_return=None, force_expert_one=False):
    vals = []
    steps = []
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        vals.append(float(rows[-1][metric]))
        steps.append(float(rows[-1]["env_interactions"]))
    if not vals:
        return None
    vals = np.asarray(vals, dtype=float)
    raw_vals = vals.copy()
    if force_expert_one and metric == "mean_return":
        vals = np.ones_like(vals)
    elif metric == "mean_return":
        vals = normalize_returns(vals, expert_return)
    return {
        "step": float(np.mean(steps)),
        "mean": float(vals.mean()),
        "stderr": stderr(vals),
        "raw_mean": float(raw_vals.mean()),
        "raw_stderr": stderr(raw_vals),
        "seed_values": raw_vals,
    }


def default_paths(args, method):
    seeds = args.seeds
    if method == "context":
        return [
            Path(args.context_dir) / f"context_accumulator_procgen-maze-seed{seed}" / "metrics.csv"
            for seed in seeds
        ]
    if method == "dpt":
        return [Path(args.dpt_dir) / f"dpt_seed{seed}.csv" for seed in seeds]
    if method == "bc_ppo":
        return [Path(args.bc_ppo_dir) / f"bc_ppo_procgen-seed{seed}" / "metrics.csv" for seed in seeds]
    if method == "expert":
        return [Path(args.expert_dir) / f"expert_seed{seed}.csv" for seed in seeds]
    raise ValueError(method)


def add_dump_rows(rows, method, metric, data, normalized, is_curve):
    if data is None:
        return
    if is_curve:
        for idx, step in enumerate(data["steps"]):
            rows.append({
                "method": method,
                "metric": metric,
                "env_interactions": float(step),
                "mean": float(data["mean"][idx]),
                "stderr": float(data["stderr"][idx]),
                "raw_mean": float(data["raw_mean"][idx]),
                "raw_stderr": float(data["raw_stderr"][idx]),
                "normalized": int(normalized),
                "seed_values": "|".join(f"{v:.8g}" for v in data["seed_values"][:, idx]),
            })
    else:
        rows.append({
            "method": method,
            "metric": metric,
            "env_interactions": float(data["step"]),
            "mean": float(data["mean"]),
            "stderr": float(data["stderr"]),
            "raw_mean": float(data["raw_mean"]),
            "raw_stderr": float(data["raw_stderr"]),
            "normalized": int(normalized),
            "seed_values": "|".join(f"{v:.8g}" for v in data["seed_values"]),
        })


def write_plot_data(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(args, metric, ylabel, output_name):
    expert_raw = load_single_point(default_paths(args, "expert"), metric)
    expert_return = expert_raw["raw_mean"] if metric == "mean_return" and expert_raw is not None else None
    expert = load_single_point(
        default_paths(args, "expert"),
        metric,
        expert_return=expert_return,
        force_expert_one=True,
    )
    context = load_curve(default_paths(args, "context"), metric, expert_return=expert_return)
    dpt = load_single_point(default_paths(args, "dpt"), metric, expert_return=expert_return)
    bc_ppo = load_curve(default_paths(args, "bc_ppo"), metric, expert_return=expert_return)
    normalized = metric == "mean_return"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    fig, ax = plt.subplots(figsize=(3.2, 3.05))

    if expert is not None:
        ax.axhline(expert["mean"], color=COLORS["Expert"], linestyle="--", linewidth=1.5, label="Expert")

    for label, curve in [("Ours", context), ("BC+PPO", bc_ppo)]:
        if curve is None:
            continue
        ax.plot(curve["steps"], curve["mean"], color=COLORS[label], marker=MARKERS[label], linewidth=2.0, label=label)
        ax.fill_between(
            curve["steps"],
            curve["mean"] - curve["stderr"],
            curve["mean"] + curve["stderr"],
            color=COLORS[label],
            alpha=0.15,
            linewidth=0,
        )

    if dpt is not None:
        xmax = max(
            [v for v in [
                context["steps"][-1] if context is not None else None,
                bc_ppo["steps"][-1] if bc_ppo is not None else None,
                dpt["step"],
            ] if v is not None]
        )
        xmin = max(1.0, min(
            [v for v in [
                context["steps"][0] if context is not None else None,
                bc_ppo["steps"][0] if bc_ppo is not None else None,
                dpt["step"],
            ] if v is not None]
        ))
        ax.hlines(dpt["mean"], xmin=xmin, xmax=xmax, color=COLORS["DPT"], linestyle=":", linewidth=2.0, label="DPT")
        ax.scatter([dpt["step"]], [dpt["mean"]], color=COLORS["DPT"], marker=MARKERS["DPT"], edgecolors="white", zorder=20)

    ax.set_xscale("log")
    ax.set_xlabel("Environment Interactions")
    ax.set_ylabel(ylabel)
    ax.set_title("Procgen Maze")
    if metric == "success_rate":
        ax.set_ylim(-0.05, 1.05)
    elif metric == "mean_return":
        ax.set_ylim(bottom=0.0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, min(len(labels), 4)),
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=1.0,
        facecolor="white",
        fontsize=8,
    )
    ax.set_box_aspect(1)
    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir) / output_name
    dump_rows = []
    add_dump_rows(dump_rows, "Expert", metric, expert, normalized, is_curve=False)
    add_dump_rows(dump_rows, "Ours", metric, context, normalized, is_curve=True)
    add_dump_rows(dump_rows, "DPT", metric, dpt, normalized, is_curve=False)
    add_dump_rows(dump_rows, "BC+PPO", metric, bc_ppo, normalized, is_curve=True)
    write_plot_data(out.with_suffix(".csv"), dump_rows)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.csv')}")


def build_parser():
    parser = argparse.ArgumentParser(description="Plot procgen sample efficiency from local CSV outputs.")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--context-dir", default="context_results")
    parser.add_argument("--dpt-dir", default="procgen_eval_results")
    parser.add_argument("--bc-ppo-dir", default="bc_ppo_procgen_results")
    parser.add_argument("--expert-dir", default="expert_procgen_results")
    parser.add_argument("--output-dir", default="procgen_plots")
    return parser


def main():
    args = build_parser().parse_args()
    plot_metric(args, "success_rate", "Success Rate", "procgen_success_vs_env_steps.png")
    plot_metric(args, "mean_return", "Normalized Return (Expert = 1)", "procgen_return_vs_env_steps.png")


if __name__ == "__main__":
    main()
