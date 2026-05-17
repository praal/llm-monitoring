"""
Tree-LTL Temporal Elasticity — Plotting
========================================

Reads all result files from results-tree/ and plots balanced accuracy vs k.

Usage:
    python plot_tree_ltl.py
    python plot_tree_ltl.py --models google/gemini-2.5-pro openai/gpt-4.1
    python plot_tree_ltl.py --results-dir results-tree
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ─────────────────────────────────────────────
# LaTeX rendering
# ─────────────────────────────────────────────

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

RESULTS_DIR = Path("results-tree")
PLOTS_DIR = Path("plots")

MODEL_DISPLAY = {
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
    "openai/gpt-4.1": "GPT-4.1",
    "openai/gpt-4.1-mini": "GPT-4.1-Mini",
    "openai/gpt-4o-mini": "GPT-4o-Mini",
    "anthropic/claude-3.5-haiku": "Claude-3.5-Haiku",
    "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    "meta-llama/llama-3.3-70B-instruct": "LLaMA-3.3-70B",
    "meta-llama/llama-3.1-8b-instruct": "LLaMA-3.1-8B",
    "qwen/qwen-2.5-7b-instruct": "Qwen-2.5-7B",
}

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_all_results(results_dir: Path,
                     model_filter: list[str] | None = None,
                     b_filter: int | None = None,
                     d_filter: int | None = None) -> dict:
    """
    Load all results_*.json files from results_dir.
    Returns {model_name: data_dict}.
    """
    all_data = {}
    for fp in sorted(results_dir.glob("results_*.json")):
        with open(fp) as f:
            data = json.load(f)
        model = data["model"]

        if model_filter and model not in model_filter:
            continue

        tree = data.get("tree", {})
        if b_filter is not None and tree.get("branching") != b_filter:
            continue
        if d_filter is not None and tree.get("depth") != d_filter:
            continue

        all_data[model] = data
        n_traces = sum(len(v["traces"]) for v in data["results_by_k"].values())
        b, d = tree.get("branching", "?"), tree.get("depth", "?")
        print(f"  Loaded {n_traces:>5} traces: {model} (b={b}, d={d})")

    return all_data


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.split("/")[-1])


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def balanced_accuracy_with_sem(traces: list[dict]) -> tuple[float, float]:
    """
    Compute balanced accuracy = (pos_acc + neg_acc) / 2 and its SEM.
    SEM of balanced accuracy = sqrt(sem_pos^2 + sem_neg^2) / 2.
    """
    pos = [t for t in traces if t["ground_truth"] == "VALID"]
    neg = [t for t in traces if t["ground_truth"] == "INVALID"]

    if not pos or not neg:
        return 0.0, 0.0

    p_pos = sum(1 for t in pos if t["correct"]) / len(pos)
    p_neg = sum(1 for t in neg if t["correct"]) / len(neg)

    sem_pos = np.sqrt(p_pos * (1 - p_pos) / len(pos)) / 2
    sem_neg = np.sqrt(p_neg * (1 - p_neg) / len(neg)) / 2

    bal_acc = (p_pos + p_neg) / 2
    bal_sem = np.sqrt(sem_pos**2 + sem_neg**2) / 2

    return bal_acc, bal_sem


def accuracy_with_sem(traces: list[dict]) -> tuple[float, float]:
    """Returns (accuracy, SEM)."""
    n = len(traces)
    if n == 0:
        return 0.0, 0.0
    p = sum(1 for t in traces if t["correct"]) / n
    sem = np.sqrt(p * (1 - p) / n)
    return p, sem


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _save_and_show(fig, save_path: Path | None) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {save_path}")
    plt.show()


def _model_style(idx: int):
    colors = plt.cm.tab10.colors
    return colors[idx % len(colors)], MARKERS[idx % len(MARKERS)]


def _get_k_values(all_data: dict) -> list[int]:
    """Extract all k values across all models, sorted."""
    ks = set()
    for data in all_data.values():
        for k_str in data["results_by_k"]:
            if k_str == "5000" or k_str == "500":
                continue
            ks.add(int(k_str))
    return sorted(ks)


# ─────────────────────────────────────────────
# Plot: Balanced accuracy vs k
# ─────────────────────────────────────────────

def plot_balanced_accuracy(all_data: dict, save_path: Path | None = None) -> None:
    """Single plot: balanced accuracy vs k for all models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    k_values = _get_k_values(all_data)

    for idx, (model, data) in enumerate(all_data.items()):
        accs, sems, ks_present = [], [], []

        for k in k_values:
            k_str = str(k)
            if k_str not in data["results_by_k"]:
                continue
            traces = data["results_by_k"][k_str]["traces"]
            if not traces:
                continue
            acc, sem = balanced_accuracy_with_sem(traces)
            accs.append(acc)
            sems.append(sem)
            ks_present.append(k)

        if not ks_present:
            continue

        color, marker = _model_style(idx)
        ax.errorbar(ks_present, accs, yerr=sems,
                    marker=marker, markersize=6, color=color,
                    linewidth=1.8, capsize=3, capthick=1,
                    label=display_name(model))

    ax.tick_params(axis='both', labelsize=18)
    ax.set_xscale("log")
    ax.set_xlabel(r"Gap (log scale)", fontsize=28)
    ax.set_ylabel(r"Accuracy", fontsize=28)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label=r"Random")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot: Positive vs Negative accuracy
# ─────────────────────────────────────────────

def plot_pos_neg_accuracy(all_data: dict, save_path: Path | None = None) -> None:
    """Two subplots: positive accuracy and negative accuracy vs k."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    k_values = _get_k_values(all_data)

    for ax_idx, (label, gt) in enumerate([("VALID", "VALID"), ("INVALID", "INVALID")]):
        ax = axes[ax_idx]

        for m_idx, (model, data) in enumerate(all_data.items()):
            accs, sems, ks_present = [], [], []

            for k in k_values:
                k_str = str(k)
                if k_str not in data["results_by_k"]:
                    continue
                traces = [t for t in data["results_by_k"][k_str]["traces"]
                          if t["ground_truth"] == gt]
                if not traces:
                    continue
                acc, sem = accuracy_with_sem(traces)
                accs.append(acc)
                sems.append(sem)
                ks_present.append(k)

            if not ks_present:
                continue

            color, marker = _model_style(m_idx)
            ax.errorbar(ks_present, accs, yerr=sems,
                        marker=marker, markersize=5, color=color,
                        linewidth=1.5, capsize=3, capthick=1,
                        label=display_name(model))

        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ (log scale)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r"\textbf{" + label + r" Traces}", fontsize=13)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=min(len(by_label), 5),
               bbox_to_anchor=(0.5, -0.08))

    axes[0].set_ylabel(r"Accuracy")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(all_data: dict) -> None:
    k_values = _get_k_values(all_data)

    for model, data in all_data.items():
        print(f"\n{'='*65}")
        print(f"  {display_name(model)}")
        print(f"{'='*65}")
        print(f"  {'k':>6} | {'Balanced':>9} | {'Pos':>7} | {'Neg':>7} | {'Overall':>8}")
        print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

        for k in k_values:
            k_str = str(k)
            if k_str not in data["results_by_k"]:
                continue
            traces = data["results_by_k"][k_str]["traces"]
            if not traces:
                continue

            bal, bal_sem = balanced_accuracy_with_sem(traces)
            pos = [t for t in traces if t["ground_truth"] == "VALID"]
            neg = [t for t in traces if t["ground_truth"] == "INVALID"]
            p_acc = sum(1 for t in pos if t["correct"]) / len(pos) if pos else 0
            n_acc = sum(1 for t in neg if t["correct"]) / len(neg) if neg else 0
            overall = sum(1 for t in traces if t["correct"]) / len(traces)

            print(f"  {k:>6} | {bal:>7.1%}±{bal_sem:.0%} | {p_acc:>7.1%} | "
                  f"{n_acc:>7.1%} | {overall:>7.1%}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tree-LTL: Plot results from all models"
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models")
    parser.add_argument("--b", type=int, default=None,
                        help="Filter to specific branching factor")
    parser.add_argument("--d", type=int, default=None,
                        help="Filter to specific tree depth")
    parser.add_argument("--results-dir", type=str, default="results-tree",
                        help="Directory containing result JSON files")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_data = load_all_results(results_dir, model_filter=args.models,
                                b_filter=args.b, d_filter=args.d)

    if not all_data:
        print(f"No result files found in {results_dir}/")
        return

    # Build filename suffix from b,d if filtered
    suffix = ""
    if args.b is not None or args.d is not None:
        b_str = f"b{args.b}" if args.b else "b*"
        d_str = f"d{args.d}" if args.d else "d*"
        suffix = f"_{b_str}_{d_str}"

    print_summary(all_data)

    print("\nGenerating plots...")

    plot_balanced_accuracy(
        all_data,
        save_path=plots_dir / f"tree_accuracy{suffix}.pdf",
    )

    plot_pos_neg_accuracy(
        all_data,
        save_path=plots_dir / f"tree_ltl_pos_neg_accuracy{suffix}.pdf",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()