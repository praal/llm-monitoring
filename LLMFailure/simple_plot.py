"""
Temporal Elasticity (simple formula) — Plotting
================================================

Reads all result files from results/ and generates plots.

Usage:
    python simple_plot.py
    python simple_plot.py --models google/gemini-2.5-flash openai/gpt-4.1
    python simple_plot.py --results-dir path/to/results
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

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

GAPS = [1, 10, 50, 100, 500, 1000]

MODEL_DISPLAY = {
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
    "openai/gpt-4.1": "GPT-4.1",
    "openai/gpt-4o-mini": "GPT-4o-Mini",
    "anthropic/claude-3.5-haiku": "Claude-3.5-Haiku",
    "meta-llama/llama-3.3-70B-instruct": "LLaMA-3.3-70B",
    "meta-llama/llama-3.1-8b-instruct": "LLaMA-3.1-8B",
    "qwen/qwen-2.5-7b-instruct": "Qwen-2.5-7B",
}

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_all_results(results_dir: Path,
                     model_filter: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Load all simple_*.json files from results_dir.
    Returns {model_name: [result_dicts]}.
    """
    all_data = {}
    for fp in sorted(results_dir.glob("simple_*.json")):
        with open(fp) as f:
            data = json.load(f)
        model = data["model"]

        if model_filter and model not in model_filter:
            continue

        all_data[model] = data["results"]
        print(f"  Loaded {len(data['results']):>5} results: {model}")

    return all_data


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.split("/")[-1])


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def accuracy_with_sem(results: list[dict]) -> tuple[float, float]:
    """Returns (accuracy, SEM)."""
    n = len(results)
    if n == 0:
        return 0.0, 0.0
    p = sum(1 for r in results if r["correct"]) / n
    sem = np.sqrt(p * (1 - p) / n) / 2
    return p, sem


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _save_and_show(fig, save_path: Path | None) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {save_path}")
    plt.show()


def _model_style(idx: int):
    """Return (color, marker) for model index."""
    colors = plt.cm.tab10.colors
    return colors[idx % len(colors)], MARKERS[idx % len(MARKERS)]


# ─────────────────────────────────────────────
# Plot 1: Unified — overall accuracy vs gap
# ─────────────────────────────────────────────

def plot_unified_accuracy(all_data: dict, save_path: Path | None = None) -> None:
    """Single plot: overall accuracy vs gap for all models."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, (model, results) in enumerate(all_data.items()):
        accs, sems, gaps_present = [], [], []

        if model not in MODEL_DISPLAY:
            print(model)
            continue
        for gap in GAPS:
            subset = [r for r in results if r["gap"] == gap]
            if not subset:
                continue
            acc, sem = accuracy_with_sem(subset)
            accs.append(acc)
            sems.append(sem)
            gaps_present.append(gap)

        if not gaps_present:
            continue

        color, marker = _model_style(idx)
        ax.errorbar(gaps_present, accs, yerr=sems,
                    marker=marker, markersize=6, color=color,
                    linewidth=1.8, capsize=3, capthick=1,
                    label=display_name(model))

    ax.tick_params(axis='both', labelsize=18)
    ax.set_xscale("log")
    ax.set_xlabel(r"Gap (log scale)", fontsize=28)
    ax.set_ylabel(r"Accuracy", fontsize=28)
    ax.set_ylim(0.40, 1.02)
   # ax.set_title(r"\textbf{Temporal Elasticity: Overall Accuracy vs Gap}",
   #              fontsize=14)
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label=r"Random")
    ax.grid(True, alpha=0.3)
   # ax.legend(fontsize=10)
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot 2: Accuracy by trace type (one subplot per type, all models)
# ─────────────────────────────────────────────

def plot_by_trace_type(all_data: dict, save_path: Path | None = None) -> None:
    """One subplot per trace type, all models overlaid."""
    trace_types = ["satisfied", "violated", "distractor"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax_idx, ttype in enumerate(trace_types):
        ax = axes[ax_idx]

        for m_idx, (model, results) in enumerate(all_data.items()):
            accs, sems, gaps_present = [], [], []

            for gap in GAPS:
                subset = [r for r in results
                          if r["gap"] == gap and r["trace_type"] == ttype]
                if not subset:
                    continue
                acc, sem = accuracy_with_sem(subset)
                accs.append(acc)
                sems.append(sem)
                gaps_present.append(gap)

            if not gaps_present:
                continue

            color, marker = _model_style(m_idx)
            ax.errorbar(gaps_present, accs, yerr=sems,
                        marker=marker, markersize=5, color=color,
                        linewidth=1.5, capsize=3, capthick=1,
                        label=display_name(model))

        ax.set_xscale("log")
        ax.set_xlabel(r"Gap (log scale)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r"\textbf{Trace Type: " + ttype.capitalize() + r"}",
                     fontsize=13)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=min(len(by_label), 5),
               bbox_to_anchor=(0.5, -0.08))

    axes[0].set_ylabel(r"Accuracy")
   # fig.suptitle(r"\textbf{Accuracy by Trace Type}",
    #             fontsize=15)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot 3: Accuracy by A-position
# ─────────────────────────────────────────────

def plot_by_position(all_data: dict, save_path: Path | None = None) -> None:
    """One subplot per A-position (early, late), all models overlaid."""
    positions = ["early", "late"]
    pos_titles = {
        "early": r"$A$ at steps 1--50",
        "late": r"$A$ at steps 101--200",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax_idx, pos in enumerate(positions):
        ax = axes[ax_idx]

        for m_idx, (model, results) in enumerate(all_data.items()):
            accs, sems, gaps_present = [], [], []

            for gap in GAPS:
                subset = [r for r in results
                          if r["gap"] == gap and r["a_position_label"] == pos]
                if not subset:
                    continue
                acc, sem = accuracy_with_sem(subset)
                accs.append(acc)
                sems.append(sem)
                gaps_present.append(gap)

            if not gaps_present:
                continue

            color, marker = _model_style(m_idx)
            ax.errorbar(gaps_present, accs, yerr=sems,
                        marker=marker, markersize=5, color=color,
                        linewidth=1.5, capsize=3, capthick=1,
                        label=display_name(model))

        ax.set_xscale("log")
        ax.set_xlabel(r"Gap (log scale)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r"\textbf{" + f"$A$ position: {pos} ({pos_titles[pos]})" + r"}",
                     fontsize=13)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=min(len(by_label), 5),
               bbox_to_anchor=(0.5, -0.08))

    axes[0].set_ylabel(r"Accuracy")
    fig.suptitle(r"\textbf{Accuracy vs Gap by $A$-Position}",
                 fontsize=15)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot 4: Distractor false satisfaction rate
# ─────────────────────────────────────────────

def plot_distractor_analysis(all_data: dict, save_path: Path | None = None) -> None:
    """How often does each model say SATISFIED on distractor traces?"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for m_idx, (model, results) in enumerate(all_data.items()):
        rates, sems, gaps_present = [], [], []

        for gap in GAPS:
            subset = [r for r in results
                      if r["gap"] == gap and r["trace_type"] == "distractor"]
            if not subset:
                continue

            n = len(subset)
            k = sum(1 for r in subset if r["predicted"] == "satisfied")
            rate = k / n
            sem = np.sqrt(rate * (1 - rate) / n)

            rates.append(rate)
            sems.append(sem)
            gaps_present.append(gap)

        if not gaps_present:
            continue

        color, marker = _model_style(m_idx)
        ax.errorbar(gaps_present, rates, yerr=sems,
                    marker=marker, markersize=6, color=color,
                    linewidth=1.5, capsize=3, capthick=1,
                    label=display_name(model))

    ax.set_xscale("log")
    ax.set_xlabel(r"Gap (log scale)")
    ax.set_ylabel(r"False Satisfaction Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"\textbf{Distractor Analysis:}" "\n"
                 r"How often does the LLM say SATISFIED" "\n"
                 r"when $B$ appears only BEFORE $A$?",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(all_data: dict) -> None:
    for model, results in all_data.items():
        print(f"\n{'='*65}")
        print(f"  {display_name(model)}  —  {len(results)} traces")
        print(f"{'='*65}")
        print(f"  {'Gap':>5} | {'Overall':>7} | {'Satisf':>7} | {'Violat':>7} | {'Distr':>7}")
        print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for gap in GAPS:
            sub_all = [r for r in results if r["gap"] == gap]
            sub_sat = [r for r in sub_all if r["trace_type"] == "satisfied"]
            sub_vio = [r for r in sub_all if r["trace_type"] == "violated"]
            sub_dis = [r for r in sub_all if r["trace_type"] == "distractor"]

            a_all = accuracy_with_sem(sub_all)[0] if sub_all else float("nan")
            a_sat = accuracy_with_sem(sub_sat)[0] if sub_sat else float("nan")
            a_vio = accuracy_with_sem(sub_vio)[0] if sub_vio else float("nan")
            a_dis = accuracy_with_sem(sub_dis)[0] if sub_dis else float("nan")

            print(f"  {gap:>5} | {a_all:>7.3f} | {a_sat:>7.3f} | "
                  f"{a_vio:>7.3f} | {a_dis:>7.3f}")

        overall = accuracy_with_sem(results)[0]
        print(f"  {'ALL':>5} | {overall:>7.3f} |")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Elasticity (simple formula): plot results"
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result JSON files")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_data = load_all_results(results_dir, model_filter=args.models)

    if not all_data:
        print(f"No result files found in {results_dir}/")
        return

    print_summary(all_data)

    print("\nGenerating plots...")

    plot_unified_accuracy(
        all_data,
        save_path=plots_dir / "case1_unified_accuracy.pdf",
    )

    plot_by_trace_type(
        all_data,
        save_path=plots_dir / "simple_by_trace_type.pdf",
    )

    plot_by_position(
        all_data,
        save_path=plots_dir / "simple_by_position.pdf",
    )

    plot_distractor_analysis(
        all_data,
        save_path=plots_dir / "simple_distractor_analysis.pdf",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()