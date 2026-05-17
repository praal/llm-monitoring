"""
Multi-Constraint Simple Formula — Plotting
============================================

Reads result files from results-multi-simple/ and plots accuracy
vs number of constraints.

Usage:
    python plot_multi_simple.py
    python plot_multi_simple.py --models google/gemini-2.5-pro openai/gpt-4.1
    python plot_multi_simple.py --n_values 1 5 10 20
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

RESULTS_DIR = Path("results-multi-simple")
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
                     glob_pattern: str = "results_multi_simple_*.json") -> dict:
    all_data = {}
    for fp in sorted(results_dir.glob(glob_pattern)):
        with open(fp) as f:
            data = json.load(f)
        model = data["model"]
        if model_filter and model not in model_filter:
            continue
        all_data[model] = data
        n_samples = sum(len(v["samples"]) for v in data["results_by_n"].values())
        print(f"  Loaded {n_samples:>5} samples: {model}")
    return all_data


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.split("/")[-1])


# ─────────────────────────────────────────────
# Metrics (per-constraint, pooled binomial SEM)
# ─────────────────────────────────────────────

def compute_metrics(samples):
    """Compute per-constraint accuracy with pooled binomial SEM."""
    all_c = [c for s in samples for c in s["per_constraint"]]
    n = len(all_c)
    if n == 0:
        return {}

    correct = sum(1 for c in all_c if c["correct"])
    accuracy = correct / n
    acc_sem = np.sqrt(accuracy * (1 - accuracy) / n)

    # Per-constraint positive = VALID, negative = INVALID
    pos = [c for c in all_c if c["ground_truth"] == "VALID"]
    neg = [c for c in all_c if c["ground_truth"] == "INVALID"]
    pos_acc = sum(1 for c in pos if c["correct"]) / len(pos) if pos else 0
    neg_acc = sum(1 for c in neg if c["correct"]) / len(neg) if neg else 0
    bal_acc = (pos_acc + neg_acc) / 2

    sem_pos = np.sqrt(pos_acc * (1 - pos_acc) / len(pos)) if pos else 0
    sem_neg = np.sqrt(neg_acc * (1 - neg_acc) / len(neg)) if neg else 0
    bal_sem = np.sqrt(sem_pos**2 + sem_neg**2) / 2

    # F1 with INVALID as positive class
    tp = sum(1 for c in all_c
             if c["ground_truth"] == "INVALID" and c["prediction"] == "INVALID")
    fp = sum(1 for c in all_c
             if c["ground_truth"] == "VALID" and c["prediction"] == "INVALID")
    fn = sum(1 for c in all_c
             if c["ground_truth"] == "INVALID" and c["prediction"] != "INVALID")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": accuracy, "acc_sem": acc_sem,
        "bal_acc": bal_acc, "bal_sem": bal_sem,
        "pos_acc": pos_acc, "neg_acc": neg_acc,
        "f1": f1, "precision": prec, "recall": rec,
        "n_valid": len(pos), "n_invalid": len(neg),
        "total": n,
    }


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


def _get_n_values(all_data: dict, n_filter: list[int] | None = None) -> list[int]:
    ns = set()
    for data in all_data.values():
        for n_str in data["results_by_n"]:
            ns.add(int(n_str))
    if n_filter:
        ns = ns & set(n_filter)
    return sorted(ns)


# ─────────────────────────────────────────────
# Plot: Balanced accuracy vs N constraints
# ─────────────────────────────────────────────

def plot_balanced_accuracy(all_data: dict, n_filter=None,
                           save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_values = _get_n_values(all_data, n_filter)

    for idx, (model, data) in enumerate(all_data.items()):
        accs, sems, ns_present = [], [], []

        for n in n_values:
            n_str = str(n)
            if n_str not in data["results_by_n"]:
                continue
            samples = data["results_by_n"][n_str]["samples"]
            if not samples:
                continue
            m = compute_metrics(samples)
            accs.append(m["bal_acc"])
            sems.append(m["bal_sem"])
            ns_present.append(n)

        if not ns_present:
            continue

        color, marker = _model_style(idx)
        ax.errorbar(ns_present, accs, yerr=sems,
                    marker=marker, markersize=6, color=color,
                    linewidth=1.8, capsize=3, capthick=1,
                    label=display_name(model))

    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel(r"Number of Constraints", fontsize=28)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_ylabel(r"Accuracy", fontsize=28)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label=r"Random")
    ax.grid(True, alpha=0.3)
   # ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot: Positive vs Negative accuracy
# ─────────────────────────────────────────────

def plot_pos_neg(all_data: dict, n_filter=None,
                 save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    n_values = _get_n_values(all_data, n_filter)

    for ax_idx, (label, gt) in enumerate([("VALID", "VALID"), ("INVALID", "INVALID")]):
        ax = axes[ax_idx]

        for m_idx, (model, data) in enumerate(all_data.items()):
            accs, sems, ns_present = [], [], []

            for n in n_values:
                n_str = str(n)
                if n_str not in data["results_by_n"]:
                    continue
                samples = data["results_by_n"][n_str]["samples"]
                if not samples:
                    continue
                all_c = [c for s in samples for c in s["per_constraint"]]
                subset = [c for c in all_c if c["ground_truth"] == gt]
                if not subset:
                    continue
                p = sum(1 for c in subset if c["correct"]) / len(subset)
                sem = np.sqrt(p * (1 - p) / len(subset))
                accs.append(p)
                sems.append(sem)
                ns_present.append(n)

            if not ns_present:
                continue

            color, marker = _model_style(m_idx)
            ax.errorbar(ns_present, accs, yerr=sems,
                        marker=marker, markersize=5, color=color,
                        linewidth=1.5, capsize=3, capthick=1,
                        label=display_name(model))

        ax.set_xlabel(r"Number of Constraints ($N$)")
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r"\textbf{" + label + r" Constraints}", fontsize=13)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, label=r"Random")
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
# Legend export
# ─────────────────────────────────────────────

def export_legend(all_data: dict, save_path: Path | None = None) -> None:
    fig_dummy, ax_dummy = plt.subplots()
    for idx, model in enumerate(all_data.keys()):
        color, marker = _model_style(idx)
        ax_dummy.plot([], [], marker=marker, color=color,
                      linewidth=1.8, markersize=8,
                      label=display_name(model))
    ax_dummy.plot([], [], color="black", linestyle=":", linewidth=1.8,
                  label=r"Random")
    plt.close(fig_dummy)

    handles, labels = ax_dummy.get_legend_handles_labels()
    fig_leg = plt.figure(figsize=(len(labels) * 2.5, 0.6))
    fig_leg.legend(handles, labels, loc="center", ncol=len(labels),
                   frameon=False, fontsize=16)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_leg.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(all_data: dict, n_filter=None) -> None:
    n_values = _get_n_values(all_data, n_filter)

    for model, data in all_data.items():
        print(f"\n{'='*75}")
        print(f"  {display_name(model)}")
        print(f"{'='*75}")
        print(f"  {'N':>4} | {'BalAcc':>10} | {'Pos':>7} | {'Neg':>7} | "
              f"{'Acc':>8} | {'F1':>6} | {'Total':>6}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-"
              f"{'-'*8}-+-{'-'*6}-+-{'-'*6}")

        for n in n_values:
            n_str = str(n)
            if n_str not in data["results_by_n"]:
                continue
            samples = data["results_by_n"][n_str]["samples"]
            if not samples:
                continue
            m = compute_metrics(samples)
            print(f"  {n:>4} | {m['bal_acc']:>5.1%}±{m['bal_sem']:>3.1%} | "
                  f"{m['pos_acc']:>6.1%} | {m['neg_acc']:>6.1%} | "
                  f"{m['accuracy']:>7.1%} | {m['f1']:>5.1%} | {m['total']:>6}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Simple: Plot results from all models"
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="Only plot these N values (e.g., 1 5 10 20)")
    parser.add_argument("--results-dir", type=str, default="results-multi-simple")
    parser.add_argument("--per-dir", type=str, default="results-multi-simple-per",
                        help="Per-constraint results directory")
    parser.add_argument("--plots-dir", type=str, default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    per_dir = Path(args.per_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load joint results
    print("Loading joint results...")
    joint_data = load_all_results(results_dir, model_filter=args.models)

    # Load per-constraint results
    print("Loading per-constraint results...")
    per_data = load_all_results(per_dir, model_filter=args.models,
                                glob_pattern="results_multi_simple_per_*.json")

    if not joint_data and not per_data:
        print(f"No result files found in {results_dir}/ or {per_dir}/")
        return

    # Joint plots
    if joint_data:
        print("\n--- Joint (all constraints in one prompt) ---")
        print_summary(joint_data, n_filter=args.n_values)

        plot_balanced_accuracy(
            joint_data, n_filter=args.n_values,
            save_path=plots_dir / "multi_simple_accuracy.pdf",
        )
        plot_pos_neg(
            joint_data, n_filter=args.n_values,
            save_path=plots_dir / "multi_simple_pos_neg.pdf",
        )


    # Per-constraint plots
    if per_data:
        print("\n--- Per-constraint (one prompt per constraint) ---")
        print_summary(per_data, n_filter=args.n_values)

        plot_balanced_accuracy(
            per_data, n_filter=args.n_values,
            save_path=plots_dir / "multi_simple_per_accuracy.pdf",
        )
        plot_pos_neg(
            per_data, n_filter=args.n_values,
            save_path=plots_dir / "multi_simple_per_pos_neg.pdf",
        )


    print("\nDone!")


if __name__ == "__main__":
    main()