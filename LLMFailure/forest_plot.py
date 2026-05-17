"""
Multi-Constraint Tree-LTL — Plotting
=====================================

Reads result files from results-multi/ and plots F1 and balanced accuracy
vs number of constraints.

Usage:
    python plot_multi_constraint.py
    python plot_multi_constraint.py --models google/gemini-2.5-pro openai/gpt-4.1
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

RESULTS_DIR = Path("results-multi")
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
                     model_filter: list[str] | None = None) -> dict:
    all_data = {}
    for fp in sorted(results_dir.glob("results_multi_*.json")):
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
# Metrics (recomputed from raw per-constraint results)
# ─────────────────────────────────────────────

def compute_metrics(samples):
    """Compute F1 (INVALID as positive), balanced accuracy, and SEMs."""
    all_c = [c for s in samples for c in s["per_constraint"]]
    n = len(all_c)
    if n == 0:
        return {"f1": 0, "f1_sem": 0, "bal_acc": 0, "bal_sem": 0,
                "accuracy": 0, "acc_sem": 0}

    # F1 with INVALID as positive
    tp = sum(1 for c in all_c
             if c["ground_truth"] == "INVALID" and c["prediction"] == "INVALID")
    fp = sum(1 for c in all_c
             if c["ground_truth"] == "VALID" and c["prediction"] == "INVALID")
    fn = sum(1 for c in all_c
             if c["ground_truth"] == "INVALID" and c["prediction"] != "INVALID")

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # Bootstrap SEM for F1
    f1_sem = _bootstrap_sem_f1(all_c, n_boot=500)

    # Balanced accuracy
    pos = [c for c in all_c if c["ground_truth"] == "VALID"]
    neg = [c for c in all_c if c["ground_truth"] == "INVALID"]
    pos_acc = sum(1 for c in pos if c["correct"]) / len(pos) if pos else 0
    neg_acc = sum(1 for c in neg if c["correct"]) / len(neg) if neg else 0
    bal_acc = (pos_acc + neg_acc) / 2

    sem_pos = np.sqrt(pos_acc * (1 - pos_acc) / len(pos)) if pos else 0
    sem_neg = np.sqrt(neg_acc * (1 - neg_acc) / len(neg)) if neg else 0
    bal_sem = np.sqrt(sem_pos**2 + sem_neg**2) / 2

    # Overall accuracy
    correct = sum(1 for c in all_c if c["correct"])
    accuracy = correct / n
    acc_sem = np.sqrt(accuracy * (1 - accuracy) / n)

    return {
        "f1": f1, "f1_sem": f1_sem,
        "precision": prec, "recall": rec,
        "bal_acc": bal_acc, "bal_sem": bal_sem,
        "accuracy": accuracy, "acc_sem": acc_sem,
        "pos_acc": pos_acc, "neg_acc": neg_acc,
    }


def _bootstrap_sem_f1(all_constraints, n_boot=500):
    """Bootstrap SEM for F1 score."""
    rng = np.random.default_rng(42)
    n = len(all_constraints)
    f1s = []
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        sample = [all_constraints[i] for i in idxs]
        tp = sum(1 for c in sample
                 if c["ground_truth"] == "INVALID" and c["prediction"] == "INVALID")
        fp = sum(1 for c in sample
                 if c["ground_truth"] == "VALID" and c["prediction"] == "INVALID")
        fn = sum(1 for c in sample
                 if c["ground_truth"] == "INVALID" and c["prediction"] != "INVALID")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.std(f1s)


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
# Plot: F1 vs N constraints
# ─────────────────────────────────────────────

def plot_f1(all_data: dict, n_filter=None, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_values = _get_n_values(all_data, n_filter)

    for idx, (model, data) in enumerate(all_data.items()):
        f1s, sems, ns_present = [], [], []

        for n in n_values:
            n_str = str(n)
            if n_str not in data["results_by_n"]:
                continue
            samples = data["results_by_n"][n_str]["samples"]
            if not samples:
                continue
            m = compute_metrics(samples)
            f1s.append(m["f1"])
            sems.append(m["f1_sem"])
            ns_present.append(n)

        if not ns_present:
            continue

        color, marker = _model_style(idx)
        ax.errorbar(ns_present, f1s, yerr=sems,
                    marker=marker, markersize=6, color=color,
                    linewidth=1.8, capsize=3, capthick=1,
                    label=display_name(model))

    ax.set_xlabel(r"Number of Constraints ($N$)")
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_ylabel(r"F1 Score (INVALID as positive)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label=r"Random")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot: Balanced accuracy vs N constraints
# ─────────────────────────────────────────────

def plot_balanced_accuracy(all_data: dict, n_filter=None, save_path: Path | None = None) -> None:
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
    #ax.legend(fontsize=10)
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Plot: Combined F1 + Balanced Accuracy (two subplots)
# ─────────────────────────────────────────────

def plot_combined(all_data: dict, n_filter=None, save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    n_values = _get_n_values(all_data, n_filter)

    for ax_idx, (metric_key, sem_key, ylabel) in enumerate([
        ("f1", "f1_sem", r"F1 Score (INVALID as positive)"),
        ("bal_acc", "bal_sem", r"Balanced Accuracy"),
    ]):
        ax = axes[ax_idx]

        for m_idx, (model, data) in enumerate(all_data.items()):
            vals, sems, ns_present = [], [], []

            for n in n_values:
                n_str = str(n)
                if n_str not in data["results_by_n"]:
                    continue
                samples = data["results_by_n"][n_str]["samples"]
                if not samples:
                    continue
                m = compute_metrics(samples)
                vals.append(m[metric_key])
                sems.append(m[sem_key])
                ns_present.append(n)

            if not ns_present:
                continue

            color, marker = _model_style(m_idx)
            ax.errorbar(ns_present, vals, yerr=sems,
                        marker=marker, markersize=5, color=color,
                        linewidth=1.5, capsize=3, capthick=1,
                        label=display_name(model))

        ax.set_xlabel(r"Number of Constraints ($N$)")
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, label=r"Random")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=min(len(by_label), 5),
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    _save_and_show(fig, save_path)


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(all_data: dict, n_filter=None) -> None:
    n_values = _get_n_values(all_data, n_filter)

    for model, data in all_data.items():
        print(f"\n{'='*75}")
        print(f"  {display_name(model)}")
        print(f"{'='*75}")
        print(f"  {'N':>4} | {'F1':>8} | {'BalAcc':>8} | {'Prec':>7} | "
              f"{'Rec':>7} | {'PosAcc':>7} | {'NegAcc':>7}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-"
              f"{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for n in n_values:
            n_str = str(n)
            if n_str not in data["results_by_n"]:
                continue
            samples = data["results_by_n"][n_str]["samples"]
            if not samples:
                continue
            m = compute_metrics(samples)
            print(f"  {n:>4} | {m['f1']:>6.1%}±{m['f1_sem']:.0%} | "
                  f"{m['bal_acc']:>6.1%}±{m['bal_sem']:.0%} | "
                  f"{m['precision']:>6.1%} | {m['recall']:>6.1%} | "
                  f"{m['pos_acc']:>6.1%} | {m['neg_acc']:>6.1%}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Constraint: Plot results from all models"
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="Only plot these N values (e.g., 1 5 10 20)")
    parser.add_argument("--results-dir", type=str, default="results-multi")
    parser.add_argument("--plots-dir", type=str, default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_data = load_all_results(results_dir, model_filter=args.models)

    if not all_data:
        print(f"No result files found in {results_dir}/")
        return

def export_legend(all_data: dict, save_path: Path | None = None) -> None:
    """Export just the legend as a separate file, all entries in one row."""
    fig_dummy, ax_dummy = plt.subplots()
    for idx, model in enumerate(all_data.keys()):
        color, marker = _model_style(idx)
        ax_dummy.plot([], [], marker=marker, color=color,
                      linewidth=1.8, markersize=8,
                      label=display_name(model))
    # Add Random baseline
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
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Constraint: Plot results from all models"
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="Only plot these N values (e.g., 1 5 10 20)")
    parser.add_argument("--results-dir", type=str, default="results-multi")
    parser.add_argument("--plots-dir", type=str, default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_data = load_all_results(results_dir, model_filter=args.models)

    if not all_data:
        print(f"No result files found in {results_dir}/")
        return

    print_summary(all_data, n_filter=args.n_values)

    print("\nGenerating plots...")

    plot_f1(
        all_data, n_filter=args.n_values,
        save_path=plots_dir / "multi_constraint_f1.pdf",
    )

    plot_balanced_accuracy(
        all_data, n_filter=args.n_values,
        save_path=plots_dir / "multi_constraint_hard_accuracy.pdf",
    )

    plot_combined(
        all_data, n_filter=args.n_values,
        save_path=plots_dir / "multi_constraint_combined.pdf",
    )

    export_legend(
        all_data,
        save_path=plots_dir / "multi_constraint_legend.pdf",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()