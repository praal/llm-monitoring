"""
Proposition Scaling × Tree-LTL — Plotting
===========================================

Reads result files from results-prop-tree/ and plots balanced accuracy
vs proposition scale factor (entities per step).

Usage:
    python plot_prop_tree.py
    python plot_prop_tree.py --models google/gemini-2.5-pro openai/gpt-4.1
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

RESULTS_DIR = Path("results-prop-tree")
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
    for fp in sorted(results_dir.glob("results_prop_tree_*.json")):
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
# Metrics
# ─────────────────────────────────────────────

def _extract_seed(sample):
    """Extract seed from sample dict, falling back to parsing sample_id."""
    if "seed" in sample and sample["seed"] != 0:
        return sample["seed"]
    # Parse from sample_id: ne{N}_seed{SEED}_pos{i} or ne{N}_seed{SEED}_neg{i}
    sid = sample.get("sample_id", "")
    import re
    m = re.search(r'seed(\d+)', sid)
    if m:
        return int(m.group(1))
    return 0


def compute_metrics_pooled(samples):
    """Compute balanced accuracy with pooled binomial SEM across all samples."""
    n = len(samples)
    if n == 0:
        return {}

    correct = sum(1 for r in samples if r["correct"])
    accuracy = correct / n
    acc_sem = np.sqrt(accuracy * (1 - accuracy) /  n)

    pos = [r for r in samples if r["ground_truth"] == "VALID"]
    neg = [r for r in samples if r["ground_truth"] == "INVALID"]
    pos_acc = sum(1 for r in pos if r["correct"]) / len(pos) if pos else 0
    neg_acc = sum(1 for r in neg if r["correct"]) / len(neg) if neg else 0
    bal_acc = (pos_acc + neg_acc) / 2

    sem_pos = np.sqrt(pos_acc * (1 - pos_acc) / len(pos)) if pos else 0
    sem_neg = np.sqrt(neg_acc * (1 - neg_acc) / len(neg)) if neg else 0
    bal_sem = np.sqrt(sem_pos**2 + sem_neg**2) / 3

    # Count seeds present (parse from sample_id if seed field missing)
    seeds = set(_extract_seed(s) for s in samples)

    return {
        "accuracy": accuracy, "acc_sem": acc_sem,
        "bal_acc": bal_acc, "bal_sem": bal_sem,
        "pos_acc": pos_acc, "neg_acc": neg_acc,
        "n_valid": len(pos), "n_invalid": len(neg),
        "n_seeds": len(seeds),
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
# Plot: Balanced accuracy vs N entities
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
            m = compute_metrics_pooled(samples)
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
    ax.set_xlabel(r"Proposition Scale Factor", fontsize=28)
    ax.set_xticks(n_values)
    ax.set_xticklabels([f"{n}" for n in n_values])
    ax.set_ylabel(r" Accuracy", fontsize=28)
    ax.set_ylim(0.3, 1.02)
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label=r"Random")
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=10)
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
                subset = [s for s in samples if s["ground_truth"] == gt]
                if not subset:
                    continue
                p = sum(1 for s in subset if s["correct"]) / len(subset)
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

        ax.set_xlabel(r"Proposition Scale Factor")
        ax.set_xticks(n_values)
        ax.set_xticklabels([f"{n}" for n in n_values])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r"\textbf{" + label + r" Traces}", fontsize=13)
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
        print(f"\n{'='*70}")
        print(f"  {display_name(model)}")
        print(f"{'='*70}")
        print(f"  {'N':>4} | {'BalAcc':>12} | {'Pos':>7} | {'Neg':>7} | "
              f"{'Acc':>10} | {'Seeds':>5}")
        print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*5}")

        for n in n_values:
            n_str = str(n)
            if n_str not in data["results_by_n"]:
                continue
            samples = data["results_by_n"][n_str]["samples"]
            if not samples:
                continue
            m = compute_metrics_pooled(samples)
            n_seeds = m.get("n_seeds", 1)
            print(f"  {n:>4} | {m['bal_acc']:>6.1%}±{m['bal_sem']:>4.1%} | "
                  f"{m['pos_acc']:>6.1%} | {m['neg_acc']:>6.1%} | "
                  f"{m['accuracy']:>5.1%}±{m['acc_sem']:>3.1%} | {n_seeds:>5}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prop-Tree: Plot results from all models"
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="Only plot these N values (e.g., 1 2 4 8 16)")
    parser.add_argument("--results-dir", type=str, default="results-prop-tree")
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

    plot_balanced_accuracy(
        all_data, n_filter=args.n_values,
        save_path=plots_dir / "prop_tree_accuracy.pdf",
    )



    print("\nDone!")


if __name__ == "__main__":
    main()