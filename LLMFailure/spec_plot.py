#!/usr/bin/env python3
"""
Plot results from the spec-level experiment.

Usage:
    python plot_spec_level.py --results results-spec-level/results_*.json
    python plot_spec_level.py --results results-spec-level/
    python plot_spec_level.py --results results-spec-level/ --output plots/ --format png

Generates:
    1. Grouped bar chart: BalAcc by pattern × spec level (per model)
    2. Per-edge-case breakdown (per model × pattern)
"""
from __future__ import annotations

import json
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ────────────────────────────────────────────────────────────────

PATTERN_NAMES = {
    1: "Universality\nG(P)",
    2: "Absence\nG(¬P)",
    3: "Response\nG(P→F(S))",
    4: "Abs/Between\nG((Q∧¬R∧◇R)→(¬P U R))",
    5: "Constr. Resp\nG(P→(¬Q U R))",
    6: "Tree\n(b=2, d=1)",
    7: "Tree\n(b=2, d=4)",
}

PATTERN_NAMES_SHORT = {
    1: "Universality",
    2: "Absence",
    3: "Response",
    4: "Abs/Between",
    5: "Constr. Resp",
    6: "Tree(b2,d1)",
    7: "Tree(b2,d4)",
}

LEVEL_ORDER = ["informal", "precise", "precise_ltl"]
LEVEL_LABELS_SHORT = {
    "informal": "Informal NL",
    "precise": "Precise NL",
    "precise_ltl": "Precise NL + LTL",
}

LEVEL_COLORS = {
    "informal": "#377eb8",   # Wong orange
    "precise": "#ff7f00",    # Wong blue
    "precise_ltl": "#4daf4a", # Wong green
}


MODEL_DISPLAY = {
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1-Mini",
    "gpt-4o-mini": "GPT-4o-Mini",
    "claude-3.5-haiku": "Claude-3.5-Haiku",
    "llama-3.3-70B-instruct": "LLaMA-3.3-70B",

}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_results(paths: list[str]) -> dict[str, dict]:
    """Load result files. Returns {model_name: results_dict}."""
    all_results = {}
    for path in paths:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "results_*.json")))
        else:
            files = [path]

        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            model = data.get("model", os.path.basename(f))
            display_name = model.split("/")[-1] if "/" in model else model
            all_results[display_name] = data

    return all_results


def extract_metrics_table(results: dict) -> dict:
    """
    Extract a structured table from results.
    Returns {(pattern_id, spec_level): metrics_dict}
    """
    table = {}
    for key, data in results.get("results_by_key", {}).items():
        m = data.get("metrics", {})
        if not m:
            continue
        match = re.match(r"p(\d+)_(.*)", key)
        if match:
            pid = int(match.group(1))
            level = match.group(2)
            table[(pid, level)] = m
    return table


def extract_edge_case_breakdown(results: dict) -> dict:
    """
    Extract per-edge-case accuracy.
    Returns {(pattern_id, spec_level, edge_tag): (n_correct, n_total)}
    """
    breakdown = defaultdict(lambda: [0, 0])
    for key, data in results.get("results_by_key", {}).items():
        match = re.match(r"p(\d+)_(.*)", key)
        if not match:
            continue
        pid = int(match.group(1))
        level = match.group(2)
        for s in data.get("samples", []):
            tag = s.get("edge_case_tag", "none")
            if tag == "none":
                continue
            k = (pid, level, tag)
            breakdown[k][1] += 1
            if s.get("correct", False):
                breakdown[k][0] += 1
    return dict(breakdown)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _clean_tag(tag: str) -> str:
    return tag.replace("_", " ").title()


# ── Plot 1: Grouped bar chart ────────────────────────────────────────────────

def plot_grouped_bars(all_results: dict, output_dir: str, fmt: str):
    """
    For each model: grouped bar chart with patterns on x-axis,
    bars colored by spec level.
    """
    for model_name, results in all_results.items():
        table = extract_metrics_table(results)
        if not table:
            continue

        pids = sorted(set(p for p, _ in table.keys()))
        n_patterns = len(pids)
        n_levels = len(LEVEL_ORDER)
        bar_width = 0.22
        x = np.arange(n_patterns)

        fig, ax = plt.subplots(figsize=(max(7, n_patterns * 2.5), 4.5))

        for i, level in enumerate(LEVEL_ORDER):
            vals = []
            errs = []
            for pid in pids:
                m = table.get((pid, level), {})
                vals.append(m.get("bal_acc", 0))
                errs.append(m.get("bal_sem", 0))

            offset = (i - (n_levels - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                yerr=errs, capsize=3,
                label=LEVEL_LABELS_SHORT[level],
                color=LEVEL_COLORS[level],
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.0%}",
                        ha="center", va="bottom", fontsize=7,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([PATTERN_NAMES.get(p, f"P{p}") for p in pids],
                           fontsize=16, rotation=10, ha="right")
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_title(f"Spec Level Effect on LLM Accuracy — {MODEL_DISPLAY[model_name]}",
                     fontsize=18, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
      #  ax.legend(loc="upper right", fontsize=16)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, zorder=1)

        fig.tight_layout()
        fname = os.path.join(output_dir,
                             f"grouped_bars_{_slug(model_name)}.{fmt}")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot 2: Edge case breakdown ─────────────────────────────────────────────

def plot_edge_breakdown(all_results: dict, output_dir: str, fmt: str):
    """
    For each model: horizontal bar chart showing accuracy per edge case tag,
    grouped by spec level.
    """
    for model_name, results in all_results.items():
        breakdown = extract_edge_case_breakdown(results)
        if not breakdown:
            continue

        edge_pids = sorted(set(pid for pid, _, _ in breakdown.keys()))

        for pid in edge_pids:
            tags = sorted(set(tag for p, _, tag in breakdown.keys()
                              if p == pid))
            if not tags:
                continue

            fig, ax = plt.subplots(
                figsize=(8, max(3, len(tags) * 0.6 * len(LEVEL_ORDER))))

            y = np.arange(len(tags))
            h = 0.25

            for i, level in enumerate(LEVEL_ORDER):
                accs = []
                for tag in tags:
                    k = (pid, level, tag)
                    if k in breakdown:
                        correct, total = breakdown[k]
                        accs.append(correct / total if total > 0 else 0)
                    else:
                        accs.append(0)

                offset = (i - (len(LEVEL_ORDER) - 1) / 2) * h
                bars = ax.barh(
                    y + offset, accs, h,
                    label=LEVEL_LABELS_SHORT[level],
                    color=LEVEL_COLORS[level],
                    edgecolor="white", linewidth=0.5, zorder=3,
                )
                for bar, val in zip(bars, accs):
                    if val > 0:
                        ax.text(bar.get_width() + 0.02,
                                bar.get_y() + bar.get_height() / 2,
                                f"{val:.0%}", ha="left", va="center",
                                fontsize=8)

            ax.set_yticks(y)
            ax.set_yticklabels([_clean_tag(t) for t in tags], fontsize=18)
            ax.set_xlabel("Accuracy", fontsize=18)
            ax.set_xlim(0, 1.3)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.set_title(
                f"Edge Case Breakdown: "
                f"{PATTERN_NAMES_SHORT.get(pid, f'P{pid}')} — {model_name}",
                fontsize=11, fontweight="bold")
            ax.legend(fontsize=18, loc="lower right")
            ax.grid(axis="x", alpha=0.3, zorder=0)
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)

            fig.tight_layout()
            fname = os.path.join(
                output_dir,
                f"edge_breakdown_p{pid}_{_slug(model_name)}.{fmt}")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")


# ── Plot 3: Multi-model side-by-side subplots ────────────────────────────────

def plot_multi_model_subplots(all_results: dict, output_dir: str, fmt: str):
    """
    Put the per-model grouped bar charts side by side as subplots
    in a single figure.
    """
    if len(all_results) < 2:
        print("  Skipping multi-model subplot (need 2+ models)")
        return

    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models,
                             figsize=(6 * n_models, 5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    for col, (model_name, results) in enumerate(zip(model_names,
                                                      [all_results[m] for m in model_names])):
        ax = axes[col]
        table = extract_metrics_table(results)
        if not table:
            continue

        pids = sorted(set(p for p, _ in table.keys()))
        n_patterns = len(pids)
        n_levels = len(LEVEL_ORDER)
        bar_width = 0.22
        x = np.arange(n_patterns)

        for i, level in enumerate(LEVEL_ORDER):
            vals = []
            errs = []
            for pid in pids:
                m = table.get((pid, level), {})
                vals.append(m.get("bal_acc", 0))
                errs.append(m.get("bal_sem", 0))

            offset = (i - (n_levels - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                yerr=errs, capsize=3,
                label=LEVEL_LABELS_SHORT[level] if col == 0 else "",
                color=LEVEL_COLORS[level],
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )
            #for bar, val in zip(bars, vals):
                #if val > 0:
                  #  ax.text(
                   #     bar.get_x() + bar.get_width() / 2,
                   #     bar.get_height() + 0.02,
                   #     f"{val:.0%}",
                   #     ha="center", va="bottom", fontsize=6,
                   #     fontweight="bold",
                   # )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PATTERN_NAMES_SHORT.get(p, f"P{p}") for p in pids],
            fontsize=18, rotation=45, ha="right",
        )
        ax.set_title(model_name, fontsize=18, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, zorder=1)

    axes[0].set_ylabel("Accuracy", fontsize=18)
    axes[0].legend(loc="upper right", fontsize=18)

    fig.suptitle("Spec Level Effect on LLM Accuracy",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = os.path.join(output_dir, f"multi_model_subplots.{fmt}")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot spec-level experiment results")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Result JSON files or directory containing them")
    parser.add_argument("--compare", nargs="+", type=str, default=None,
                        help="Model names to include in the multi-model subplot "
                             "(partial match, e.g., 'flash haiku mini')")
    parser.add_argument("--output", type=str, default="plots-spec-level",
                        help="Output directory for plots")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading results...")
    all_results = load_results(args.results)
    print(f"  Loaded {len(all_results)} model(s): "
          f"{', '.join(all_results.keys())}")

    print("\n1. Grouped bar charts (per model):")
    plot_grouped_bars(all_results, args.output, args.format)

    print("\n2. Edge case breakdowns (per model × pattern):")
   # plot_edge_breakdown(all_results, args.output, args.format)

    # Filter models for comparison subplot
    if args.compare:
        compare_results = {}
        for model_name, data in all_results.items():
            if any(c.lower() in model_name.lower() for c in args.compare):
                compare_results[model_name] = data
        if compare_results:
            print(f"\n3. Multi-model subplots ({', '.join(compare_results.keys())}):")
            plot_multi_model_subplots(compare_results, args.output, args.format)
        else:
            print(f"\n3. No models matched --compare {args.compare}")
    else:
        print("\n3. Multi-model subplots (all models):")
        plot_multi_model_subplots(all_results, args.output, args.format)
    export_legend(args.output, args.format)
    print("\n4. Legend:")
    print(f"\nAll plots saved to {args.output}/")


def export_legend(output_dir: str, fmt: str):
    """Export the legend as a separate file."""
    fig, ax = plt.subplots(figsize=(6, 0.4))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LEVEL_COLORS[level])
        for level in LEVEL_ORDER
    ]
    labels = [LEVEL_LABELS_SHORT[level] for level in LEVEL_ORDER]
    legend = ax.legend(handles, labels, loc="center", ncol=len(LEVEL_ORDER),
                       fontsize=14, frameon=False)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fname = os.path.join(output_dir, f"legend_spec_level.{fmt}")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")



if __name__ == "__main__":
    main()