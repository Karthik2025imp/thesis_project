"""
aggregate_results.py -- combine per-process summary CSVs and plot loss
curves, NLL-loss variant.

Run after all 5 prefix_run_{0..4}.py processes have completed:
    cd ~/project
    python3 src/gcg_complete/new/instruct-nll-scaled/aggregate_results.py

Reads:  insertion_nll_summary_p{0..4}.csv
Writes: insertion_nll_summary_all.csv  (combined)
        insertion_nll_loss_curves.png  (per-prompt trace, faceted by acceptance_mode)
        insertion_nll_aggregate.txt    (printed stats)
"""

import os
import sys
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")

with open("configs/attack_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

N_PROCESSES = 5
LOG_DIR = 'logs/gcg_complete/instruct-nll-scaled'
K_INSERTIONS = 5
VARIANTS = ["prefix"]
ACCEPTANCE_MODES = ["strict", "unconditional"]

SUMMARY_PATHS = [
    os.path.join(LOG_DIR, f"insertion_nll_summary_p{i}.csv")
    for i in range(N_PROCESSES)
]
ALL_SUMMARY_PATH = os.path.join(LOG_DIR, "insertion_nll_summary_all.csv")
PLOT_PATH = os.path.join(LOG_DIR, "insertion_nll_loss_curves.png")
STATS_PATH = os.path.join(LOG_DIR, "insertion_nll_aggregate.txt")


def load_and_combine():
    frames = []
    for i, path in enumerate(SUMMARY_PATHS):
        if not os.path.exists(path):
            print(f"  [WARNING] Process {i} summary not found: {path} -- skipping")
            continue
        df = pd.read_csv(path)
        print(f"  Process {i}: {len(df)} completed runs from {path}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No summary CSVs found. Have any processes completed?")

    combined = pd.concat(frames, ignore_index=True)

    # acceptance_mode must be in the dedup subset, or a prompt's "strict"
    # and "unconditional" rows collide as duplicates of each other.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["prompt_idx", "variant", "acceptance_mode"], keep="last")
    after = len(combined)
    if before != after:
        print(f"  [INFO] Dropped {before - after} duplicate (prompt_idx, variant, "
              f"acceptance_mode) rows (kept last)")

    combined = combined.sort_values(["prompt_idx", "variant", "acceptance_mode"]).reset_index(drop=True)
    return combined


def print_aggregate_stats(df, out_file=None):
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"AGGREGATE RESULTS (NLL loss) -- {df['prompt_idx'].nunique()} unique prompts, "
                 f"{len(df)} total rows")
    lines.append(f"{'='*70}")

    for variant in VARIANTS:
        for acceptance_mode in ACCEPTANCE_MODES:
            sub = df[(df["variant"] == variant) & (df["acceptance_mode"] == acceptance_mode)]
            if len(sub) == 0:
                continue
            lines.append(f"\nVariant: {variant}  |  acceptance_mode: {acceptance_mode}  (n={len(sub)})")

            lines.append(f"  -- walk endpoint --")
            lines.append(f"  mean NLL cost change:     {sub['cost_pct_change'].mean():+.1f}%")
            lines.append(f"  median NLL cost change:   {sub['cost_pct_change'].median():+.1f}%")
            lines.append(f"  mean y_refusal_rate:      {sub['y_refusal_rate'].mean():.3f}")
            lines.append(f"  mean yprime_refusal_rate: {sub['yprime_refusal_rate'].mean():.3f}")
            mean_change = sub['refusal_rate_change'].mean()
            lines.append(f"  mean refusal_rate_change: {mean_change:+.3f}")
            n_improved = (sub["refusal_rate_change"] < 0).sum()
            n_unchanged = (sub["refusal_rate_change"] == 0).sum()
            n_worsened = (sub["refusal_rate_change"] > 0).sum()
            lines.append(f"  refusal rate improved:    {n_improved}/{len(sub)}")
            lines.append(f"  refusal rate unchanged:   {n_unchanged}/{len(sub)}")
            lines.append(f"  refusal rate worsened:    {n_worsened}/{len(sub)}")

            if "best_nll_cost" in sub.columns:
                lines.append(f"  -- best point observed at any iteration --")
                lines.append(f"  mean best NLL cost change: {sub['best_cost_pct_change'].mean():+.1f}%")
                lines.append(f"  mean best_refusal_rate:    {sub['best_refusal_rate'].mean():.3f}")
                mean_best_change = sub['best_refusal_rate_change'].mean()
                lines.append(f"  mean best_refusal_rate_change: {mean_best_change:+.3f}")
                n_best_improved = (sub["best_refusal_rate_change"] < 0).sum()
                lines.append(f"  best refusal rate improved: {n_best_improved}/{len(sub)}")

                drift = sub["final_nll_cost"] - sub["best_nll_cost"]
                lines.append(f"  mean (final - best) cost drift: {drift.mean():+.4f}  "
                             f"(0 expected for 'strict'; >0 shows jitter cost for 'unconditional')")

    text = "\n".join(lines)
    print(text)

    if out_file:
        with open(out_file, "w") as f:
            f.write(text + "\n")
        print(f"\nStats written to: {out_file}")


def plot_loss_curves(df):
    if "cost_history" not in df.columns:
        print("No cost_history column found -- skipping plot.")
        return

    sub_all = df.dropna(subset=["cost_history"])
    if len(sub_all) == 0:
        print("All cost_history entries are NaN -- skipping plot.")
        return

    modes_present = [m for m in ACCEPTANCE_MODES if (sub_all["acceptance_mode"] == m).any()]
    if not modes_present:
        print("No rows match known acceptance_modes -- skipping plot.")
        return

    fig, axes = plt.subplots(1, len(modes_present), figsize=(7 * len(modes_present), 8), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("tab20")

    for ax, acceptance_mode in zip(axes, modes_present):
        sub = sub_all[sub_all["acceptance_mode"] == acceptance_mode]

        for i, (_, row) in enumerate(sub.iterrows()):
            trace = [float(v) for v in str(row["cost_history"]).split(",")]

            insertion_part = trace[:K_INSERTIONS + 1]
            refine_part = trace[K_INSERTIONS + 1:]

            # Running minimum over the insertion phase (non-increasing by
            # construction); refine_part is left raw so "unconditional"'s
            # jitter isn't smoothed away.
            running_min, best = [], float("inf")
            for c in insertion_part:
                best = min(best, c)
                running_min.append(best)

            display_trace = running_min + refine_part
            label = f"p{int(row['prompt_idx'])}"
            ax.plot(
                range(len(display_trace)), display_trace,
                marker=".", markersize=3, linewidth=0.8,
                color=cmap(i % 20), label=label,
            )

        ax.axvline(
            K_INSERTIONS, color="gray", linestyle="--", linewidth=1.2, alpha=0.6,
            label=f"insertion→refinement boundary (step {K_INSERTIONS})",
        )
        ax.set_xlabel("Step  (insertion phase → refinement phase)")
        ax.set_ylabel("NLL cost (official MC-reweighted, lower = better)")
        ax.set_title(f"acceptance_mode = {acceptance_mode}  (n={len(sub)})")
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Loss curves -- prefix insertion attack (NLL loss), {sub_all['prompt_idx'].nunique()} prompts\n"
        f"(insertion phase: steps 0-{K_INSERTIONS}, refinement: steps {K_INSERTIONS+1}+; "
        f"'unconditional' is expected to jitter, 'strict' is monotonic by construction)"
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nLoss curve plot saved to: {PLOT_PATH}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\nLoading per-process summaries from {LOG_DIR}...")
    df = load_and_combine()

    print(f"\nSaving combined summary to: {ALL_SUMMARY_PATH}")
    df.to_csv(ALL_SUMMARY_PATH, index=False)

    print()
    print_aggregate_stats(df, out_file=STATS_PATH)
    plot_loss_curves(df)

    print(f"\nDone. {len(df)} runs aggregated.")


if __name__ == "__main__":
    main()
