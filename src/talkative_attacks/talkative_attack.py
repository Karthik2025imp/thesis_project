"""
Talkative attack (EOS/EOT suppression) on LLaDA: for N = 10..100 and all
51 prompts, suppress EOS/EOT at every position within the N-length
canvas, using natural (confidence-based) position ordering, and
quantify the total TV cost of that suppression.

Cost is a direct TV comparison between a row's natural softmax and its
softmax with EOS-like logits masked out (a hard mask, not a minimal-
epsilon nudge like target_force_cost).

Uses LLaDAPlanner instead of the pilot's ConfidencePlanner. At
temperature=0, LLaDAPlanner's confidence reduces exactly to
ConfidencePlanner's (max softmax probability), so position selection is
unchanged; only the naturally-sampled token is discarded and replaced
with the best non-EOS token.

Usage:
    cd ~/project
    python3 -u src/talkative_attack.py 2>&1 | tee src/talkative_attack_results.txt
"""

import sys
import os
import random
import yaml
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")

from llada_wrapper import LLADAWrapper
from llada_planner import LLaDAPlanner
from prob_mass_utils import row_softmax, row_softmax_masked, total_variation
from advbench_utils import filter_aligned_pairs

with open("configs/attack_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

N_VALUES = list(range(10, 101, 10))
N_PROMPTS = 51
SAMPLE_SEED = 42
SAMPLE_FRAC = 0.10
ADVBENCH_PATH = "data/advbench.csv"
CHECKPOINT_EVERY = 1  # checkpoint after every prompt (up to 1000 forward passes each)


def recover_sampled_prompts():
    """Identical filtering + sampling to test.py, so this recovers the
    exact same 51-prompt pool used throughout the project."""
    advbench = pd.read_csv(ADVBENCH_PATH)
    prompt_col = "goal" if "goal" in advbench.columns else "prompt"
    advbench_clean, _ = filter_aligned_pairs(
        advbench, goal_col=prompt_col, target_col="target", threshold=0.2,
    )
    all_prompts = advbench_clean[prompt_col].tolist()
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(all_prompts, k=int(len(all_prompts) * SAMPLE_FRAC))


def eos_suppression_tv(logits_row: torch.Tensor, eos_ids: set) -> float:
    """TV cost of hard-suppressing EOS-like tokens at this row. Returns
    0.0 if the row's natural argmax is already non-EOS (nothing to
    suppress)."""
    natural_id = int(torch.argmax(logits_row).item())
    if natural_id not in eos_ids:
        return 0.0
    p_before = row_softmax(logits_row)
    p_after = row_softmax_masked(logits_row, eos_ids)
    return total_variation(p_before, p_after)


def best_non_eos_token(logits_row: torch.Tensor, eos_ids: set) -> int:
    """Argmax with EOS-like ids excluded."""
    p_masked = row_softmax_masked(logits_row, eos_ids)
    return int(torch.argmax(p_masked).item())


def talkative_simulation(llada, planner, prompt_text, N, eos_ids):
    """At each of N steps: LLaDAPlanner picks the most confident
    remaining masked position (temperature=0, single block spanning
    the whole canvas), we compute+accumulate the suppression cost
    there, then commit the best non-EOS token so later steps see
    genuine updated context.

    Returns (total_cost, final decoded response).
    """
    x, prompt_index = llada.build_input(
        prompt=prompt_text, gen_length=N,
        use_chat_template=base_config.get("use_chat_template", True),
    )
    prompt_len = prompt_index[0].sum().item()
    block_end = prompt_len + N

    total_cost = 0.0

    for step in range(N):
        mask_positions = llada.get_mask_positions(x)
        if len(mask_positions) == 0:
            break

        logits = llada.get_logits(x)

        _, selected_positions = planner.step(
            x=x,
            logits=logits,
            block_start=prompt_len,
            block_end=block_end,
            num_transfer_tokens=1,
            temperature=0.0,
        )
        pos = int(selected_positions[0].item())
        row = logits[pos]

        cost = eos_suppression_tv(row, eos_ids)
        total_cost += cost

        token_id = best_non_eos_token(row, eos_ids)
        x[0, pos] = token_id

    response = llada.decode_response(x, prompt_len)
    return total_cost, response


def main():
    sampled_prompts = recover_sampled_prompts()
    prompts_to_use = sampled_prompts[:N_PROMPTS]

    summary_path = os.path.join(base_config["log_dir"], "eos_suppression_attack/talkative_attack_summary.csv")
    REQUIRED_COLUMNS = {"prompt_idx", "N", "total_suppression_tv", "response"}

    summary_records = []
    completed_prompt_idxs = set()
    if os.path.exists(summary_path):
        prior_df = pd.read_csv(summary_path)
        missing = REQUIRED_COLUMNS - set(prior_df.columns)
        if missing:
            raise RuntimeError(
                f"Existing checkpoint at {summary_path} missing columns "
                f"{sorted(missing)} -- delete and re-run:\n    rm {summary_path}"
            )
        summary_records = prior_df.to_dict("records")
        # A prompt counts as "completed" only if all its N values are present
        counts = prior_df.groupby("prompt_idx")["N"].nunique()
        completed_prompt_idxs = set(counts[counts == len(N_VALUES)].index)
        print(f"Found existing checkpoint: {len(completed_prompt_idxs)} prompts "
              f"fully completed -- resuming.")

    remaining = [
        (idx, p) for idx, p in enumerate(prompts_to_use) if idx not in completed_prompt_idxs
    ]
    print(f"{len(completed_prompt_idxs)} prompts already done, "
          f"{len(remaining)} remaining ({len(remaining) * len(N_VALUES)} runs)\n")

    if remaining:
        print("Loading model...")
        llada = LLADAWrapper(
            model_name=base_config["model_name"],
            device=base_config.get("device", "cuda"),
        )
        planner = LLaDAPlanner(mask_id=llada.mask_token_id)
        eos_ids = set(llada.get_eos_like_token_ids())
        print(f"EOS-like token ids: {eos_ids}\n")

        for run_idx, (prompt_idx, prompt_text) in enumerate(remaining, 1):
            print(f"\n{'='*70}")
            print(f"Prompt {run_idx}/{len(remaining)} remaining -- prompt_idx={prompt_idx}")
            print(f"  {prompt_text[:70]}")
            print(f"{'='*70}")

            for N in N_VALUES:
                total_cost, response = talkative_simulation(llada, planner, prompt_text, N, eos_ids)
                print(f"  N={N:<4} total_suppression_tv={total_cost:.6f}")
                print(f"    Final response: {repr(response)}")

                summary_records.append({
                    "prompt_idx": prompt_idx,
                    "prompt": prompt_text,
                    "N": N,
                    "total_suppression_tv": total_cost,
                    "response": response,
                })

            if run_idx % CHECKPOINT_EVERY == 0 or run_idx == len(remaining):
                os.makedirs(base_config["log_dir"], exist_ok=True)
                pd.DataFrame(summary_records).to_csv(summary_path, index=False)
                print(f"  [Checkpoint saved: {run_idx}/{len(remaining)} prompts -> {summary_path}]")

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE -- generating plots from full summary")
    print(f"{'='*70}\n")

    summary_df = pd.read_csv(summary_path)

    # Plot 1: total cost vs N, one line per prompt + mean
    plt.figure(figsize=(10, 7))
    for prompt_idx in summary_df["prompt_idx"].unique():
        sub = summary_df[summary_df["prompt_idx"] == prompt_idx].sort_values("N")
        plt.plot(sub["N"], sub["total_suppression_tv"], alpha=0.25, color="steelblue")
    mean_by_N = summary_df.groupby("N")["total_suppression_tv"].mean()
    plt.plot(mean_by_N.index, mean_by_N.values, marker="s", linewidth=3,
              color="black", label="mean (all 51 prompts)")
    plt.xlabel("N (generation length; EOS suppressed at all N positions)")
    plt.ylabel("Total suppression TV cost")
    plt.title("Cost of forced non-termination vs. generation length (51 prompts)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot1_path = os.path.join(base_config["log_dir"], "eos_suppression_attack/talkative_cost_vs_N.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot1_path}")

    # Histogram: at which N does each prompt's total cost peak?
    peak_rows = summary_df.loc[summary_df.groupby("prompt_idx")["total_suppression_tv"].idxmax()]
    peak_N_counts = peak_rows["N"].value_counts().reindex(N_VALUES, fill_value=0)

    plt.figure(figsize=(9, 6))
    plt.bar([str(n) for n in N_VALUES], peak_N_counts.values, color="steelblue")
    plt.xlabel("N at which total suppression cost peaks")
    plt.ylabel("Number of prompts (out of 51)")
    plt.title("Distribution of peak-cost N across prompts")
    plt.grid(True, alpha=0.3, axis="y")
    hist_path = os.path.join(base_config["log_dir"], "eos_suppression_attack/talkative_peak_N_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {hist_path}")
    print(f"\nPeak-N distribution:\n{peak_N_counts.to_string()}")


if __name__ == "__main__":
    main()
