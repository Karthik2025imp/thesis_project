"""
Usage:
    cd ~/project
    python3 -u src/dp_optimal_order_sweep.py 2>&1 | tee src/dp_sweep_results.txt
"""

import sys
import os
import yaml
import random
import torch
import pandas as pd
import urllib.request

sys.path.insert(0, "src")

from llada_wrapper import LLADAWrapper
from llada_planner import LLaDAPlanner
from prob_mass_utils import target_force_cost
from advbench_utils import filter_aligned_pairs

with open("configs/attack_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

KAPPA = base_config.get("kappa", 1e-5)

# Same sampling as forced_response.py -- same 51 prompts
SAMPLE_SEED = 42
SAMPLE_FRAC = 0.10

MAX_DP_LENGTH = 10  # truncate every target to at most this many tokens;
                     # cost grows as 2^L (2^10 = 1024 subsets/prompt)

ADVBENCH_PATH = "data/advbench.csv"
ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


def download_advbench():
    if os.path.exists(ADVBENCH_PATH):
        print(f"AdvBench found at {ADVBENCH_PATH} — skipping download.")
        return
    os.makedirs(os.path.dirname(ADVBENCH_PATH), exist_ok=True)
    urllib.request.urlretrieve(ADVBENCH_URL, ADVBENCH_PATH)


download_advbench()


# --------------------------------------------------
# DP machinery
# --------------------------------------------------

def compute_all_subset_costs(llada, x_base, prompt_len, target_ids, kappa=KAPPA, verbose=False):
    """Forcing cost (TV) for every (subset-already-forced, next-position)
    pair, over all 2^L subsets of the truncated target."""
    L = len(target_ids)
    n_subsets = 2 ** L
    cost = {}
    for S in range(n_subsets):
        x_S = x_base.clone()
        for i in range(L):
            pos = prompt_len + i
            x_S[0, pos] = int(target_ids[i].item()) if (S & (1 << i)) else llada.mask_token_id
        with torch.no_grad():
            logits = llada.get_logits(x_S)
        cost[S] = {}
        for i in range(L):
            if S & (1 << i):
                continue
            pos = prompt_len + i
            row = logits[pos]
            natural_id = int(torch.argmax(row).item())
            target_id = int(target_ids[i].item())
            _, tv = target_force_cost(row, natural_id, target_id, kappa)
            cost[S][i] = tv
        if verbose and S > 0 and n_subsets >= 20 and S % max(1, n_subsets // 5) == 0:
            print(f"    {S}/{n_subsets} subsets...")
    return cost


def held_karp_dp(L, cost):
    """Exact minimum-cost forcing order via Held-Karp DP over subsets."""
    full_mask = (1 << L) - 1
    dp = [float("inf")] * (1 << L)
    parent = [None] * (1 << L)
    dp[0] = 0.0
    for S in range(1 << L):
        if dp[S] == float("inf"):
            continue
        for i in range(L):
            if S & (1 << i):
                continue
            new_S = S | (1 << i)
            c = dp[S] + cost[S][i]
            if c < dp[new_S]:
                dp[new_S] = c
                parent[new_S] = (S, i)
    order = []
    S = full_mask
    while S != 0:
        S_prev, i = parent[S]
        order.append(i)
        S = S_prev
    order.reverse()
    return dp[full_mask], order


# --------------------------------------------------
# Natural (planner-driven) and greedy-adaptive cost, on the same
# truncated target -- re-derived here rather than imported, since both
# need to operate on a gen_length == MAX_DP_LENGTH sequence.
# --------------------------------------------------

def greedy_order_cost(llada, x_base, prompt_len, target_ids, kappa=KAPPA):
    """Cost and position order of the greedy-adaptive baseline: at each
    step, force whichever remaining position is currently cheapest."""
    x = x_base.clone()
    total_tv = 0.0
    order = []
    L = len(target_ids)
    for step in range(L):
        mask_positions = llada.get_mask_positions(x)
        if len(mask_positions) == 0:
            break
        logits = llada.get_logits(x)
        best_pos, best_tv = None, float("inf")
        for pos_t in mask_positions:
            pos = int(pos_t.item())
            rel = pos - prompt_len
            target_id = int(target_ids[rel].item())
            row = logits[pos]
            natural_id = int(torch.argmax(row).item())
            _, tv = target_force_cost(row, natural_id, target_id, kappa)
            if tv < best_tv:
                best_tv, best_pos = tv, pos
        total_tv += best_tv
        rel_best = best_pos - prompt_len
        x[0, best_pos] = int(target_ids[rel_best].item())
        order.append(rel_best)
    return total_tv, order


def natural_order_cost(llada, x_base, prompt_len, target_ids, kappa=KAPPA, temperature=0.0):
    """Cost and position order LLaDA's own planner (LLaDAPlanner) would
    use naturally. Position selection determines order only -- the
    token actually written at each position is still the caller's own
    target_id (via target_force_cost), matching ForcedResponseAttack's
    semantics: whatever the planner would naturally have written is
    discarded.

    Processes the whole target region as a single block
    (block_start=prompt_len, block_end=prompt_len+L) and requests
    exactly one position per step, matching the single-token-per-step
    accumulation this cost measurement needs.
    """
    planner = LLaDAPlanner(mask_id=llada.mask_token_id)

    x = x_base.clone()
    total_tv = 0.0
    order = []
    L = len(target_ids)
    block_start = prompt_len
    block_end = prompt_len + L

    for step in range(L):
        mask_positions = llada.get_mask_positions(x)
        if len(mask_positions) == 0:
            break
        logits = llada.get_logits(x)

        _, selected_positions = planner.step(
            x=x,
            logits=logits,
            block_start=block_start,
            block_end=block_end,
            num_transfer_tokens=1,
            temperature=temperature,
        )
        if len(selected_positions) == 0:
            break  # every remaining position's confidence was suppressed
        pos = int(selected_positions[0].item())

        rel = pos - prompt_len
        target_id = int(target_ids[rel].item())
        row = logits[pos]
        natural_id = int(torch.argmax(row).item())
        _, tv = target_force_cost(row, natural_id, target_id, kappa)
        total_tv += tv
        x[0, pos] = target_id  # force our target, discarding what step() sampled
        order.append(rel)

    return total_tv, order


# --------------------------------------------------
# Load, filter, sample -- same 51 prompts as forced_response.py
# --------------------------------------------------

print(f"\nLoading AdvBench from {ADVBENCH_PATH}...")
advbench = pd.read_csv(ADVBENCH_PATH)
goal_col = "goal" if "goal" in advbench.columns else "prompt"
target_col = "target"

advbench_clean, advbench_flagged = filter_aligned_pairs(
    advbench, goal_col=goal_col, target_col=target_col, threshold=0.2,
)
print(f"{len(advbench_clean)} rows remain after filtering "
      f"({len(advbench_flagged)} dropped).")

rng = random.Random(SAMPLE_SEED)
sampled_positions = rng.sample(range(len(advbench_clean)), k=int(len(advbench_clean) * SAMPLE_FRAC))
sampled_pairs = [
    (
        int(advbench_clean.iloc[pos]["orig_index"]),
        advbench_clean.iloc[pos][goal_col],
        advbench_clean.iloc[pos][target_col],
    )
    for pos in sampled_positions
]
print(f"Sampled {len(sampled_pairs)} pairs -- same set as forced_response.py.\n")


# --------------------------------------------------
# Load model, estimate total time, run (checkpointed + resumable)
# --------------------------------------------------

print("Loading model...")
llada = LLADAWrapper(
    model_name=base_config["model_name"],
    device=base_config.get("device", "cuda"),
)

n_prompts = len(sampled_pairs)
est_subsets_per_prompt = 2 ** MAX_DP_LENGTH
est_seconds_per_prompt = est_subsets_per_prompt * 0.15  # from pilot timing
est_total_hours = (n_prompts * est_seconds_per_prompt) / 3600
print(
    f"\nMAX_DP_LENGTH={MAX_DP_LENGTH} -> {est_subsets_per_prompt} subsets/prompt "
    f"-> ~{est_seconds_per_prompt:.0f}s/prompt for the DP phase\n"
    f"Estimated total time for {n_prompts} prompts: ~{est_total_hours:.1f} hours "
    f"(DP phase only; natural/greedy phases add a small amount more)\n"
)

summary_path = os.path.join(base_config["log_dir"], "dp_new_sweep_summary.csv")
REQUIRED_COLUMNS = {"prompt_idx", "L_truncated", "dp_min_tv", "natural_tv", "greedy_tv"}

summary_records = []
completed_idxs = set()
if os.path.exists(summary_path):
    prior_df = pd.read_csv(summary_path)
    missing = REQUIRED_COLUMNS - set(prior_df.columns)
    if missing:
        raise RuntimeError(
            f"Existing checkpoint at {summary_path} missing columns "
            f"{sorted(missing)} -- delete and re-run:\n    rm {summary_path}"
        )
    summary_records = prior_df.to_dict("records")
    completed_idxs = set(prior_df["prompt_idx"])
    print(f"Found existing checkpoint with {len(prior_df)} completed runs -- resuming.")

remaining = [p for p in sampled_pairs if p[0] not in completed_idxs]
print(f"{len(completed_idxs)} already done, {len(remaining)} remaining\n")

failed_runs = []
CHECKPOINT_EVERY = 2  # small -- each run is itself expensive

for run_idx, (prompt_idx, prompt, target_text) in enumerate(remaining, 1):
    try:
        target_ids_full = llada.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt",
        )["input_ids"][0].to(llada.device)
        L_full = target_ids_full.shape[0]
        L = min(L_full, MAX_DP_LENGTH)
        target_ids = target_ids_full[:L]

        x_base, prompt_index = llada.build_input(
            prompt=prompt, gen_length=L,
            use_chat_template=base_config.get("use_chat_template", True),
        )
        prompt_len = prompt_index[0].sum().item()

        cost = compute_all_subset_costs(llada, x_base, prompt_len, target_ids)
        dp_tv, dp_order = held_karp_dp(L, cost)
        natural_tv, natural_order = natural_order_cost(llada, x_base, prompt_len, target_ids)
        greedy_tv, greedy_order = greedy_order_cost(llada, x_base, prompt_len, target_ids)

        ltr_order = list(range(L))
        S = 0
        ltr_tv = 0.0
        for i in ltr_order:
            ltr_tv += cost[S][i]
            S |= (1 << i)

        result = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "L_truncated": L,
            "greedy_tv": greedy_tv,
            "greedy_order": " ".join(str(i) for i in greedy_order),
            "natural_tv": natural_tv,
            "natural_order": " ".join(str(i) for i in natural_order),
            "ltr_tv": ltr_tv,
            "ltr_order": " ".join(str(i) for i in ltr_order),
            "dp_min_tv": dp_tv,
            "dp_order": " ".join(str(i) for i in dp_order),
        }
    except Exception as e:
        print(f"prompt_idx={prompt_idx} FAILED: {type(e).__name__}: {e}")
        failed_runs.append({"prompt_idx": prompt_idx, "error": f"{type(e).__name__}: {e}"})
        torch.cuda.empty_cache()
        continue

    summary_records.append(result)
    print(f"\nprompt_idx={prompt_idx}")
    print(f"greedy     tv={greedy_tv:.6f}  sequence={greedy_order}")
    print(f"natural    tv={natural_tv:.6f}  sequence={natural_order}")
    print(f"left-right tv={ltr_tv:.6f}  sequence={ltr_order}")
    print(f"dp-optimal tv={dp_tv:.6f}  sequence={dp_order}")

    if run_idx % CHECKPOINT_EVERY == 0 or run_idx == len(remaining):
        os.makedirs(base_config["log_dir"], exist_ok=True)
        pd.DataFrame(summary_records).to_csv(summary_path, index=False)
        if failed_runs:
            pd.DataFrame(failed_runs).to_csv(
                os.path.join(base_config["log_dir"], "dp_new_sweep_failed.csv"), index=False,
            )
