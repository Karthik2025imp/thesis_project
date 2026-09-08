"""
Talkative attack (EOS/EOT suppression) on Llama-3-8B-Instruct -- the AR
comparison point for the diffusion-model version (talkative_attack.py):
does LLaDA show more/less/equally robust behavior against EOS
suppression compared to a comparable-scale autoregressive model?

Llama-3-8B-Instruct is used both for comparable scale and because
LLaDA's own chat template is inherited from the Llama-3 template
format, giving a cleaner comparison.

eos_suppression_tv and best_non_eos_token are reused unchanged from
talkative_attack.py (architecture-agnostic, operate on a single logits
row). The generation loop is new: AR has no masking/planner concept --
strictly left-to-right, one next position per step.

Same 51-prompt sample and N values (10..100) as talkative_attack.py,
for a directly comparable sweep.

Usage:
    cd ~/project
    python3 -u src/talkative_attack_ar.py 2>&1 | tee src/talkative_attack_ar_results.txt
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")

from prob_mass_utils import row_softmax, row_softmax_masked, total_variation
from advbench_utils import filter_aligned_pairs

with open("configs/attack_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

N_VALUES = list(range(10, 101, 10))
N_PROMPTS = 51
SAMPLE_SEED = 42
SAMPLE_FRAC = 0.10
ADVBENCH_PATH = "data/advbench.csv"

AR_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def recover_sampled_prompts():
    """Identical sampling to talkative_attack.py -- same seed/frac and
    filtering, so both sweeps run on the exact same 51 prompts."""
    advbench = pd.read_csv(ADVBENCH_PATH)
    prompt_col = "goal" if "goal" in advbench.columns else "prompt"
    advbench_clean, _ = filter_aligned_pairs(
        advbench, goal_col=prompt_col, target_col="target", threshold=0.2,
    )
    all_prompts = advbench_clean[prompt_col].tolist()
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(all_prompts, k=int(len(all_prompts) * SAMPLE_FRAC))


def eos_suppression_tv(logits_row: torch.Tensor, eos_ids: set) -> float:
    """TV cost of hard-suppressing EOS-like tokens at this row."""
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


def get_ar_eos_ids(tokenizer) -> set:
    """Llama-3's EOS-like tokens: <|end_of_text|>, <|eot_id|>, and
    tokenizer.eos_token_id. Only includes ids that genuinely exist in
    the vocabulary."""
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for special in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.add(tid)
    return eos_ids


def talkative_simulation_ar(model, tokenizer, eos_ids, prompt_text, N):
    """Strictly left-to-right: forward pass, logits at the last
    position, suppress EOS if it's the top choice, append the best
    non-EOS token, repeat for N steps.

    Returns (total_cost, final decoded response).
    """
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    generated_ids = input_ids.clone()
    total_cost = 0.0

    for step in range(N):
        with torch.no_grad():
            outputs = model(generated_ids)
        logits = outputs.logits[0, -1, :]  # next-token logits, last position only

        cost = eos_suppression_tv(logits, eos_ids)
        total_cost += cost

        token_id = best_non_eos_token(logits, eos_ids)
        next_token = torch.tensor([[token_id]], device=model.device, dtype=generated_ids.dtype)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    response_ids = generated_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return total_cost, response


def main():
    sampled_prompts = recover_sampled_prompts()
    prompts_to_use = sampled_prompts[:N_PROMPTS]

    summary_path = os.path.join(base_config["log_dir"], "eos_suppression_attack/talkative_attack_ar_summary.csv")
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
        print(f"Loading {AR_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(AR_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            AR_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()

        # Fail fast if the model got split across multiple devices
        # (recurring issue on this shared GPU).
        devices_found = set(str(p.device) for p in model.parameters())
        if len(devices_found) > 1:
            raise RuntimeError(
                f"Model parameters are split across multiple devices: {devices_found}. "
                f"Check `nvidia-smi` for current usage before retrying."
            )
        print(f"Model fully loaded on a single device: {devices_found.pop()}")

        eos_ids = get_ar_eos_ids(tokenizer)
        print(f"EOS-like token ids: {eos_ids}\n")

        for run_idx, (prompt_idx, prompt_text) in enumerate(remaining, 1):
            print(f"\n{'='*70}")
            print(f"Prompt {run_idx}/{len(remaining)} remaining -- prompt_idx={prompt_idx}")
            print(f"  {prompt_text[:70]}")
            print(f"{'='*70}")

            for N in N_VALUES:
                total_cost, response = talkative_simulation_ar(
                    model, tokenizer, eos_ids, prompt_text, N,
                )
                print(f"  N={N:<4} total_suppression_tv={total_cost:.6f}")
                print(f"    Final response: {repr(response)}")

                summary_records.append({
                    "prompt_idx": prompt_idx,
                    "prompt": prompt_text,
                    "N": N,
                    "total_suppression_tv": total_cost,
                    "response": response,
                })

            os.makedirs(base_config["log_dir"], exist_ok=True)
            pd.DataFrame(summary_records).to_csv(summary_path, index=False)
            print(f"  [Checkpoint saved: {run_idx}/{len(remaining)} prompts]")

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE -- generating comparison plot")
    print(f"{'='*70}\n")

    ar_df = pd.read_csv(summary_path)
    ar_mean_by_N = ar_df.groupby("N")["total_suppression_tv"].mean()

    # Comparison plot: LLaDA vs Llama-3, overlaid
    llada_path = os.path.join(base_config["log_dir"], "eos_suppression_attack/talkative_attack_summary.csv")
    if os.path.exists(llada_path):
        llada_df = pd.read_csv(llada_path)
        llada_mean_by_N = llada_df.groupby("N")["total_suppression_tv"].mean()

        plt.figure(figsize=(10, 7))
        plt.plot(llada_mean_by_N.index, llada_mean_by_N.values, marker="o",
                 linewidth=2, label="LLaDA-8B-Instruct (diffusion)")
        plt.plot(ar_mean_by_N.index, ar_mean_by_N.values, marker="s",
                 linewidth=2, label="Llama-3-8B-Instruct (AR)")
        plt.xlabel("N (generation length; EOS suppressed at all N positions)")
        plt.ylabel("Mean total suppression TV cost")
        plt.title("Cost of forced non-termination: diffusion vs. autoregressive")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(base_config["log_dir"], "eos_suppression_attack/talkative_attack_diffusion_vs_ar.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot: {plot_path}")

        print(f"\n{'='*70}")
        print("MEAN COST BY N -- side by side")
        print(f"{'='*70}")
        print(f"{'N':<6}{'LLaDA':<12}{'Llama-3':<12}")
        for N in N_VALUES:
            l_val = llada_mean_by_N.get(N, float("nan"))
            a_val = ar_mean_by_N.get(N, float("nan"))
            print(f"{N:<6}{l_val:<12.4f}{a_val:<12.4f}")
    else:
        print(f"LLaDA comparison data not found at {llada_path} -- "
              f"skipping overlay plot, AR-only results saved.")


if __name__ == "__main__":
    main()
