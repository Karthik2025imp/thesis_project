# Thesis: Adversarial Attacks on Diffusion Models

Code accompanying the thesis investigating adversarial attacks and robustness
properties of [LLaDA](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), a
masked discrete diffusion language model.

## Repository structure

```
configs/
    attack_config.yaml       Shared config: model_name, device, kappa,
                              log_dir, use_chat_template

data/                        AdvBench dataset (gitignored — see Data below)

logs/                        All run outputs: checkpoints, summary CSVs,
                              plots (gitignored)
    planner_attacks/
    tv_costs/
    talkative_attacks/
    gcg_complete/
        base-natural/
        instruct-natural/
        base-nll-scaled/
        instruct-nll-scaled/

src/
    llada_wrapper.py         LLaDA model wrapper (load, forward pass, decode)
    llada_planner.py         Official LLaDA position-selection algorithm,
                              ported from ML-GSAI/LLaDA/generate.py
    scheduler.py              Early-attack step scheduler
    advbench_utils.py        AdvBench loading / goal-target alignment filter
    prob_mass_utils.py       Total-variation cost utilities (shared by the
                              -natural experiments)

    planner_attacks/         Experiment 1
        test.py               Main sweep: no_attack / minimal_epsilon /
                               max_disruption vs AdvBench
        planner.py            ConfidencePlanner (this experiment only)
        attack.py             DynamicMinimalEpsilonAttack, ForcedResponseAttack
        logger.py              AttackLogger, ForcedResponseLogger
        external_ppl.py       External-LM (Qwen2.5-0.5B) perplexity scoring

    tv_costs/                 Experiment 2
        dp_optimal_order_sweep.py   Held-Karp DP validation of optimal
                                     forcing order vs natural/greedy/left-right

    talkative_attacks/        Experiment 3
        talkative_attack.py         EOS/EOT suppression cost sweep (LLaDA)
        talkative_attack_ar.py      Same sweep on Llama-3-8B-Instruct (AR
                                     comparison point)

    gcg_complete/              Experiment 4 — prefix-insertion GCG-style attack
        new/
            base-natural/       LLaDA-8B-Base,     natural-order-cost loss
            instruct-natural/   LLaDA-8B-Instruct, natural-order-cost loss
            base-nll-scaled/    LLaDA-8B-Base,     official MC-reweighted NLL loss
            instruct-nll-scaled/ LLaDA-8B-Instruct, official MC-reweighted NLL loss

            Each variant folder contains:
                prefix_run_worker.py   Shared per-variant attack logic
                prefix_run_0.py .. prefix_run_4.py   5 parallel process entry
                                                       points (each slices a
                                                       range of AdvBench prompts)
                aggregate_results.py   Combines the 5 processes' summaries,
                                       prints stats, plots loss curves
```

## Experiments

1. **Planner attack** (`src/planner_attacks/`) — adversarial logit
   perturbation targeting the planner's position-selection confidence, in two
   modes (`minimal_epsilon`, `max_disruption`), measured against AdvBench.
2. **TV cost / optimal ordering** (`src/tv_costs/`) — exact (Held-Karp DP)
   vs heuristic (natural, greedy, left-to-right) minimum-cost token-forcing
   order, on truncated targets.
3. **Talkative attack** (`src/talkative_attacks/`) — cost of suppressing
   EOS/EOT to force non-termination, on LLaDA and on Llama-3 (autoregressive)
   for comparison.
4. **GCG-style prefix insertion** (`src/gcg_complete/`) — multi-position
   joint-search token insertion attack run across 4 combinations of model
   (Base/Instruct) and loss (natural-order cost / official MC-reweighted
   NLL).

## Setup

```bash
pip install -r requirements.txt
```

Edit `configs/attack_config.yaml` for `model_name`, `device`, `kappa`, and
`log_dir` (experiments 1–3; experiment 4 pins its own model/log_dir per
variant folder instead — see each `prefix_run_worker.py`).

## Data

Most entry scripts auto-download AdvBench
(`llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv`) to
`data/advbench.csv` on first run.
## Notes on running this code

1. This repo documents the code **as used** on the compute environment
(Isambard) it was developed on, not a guaranteed-portable package.

2. If you're setting this up to run yourself, expect to adjust paths — this
README documents intent and structure, not a plug-and-play pipeline.

3. `meta-llama/Meta-Llama-3-8B-Instruct` (used in `talkative_attack_ar.py`)
  is a gated HuggingFace model; a valid HF token/login is required.
