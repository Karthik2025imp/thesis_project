"""CSV logger for planner robustness analysis.

Logs per-step epsilon, margin, positions, planner flip status, masked-
token count, and probability-mass-shift (total variation) fields.
"""

import os
import pandas as pd


class AttackLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records = []

    def log_step(
        self,
        step: int,
        attacked: bool,
        planner_before: dict,
        planner_after: dict,
        metadata: dict,
        num_masked_remaining: int = 0,
    ):
        record = {
            "step":                 step,
            "attacked":             attacked,
            "num_masked_remaining": num_masked_remaining,
            "epsilon":              metadata.get("epsilon", 0.0),
            "margin":               metadata.get("margin", 0.0),
            "original_position":    planner_before["selected_position"],
            "new_position":         planner_after["selected_position"],
            "planner_changed": (
                planner_before["selected_position"] != planner_after["selected_position"]
            ),
            "skipped":              metadata.get("skipped", False),
            # total_variation = tv_selected + tv_target (see prob_mass_utils.py)
            "tv_selected":          metadata.get("tv_selected", 0.0),
            "tv_target":            metadata.get("tv_target", 0.0),
            "total_variation":      metadata.get("total_variation", 0.0),
            # "Content-only" TV, with EOS/EOT-like tokens masked out first
            "tv_selected_excl_eos":     metadata.get("tv_selected_excl_eos", 0.0),
            "tv_target_excl_eos":       metadata.get("tv_target_excl_eos", 0.0),
            "total_variation_excl_eos": metadata.get("total_variation_excl_eos", 0.0),
        }
        self.records.append(record)

    def save(self, filename: str = "attack_log.csv"):
        df = pd.DataFrame(self.records)
        output_path = os.path.join(self.log_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved logs to: {output_path}")

    def summary(self):
        """Print attack success rate summary."""
        if not self.records:
            print("No records logged.")
            return

        attacked = [r for r in self.records if r["attacked"] and not r["skipped"]]
        flipped = [r for r in attacked if r["planner_changed"]]

        print("\n========== Attack Summary ==========")
        print(f"Total steps:     {len(self.records)}")
        print(f"Steps attacked:  {len(attacked)}")
        print(f"Planner flipped: {len(flipped)}")
        if attacked:
            rate = len(flipped) / len(attacked) * 100
            avg_eps = sum(r["epsilon"] for r in attacked) / len(attacked)
            avg_margin = sum(r["margin"] for r in attacked) / len(attacked)
            avg_tv = sum(r["total_variation"] for r in attacked) / len(attacked)
            avg_tv_excl_eos = sum(r["total_variation_excl_eos"] for r in attacked) / len(attacked)
            mean_tv = avg_tv / 2.0  # per-row-normalized (2 rows touched per step)
            print(f"Flip rate:       {rate:.1f}%")
            print(f"Avg epsilon:     {avg_eps:.6f}  (raw logit units)")
            print(f"Avg margin:      {avg_margin:.6f}")
            print(f"Avg total TV:    {avg_tv:.6f}  (summed over both rows; ceiling 1.0)")
            print(f"Mean TV (per row): {mean_tv:.6f}")
            print(f"Avg content-only TV: {avg_tv_excl_eos:.6f}  (EOS/EOT masked out first)")
        print("=====================================\n")


class ForcedResponseLogger:
    """Per-position logger for forced-response (complete forcing)
    experiments. Each record is tagged with a "phase":
        "FORCED"          -- forced via ForcedResponseAttack.force_positions()
        "FREE GENERATION" -- appended after forcing completes, decoded
                              with zero intervention
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records = []

    def log_positions(self, step: int, position_records: list):
        """position_records: list of dicts, one per position handled
        this step, each already tagged with a "phase" key."""
        for r in position_records:
            record = dict(r)
            record["step"] = step
            self.records.append(record)

    def save(self, filename: str = "forced_response_log.csv"):
        df = pd.DataFrame(self.records)
        output_path = os.path.join(self.log_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved logs to: {output_path}")
        return output_path

    def summary(self):
        """Compute and print per-run budget summary. Epsilon-based
        stats are computed only over phase == "FORCED" rows (or all
        rows if no "phase" column is present, for backward compat)."""
        if not self.records:
            print("No records logged.")
            return {}

        df = pd.DataFrame(self.records)

        if "phase" in df.columns:
            forced_df = df[df["phase"] == "FORCED"]
            free_gen_tokens = int((df["phase"] != "FORCED").sum())
        else:
            forced_df = df
            free_gen_tokens = 0

        if len(forced_df) == 0:
            print("No FORCED rows logged (free-generation-only log?).")
            return {"free_generation_tokens": free_gen_tokens}

        total_budget = float(forced_df["epsilon"].sum())
        l2_norm = float(((forced_df["epsilon"] * (2 ** 0.5)) ** 2).sum() ** 0.5)
        max_epsilon = float(forced_df["epsilon"].max())
        bottleneck = forced_df.loc[forced_df["epsilon"].idxmax()]
        n_free = int(forced_df["already_matched"].sum())
        n_total = len(forced_df)

        has_tv = "total_variation" in forced_df.columns
        if has_tv:
            total_budget_tv = float(forced_df["total_variation"].sum())
            max_tv = float(forced_df["total_variation"].max())
            bottleneck_tv = forced_df.loc[forced_df["total_variation"].idxmax()]

        has_tv_excl_eos = "total_variation_excl_eos" in forced_df.columns
        if has_tv_excl_eos:
            total_budget_tv_excl_eos = float(forced_df["total_variation_excl_eos"].sum())
            max_tv_excl_eos = float(forced_df["total_variation_excl_eos"].max())
            bottleneck_tv_excl_eos = forced_df.loc[forced_df["total_variation_excl_eos"].idxmax()]

        summary = {
            "total_positions": n_total,
            "already_matched": n_free,
            "forced": n_total - n_free,
            "total_budget_eps": total_budget,
            "l2_norm": l2_norm,
            "max_epsilon": max_epsilon,
            "bottleneck_rel_position": int(bottleneck["rel_position"]),
            "bottleneck_rel_fraction": (
                int(bottleneck["rel_position"]) / (n_total - 1) if n_total > 1 else 0.0
            ),
            "free_generation_tokens": free_gen_tokens,
        }

        if has_tv:
            summary["total_budget_tv"] = total_budget_tv
            summary["max_tv"] = max_tv
            summary["bottleneck_tv_rel_position"] = int(bottleneck_tv["rel_position"])
            summary["bottleneck_tv_rel_fraction"] = (
                int(bottleneck_tv["rel_position"]) / (n_total - 1) if n_total > 1 else 0.0
            )
            summary["bottleneck_metrics_agree"] = (
                int(bottleneck["rel_position"]) == int(bottleneck_tv["rel_position"])
            )
            if "target_token_str" in forced_df.columns:
                summary["bottleneck_tv_target_token_str"] = bottleneck_tv["target_token_str"]
                summary["bottleneck_tv_natural_token_str"] = bottleneck_tv["natural_token_str"]

        if has_tv_excl_eos:
            summary["total_budget_tv_excl_eos"] = total_budget_tv_excl_eos
            summary["max_tv_excl_eos"] = max_tv_excl_eos
            summary["bottleneck_tv_excl_eos_rel_position"] = int(bottleneck_tv_excl_eos["rel_position"])
            summary["bottleneck_tv_excl_eos_rel_fraction"] = (
                int(bottleneck_tv_excl_eos["rel_position"]) / (n_total - 1) if n_total > 1 else 0.0
            )
            if "target_token_str" in forced_df.columns:
                summary["bottleneck_tv_excl_eos_target_token_str"] = bottleneck_tv_excl_eos["target_token_str"]
                summary["bottleneck_tv_excl_eos_natural_token_str"] = bottleneck_tv_excl_eos["natural_token_str"]

        if "relative_severity" in forced_df.columns:
            # .astype(bool) avoids bitwise (not logical) ~ on an
            # object-dtype column caused by mixed FORCED/FREE rows.
            already_matched_bool = forced_df["already_matched"].astype(bool)
            actually_forced = forced_df[~already_matched_bool]
            summary["mean_relative_severity"] = (
                float(actually_forced["relative_severity"].mean()) if len(actually_forced) > 0 else 0.0
            )

        if "target_token_str" in forced_df.columns:
            summary["bottleneck_target_token_str"] = bottleneck["target_token_str"]
            summary["bottleneck_natural_token_str"] = bottleneck["natural_token_str"]
        if "worst_case_token_str" in forced_df.columns:
            summary["bottleneck_worst_case_token_str"] = bottleneck["worst_case_token_str"]

        print("\n===== Forced Response Summary =====")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("====================================\n")

        return summary
