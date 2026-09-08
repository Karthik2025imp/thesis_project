"""Dynamic minimal-epsilon planner attack for LLaDA masked diffusion.

Two attack modes:
    "minimal_epsilon": target = runner-up masked position; minimises
        the perturbation needed to flip the planner's selection.
    "max_disruption":  target = lowest-confidence masked position;
        perturbation must beat every other masked position, not just
        the one it displaces.

Operates entirely in probability space (softmax of the logits), via
exact proportional mass redistribution rather than a linear logit-space
nudge, then converts back to a logits-shaped tensor via log() so
downstream code is unaffected (softmax(log(p)) recovers p exactly).

Decreasing a row's top probability is capped at the point where the
row's own internal runner-up would overtake it; the compensating
increase on the target row is then computed against the selected row's
true resulting confidence, guaranteeing an exact flip. See
`compute_deltas` for the derivation.
"""

import torch
from prob_mass_utils import (
    row_softmax, total_variation, row_tv_masked,
    worst_case_row_tv, relative_severity,
)


def _row_top2(probs_row: torch.Tensor):
    """Return (top_value, top_index, second_value) for a 1D probability row."""
    top_value, top_index = torch.max(probs_row, dim=-1)
    masked = probs_row.clone()
    masked[top_index] = float("-inf")
    second_value = masked.max().item()
    return top_value.item(), top_index.item(), second_value


def _redistribute_row(probs_row: torch.Tensor, top_index: int, new_top_value: float) -> torch.Tensor:
    """Set probs_row[top_index] = new_top_value and rescale every other
    entry proportionally so the row still sums to 1 (exact, no
    linearisation).
    """
    old_top_value = probs_row[top_index].item()
    denom = 1.0 - old_top_value

    if denom <= 0.0:
        # Fully-peaked row: nothing to redistribute from.
        new_row = probs_row.clone()
        new_row[top_index] = new_top_value
        return new_row

    scale = (1.0 - new_top_value) / denom
    new_row = probs_row * scale
    new_row[top_index] = new_top_value
    return new_row


class DynamicMinimalEpsilonAttack:
    def __init__(self, kappa: float = 1e-5, attack_mode: str = "minimal_epsilon"):
        assert attack_mode in ("minimal_epsilon", "max_disruption"), (
            f"attack_mode must be 'minimal_epsilon' or 'max_disruption', got '{attack_mode}'"
        )
        self.kappa = kappa
        self.attack_mode = attack_mode

    def choose_target_position(self, row_confidences: torch.Tensor, local_selected: int) -> int:
        """Select the target masked position for this attack_mode.

        "minimal_epsilon": runner-up (second-highest confidence across positions).
        "max_disruption":  lowest-confidence position.
        """
        confidences = row_confidences.clone()
        sentinel = float("inf") if self.attack_mode == "max_disruption" else float("-inf")
        confidences[local_selected] = sentinel

        if self.attack_mode == "minimal_epsilon":
            return torch.argmax(confidences).item()
        else:
            return torch.argmin(confidences).item()

    def compute_deltas(
        self,
        masked_probs: torch.Tensor,
        row_confidences: torch.Tensor,
        local_selected: int,
        target_local: int,
    ) -> dict:
        """Compute delta1 (decrease on the selected row) and delta2
        (compensating increase on the target row), in probability space.

        delta1 is capped at epsilon_max, the threshold below which the
        selected row's own internal runner-up would overtake it. delta2
        is then computed against the selected row's true resulting
        confidence, so the flip is exact rather than approximate.
        """
        selected_conf = row_confidences[local_selected].item()
        target_conf = row_confidences[target_local].item()

        _, _, p2 = _row_top2(masked_probs[local_selected])

        denom = 1.0 - selected_conf + p2
        eps_max = (1.0 - selected_conf) * (selected_conf - p2) / denom if denom > 0 else 0.0
        eps_max = max(eps_max, 0.0)

        m3 = None
        if self.attack_mode == "max_disruption":
            confidences = row_confidences.clone()
            confidences[local_selected] = float("-inf")
            confidences[target_local] = float("-inf")
            if confidences.numel() > 0 and not torch.all(confidences == float("-inf")):
                m3 = torch.max(confidences).item()
            else:
                m3 = 0.0  # only 2 masked positions -- no third exists
            naive_delta1 = max(m3 - target_conf, 0.0)
        else:
            naive_delta1 = (selected_conf - target_conf) / 2.0

        capped = naive_delta1 > eps_max
        delta1 = (eps_max - self.kappa) if capped else naive_delta1
        delta1 = max(delta1, 0.0)

        actual_selected_conf_after = selected_conf - delta1

        bar = actual_selected_conf_after
        if self.attack_mode == "max_disruption":
            bar = max(bar, m3)

        delta2 = (bar - target_conf) + self.kappa

        return {
            "delta1": delta1,
            "delta2": delta2,
            "epsilon_max": eps_max,
            "capped": capped,
            "m3": m3,
            "actual_selected_conf_after": actual_selected_conf_after,
        }

    def perturb_logits(self, logits: torch.Tensor, planner_output: dict, exclude_token_ids=None):
        """Apply the perturbation to two rows of the probability matrix:
        the selected position's row (top entry decreased) and the target
        position's row (top entry increased, compensating). Returns a
        logits-shaped tensor (original logits elsewhere) plus metadata.
        """
        selected_position = planner_output["selected_position"]
        local_selected = planner_output["local_selected"]
        mask_positions = planner_output["mask_positions"]

        if len(mask_positions) < 2:
            metadata = {
                "epsilon": 0.0,
                "delta1": 0.0,
                "delta2": 0.0,
                "epsilon_max": 0.0,
                "capped": False,
                "original_position": selected_position,
                "target_position": selected_position,
                "margin": 0.0,
                "skipped": True,
                "attack_mode": self.attack_mode,
                "tv_selected": 0.0,
                "tv_target": 0.0,
                "total_variation": 0.0,
            }
            if exclude_token_ids is not None:
                metadata["tv_selected_excl_eos"] = 0.0
                metadata["tv_target_excl_eos"] = 0.0
                metadata["total_variation_excl_eos"] = 0.0
            return logits.clone().float(), metadata

        # The one authoritative softmax call -- everything from here on
        # operates on probabilities, never on raw logits.
        logits_f32 = logits.float()
        masked_probs = torch.softmax(logits_f32[mask_positions], dim=-1)  # [num_masked, vocab]
        row_confidences, row_indices = torch.max(masked_probs, dim=-1)

        selected_conf = row_confidences[local_selected].item()
        selected_vocab_index = row_indices[local_selected].item()

        target_local = self.choose_target_position(row_confidences, local_selected)
        target_position = mask_positions[target_local].item()
        target_conf = row_confidences[target_local].item()
        target_vocab_index = row_indices[target_local].item()

        margin = selected_conf - target_conf

        deltas = self.compute_deltas(
            masked_probs=masked_probs,
            row_confidences=row_confidences,
            local_selected=local_selected,
            target_local=target_local,
        )
        delta1, delta2 = deltas["delta1"], deltas["delta2"]

        selected_row_before = masked_probs[local_selected].clone()
        target_row_before = masked_probs[target_local].clone()

        selected_row_after = _redistribute_row(selected_row_before, selected_vocab_index, selected_conf - delta1)
        target_row_after = _redistribute_row(target_row_before, target_vocab_index, target_conf + delta2)

        actual_selected_conf = selected_row_after.max().item()
        actual_target_conf = target_row_after.max().item()

        assert abs(actual_selected_conf - deltas["actual_selected_conf_after"]) < 1e-4, (
            f"Selected row's actual confidence ({actual_selected_conf:.6f}) does not "
            f"match the value compute_deltas guaranteed ({deltas['actual_selected_conf_after']:.6f})."
        )
        assert actual_target_conf > actual_selected_conf, (
            f"Flip not achieved.\n"
            f"  attack_mode: {self.attack_mode}\n"
            f"  selected_position: {selected_position}  new_conf={actual_selected_conf:.8f}\n"
            f"  target_position:   {target_position}  new_conf={actual_target_conf:.8f}\n"
            f"  delta1={delta1:.8f}  delta2={delta2:.8f}"
        )

        if self.attack_mode == "max_disruption":
            # Verify target is the true argmax over all masked positions.
            all_confs = row_confidences.clone()
            all_confs[local_selected] = actual_selected_conf
            all_confs[target_local] = actual_target_conf
            true_winner = torch.argmax(all_confs).item()
            assert true_winner == target_local, (
                f"max_disruption: target is not the true argmax.\n"
                f"  target_local={target_local}  true_winner={true_winner}\n"
                f"  target_conf={actual_target_conf:.8f}\n"
                f"  winner_conf={all_confs[true_winner].item():.8f}"
            )

        attacked_logits = logits_f32.clone()
        attacked_logits[selected_position] = torch.log(selected_row_after)
        attacked_logits[target_position] = torch.log(target_row_after)

        tv_selected = total_variation(selected_row_before, selected_row_after)
        tv_target = total_variation(target_row_before, target_row_after)

        metadata = {
            "epsilon": delta2,  # backward-compat single-number field
            "delta1": delta1,
            "delta2": delta2,
            "epsilon_max": deltas["epsilon_max"],
            "capped": deltas["capped"],
            "m3": deltas["m3"],
            "original_position": selected_position,
            "target_position": target_position,
            "margin": margin,
            "skipped": False,
            "attack_mode": self.attack_mode,
            "tv_selected": tv_selected,
            "tv_target": tv_target,
            "total_variation": tv_selected + tv_target,
        }

        if exclude_token_ids is not None:
            tv_selected_excl_eos = row_tv_masked(
                logits_f32[selected_position], attacked_logits[selected_position], exclude_token_ids,
            )
            tv_target_excl_eos = row_tv_masked(
                logits_f32[target_position], attacked_logits[target_position], exclude_token_ids,
            )
            metadata["tv_selected_excl_eos"] = tv_selected_excl_eos
            metadata["tv_target_excl_eos"] = tv_target_excl_eos
            metadata["total_variation_excl_eos"] = tv_selected_excl_eos + tv_target_excl_eos

        return attacked_logits, metadata


class ForcedResponseAttack:
    """Forces LLaDA's diffusion generation to reproduce an exact target
    response, token-for-token, at whichever position the planner
    naturally selects each step.

    Unlike DynamicMinimalEpsilonAttack, position selection is untouched
    (natural, unattacked logits); only the token forced at each selected
    position is fixed in advance by the target sequence. At each forced
    position, exactly 2 logit coordinates are modified: the natural
    argmax (decreased) and the target token (increased), by
        epsilon = max((natural_conf - target_conf) / 2, M3 - target_conf) + kappa
    where M3 is the max confidence over every other (untouched) vocab
    coordinate in the row -- this guarantees the target becomes the new
    row-argmax against all competitors, not just the natural one.

    If the natural argmax already equals the target token, epsilon = 0.0
    and no perturbation is applied (still logged, not skipped).

    The actual token write should go through
    LLADAWrapper.force_write_positions rather than unmask_positions,
    since no sampling is needed here -- the 2-logit swap is still
    performed and asserted so the logit-space cost is measured.
    """

    def __init__(self, kappa: float = 1e-5):
        self.kappa = kappa

    def force_positions(
        self,
        logits: torch.Tensor,
        positions_to_force: torch.Tensor,
        target_token_ids: torch.Tensor,
        prompt_len: int,
        exclude_token_ids=None,
    ):
        """Force target tokens at `positions_to_force`.

        Args:
            logits:              [seq_len, vocab_size]
            positions_to_force:  absolute positions to force this step
            target_token_ids:    full target sequence, indexed by
                                  relative position (position - prompt_len)
            prompt_len:          number of prompt tokens
            exclude_token_ids:   optional vocab ids to exclude when
                                  computing content-only TV distance

        Returns:
            attacked_logits: [seq_len, vocab_size] float32
            records: list of per-position dicts
        """
        attacked_logits = logits.clone().float()
        records = []

        for pos_tensor in positions_to_force:
            pos = int(pos_tensor.item())
            rel_pos = pos - prompt_len

            assert 0 <= rel_pos < len(target_token_ids), (
                f"Position {pos} (rel {rel_pos}) falls outside the target "
                f"sequence (len={len(target_token_ids)}). prompt_len={prompt_len}."
            )
            target_id = int(target_token_ids[rel_pos].item())

            row = attacked_logits[pos]
            natural_id = torch.argmax(row).item()
            natural_conf = row[natural_id].item()
            target_conf = row[target_id].item()
            row_before = row.clone()  # snapshot before mutation, for TV

            if natural_id == target_id:
                epsilon = 0.0
                already_matched = True
                tv = 0.0
            else:
                # M3: max over every other coordinate in this row (i.e.
                # excluding natural_id and target_id) -- these stay
                # untouched, so must already be beaten by target_conf + epsilon.
                row_masked = row.clone()
                row_masked[natural_id] = float("-inf")
                row_masked[target_id] = float("-inf")
                m3 = row_masked.max().item()  # -inf if vocab_size <= 2

                eps_vs_natural = (natural_conf - target_conf) / 2.0
                eps_vs_m3 = m3 - target_conf
                epsilon = max(eps_vs_natural, eps_vs_m3) + self.kappa

                attacked_logits[pos, target_id] += epsilon
                attacked_logits[pos, natural_id] -= epsilon
                already_matched = False

                new_argmax = torch.argmax(attacked_logits[pos]).item()
                assert new_argmax == target_id, (
                    f"Forcing failed at position {pos} (rel {rel_pos}).\n"
                    f"  target_id={target_id}  natural_id={natural_id}\n"
                    f"  natural_conf={natural_conf:.8f}  target_conf={target_conf:.8f}\n"
                    f"  m3={m3:.8f}  epsilon={epsilon:.8f}  new_argmax={new_argmax}"
                )

                tv = total_variation(row_softmax(row_before), row_softmax(attacked_logits[pos]))

                if exclude_token_ids is not None:
                    tv_excl_eos = row_tv_masked(row_before, attacked_logits[pos], exclude_token_ids)

            # Per-row empirical worst case (what TV would result if forced
            # to this row's own lowest-logit candidate) -- normalises how
            # "hard" this particular target was relative to the row.
            worst_case_tv, worst_case_target_id, worst_case_epsilon = worst_case_row_tv(
                row_before, natural_id, kappa=self.kappa
            )
            severity = relative_severity(tv, worst_case_tv)

            record = {
                "position": pos,
                "rel_position": rel_pos,
                "target_token_id": target_id,
                "natural_token_id": natural_id,
                "epsilon": epsilon,
                "already_matched": already_matched,
                "total_variation": tv,
                "worst_case_tv": worst_case_tv,
                "worst_case_target_id": worst_case_target_id,
                "worst_case_epsilon": worst_case_epsilon,
                "relative_severity": severity,
            }
            if exclude_token_ids is not None:
                # already_matched positions never got perturbed, so
                # content-only shift is 0.0 too, same convention as tv.
                record["total_variation_excl_eos"] = 0.0 if already_matched else tv_excl_eos
            records.append(record)

        return attacked_logits, records
