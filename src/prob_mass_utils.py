"""Probability-mass utilities for measuring adversarial intervention
magnitude in probability space (via total variation distance) rather
than raw logit space, which isn't directly comparable across positions
or attack mechanisms due to softmax's nonlinearity.
"""

import torch


def row_softmax(logits_row: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over a single row, computed in float64."""
    return torch.softmax(logits_row.to(torch.float64), dim=-1)


def total_variation(p_before: torch.Tensor, p_after: torch.Tensor) -> float:
    """TV(p, q) = 0.5 * sum_i |p_i - q_i| -- probability mass moved."""
    return 0.5 * torch.sum(torch.abs(p_after - p_before)).item()


def row_tv(logits_before_row: torch.Tensor, logits_after_row: torch.Tensor) -> float:
    """Softmax both rows, then compute TV distance between them."""
    p_before = row_softmax(logits_before_row)
    p_after = row_softmax(logits_after_row)
    return total_variation(p_before, p_after)


def row_softmax_masked(logits_row: torch.Tensor, exclude_ids) -> torch.Tensor:
    """Softmax over a row with specific vocab ids excluded (renormalized
    to ~0 mass on excluded ids). Falls back to a uniform distribution
    over the remaining ids if all mass would otherwise collapse.
    """
    masked = logits_row.clone().to(torch.float64)

    if isinstance(exclude_ids, (list, tuple, set, torch.Tensor)):
        masked[list(exclude_ids)] = float("-inf")

    probs = torch.softmax(masked, dim=-1)

    if torch.isnan(probs).any() or probs.sum() == 0:
        unmasked_mask = torch.ones_like(probs, dtype=torch.bool)
        if isinstance(exclude_ids, (list, tuple, set, torch.Tensor)):
            unmasked_mask[list(exclude_ids)] = False

        probs = torch.zeros_like(probs)
        num_unmasked = unmasked_mask.sum().item()
        if num_unmasked > 0:
            probs[unmasked_mask] = 1.0 / num_unmasked

    return probs


def row_tv_masked(logits_before_row: torch.Tensor, logits_after_row: torch.Tensor, exclude_ids) -> float:
    """Masked-softmax both rows (excluding exclude_ids), then TV distance."""
    p_before = row_softmax_masked(logits_before_row, exclude_ids)
    p_after = row_softmax_masked(logits_after_row, exclude_ids)

    if torch.isnan(p_before).any() or torch.isnan(p_after).any():
        return 0.0

    return total_variation(p_before, p_after)


def min_force_epsilon(logits_row: torch.Tensor, natural_id: int, target_id: int, kappa: float = 1e-5) -> float:
    """Minimal logit perturbation to make target_id the new row-argmax:
        epsilon = max((natural_conf - target_conf)/2, M3 - target_conf) + kappa
    where M3 = max over the row excluding {natural_id, target_id}.
    Mirrors ForcedResponseAttack's formula in attack.py (duplicated here
    so this module has no dependency on attack.py). Returns 0.0 if
    natural_id == target_id.
    """
    if natural_id == target_id:
        return 0.0
    natural_conf = logits_row[natural_id].item()
    target_conf = logits_row[target_id].item()

    row_masked = logits_row.clone()
    row_masked[natural_id] = float("-inf")
    row_masked[target_id] = float("-inf")
    m3 = row_masked.max().item()

    eps_vs_natural = (natural_conf - target_conf) / 2.0
    eps_vs_m3 = m3 - target_conf
    return max(eps_vs_natural, eps_vs_m3) + kappa


def target_force_cost(logits_row: torch.Tensor, natural_id: int, target_id: int, kappa: float = 1e-5):
    """Cost (epsilon, TV) to force target_id at this row, given
    natural_id is the row's current argmax. Pure evaluation -- does not
    mutate logits_row. Used for cost-based position ordering (ranking
    candidates before committing to force any of them).
    Returns (0.0, 0.0) if natural_id == target_id.
    """
    if natural_id == target_id:
        return 0.0, 0.0
    epsilon = min_force_epsilon(logits_row, natural_id, target_id, kappa)
    row_after = logits_row.clone()
    row_after[target_id] += epsilon
    row_after[natural_id] -= epsilon
    tv = total_variation(row_softmax(logits_row), row_softmax(row_after))
    return epsilon, tv


def worst_case_row_tv(logits_row: torch.Tensor, natural_id: int, kappa: float = 1e-5):
    """Per-row empirical worst case: the TV that would result if the
    target had been forced to this row's own lowest-logit candidate
    (excluding natural_id) -- the TV-maximizing choice within a fixed
    row, giving a valid per-row ceiling for the observed TV.

    Returns:
        (worst_case_tv, worst_case_target_id, worst_case_epsilon)
    """
    V = logits_row.shape[0]
    candidates = [i for i in range(V) if i != natural_id]
    if not candidates:
        return 0.0, natural_id, 0.0

    worst_case_target_id = min(candidates, key=lambda i: logits_row[i].item())
    epsilon = min_force_epsilon(logits_row, natural_id, worst_case_target_id, kappa)

    row_after = logits_row.clone()
    row_after[worst_case_target_id] += epsilon
    row_after[natural_id] -= epsilon

    tv = total_variation(row_softmax(logits_row), row_softmax(row_after))
    return tv, worst_case_target_id, epsilon


def relative_severity(observed_tv: float, worst_case_tv: float) -> float:
    """observed_tv / worst_case_tv for this row, bounded in [0, 1].
    Near 1.0 means the position's resistance was mostly about this
    specific target; well below 1.0 means most of the cost came from
    the row's overall confidence (often EOS/EOT-driven) rather than
    this particular target's difficulty. Returns 0.0 if worst_case_tv <= 0.
    """
    if worst_case_tv <= 0:
        return 0.0
    return observed_tv / worst_case_tv
