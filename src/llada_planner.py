"""Faithful port of LLaDA's official reference generation algorithm
(remasking='low_confidence' branch only), ported from
https://github.com/ML-GSAI/LLaDA/blob/main/generate.py.
"""

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Port of ML-GSAI/LLaDA's add_gumbel_noise (float64 for precision)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Port of ML-GSAI/LLaDA's get_num_transfer_tokens: precomputes how
    many positions to commit at each step, evenly divided with the
    remainder distributed to the first `remainder` steps.

    Args:
        mask_index: [B, block_length] bool, True at masked positions
                    within the current block
        steps:      diffusion steps allotted to this block

    Returns:
        [B, steps] int64 -- number of positions to commit at each step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


class LLaDAPlanner:
    """Per-step position-selection + token-sampling logic from
    ML-GSAI/LLaDA/generate.py (remasking='low_confidence'), for a single
    sequence (batch size 1). Unlike ConfidencePlanner, position choice
    and token sampling happen together: candidates are sampled for
    every position, ranked by the clean probability of the sampled
    token, then the top-k are committed.
    """

    def __init__(self, mask_id: int = 126336):
        self.mask_id = mask_id

    def step(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        block_start: int,
        block_end: int,
        num_transfer_tokens: int,
        temperature: float = 0.0,
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
        eos_token_id: int = 126081,
        eot_token_id: int = 126348,
    ):
        """One diffusion step, following generate.py's per-step body.

        Args:
            x:            [1, seq_len] or [seq_len] current sequence
            logits:       [1, seq_len, vocab] or [seq_len, vocab]
            block_start:  absolute index of this block's first position
            block_end:    absolute index one past this block's last
                          position; positions outside [block_start, block_end)
                          are never eligible this step
            num_transfer_tokens: positions to commit this step
            temperature:  Gumbel sampling temperature (0 = deterministic)
            logits_eos_inf: if True, set EOS logit to -inf before
                          sampling, so EOS can never be sampled this step
            confidence_eos_eot_inf: if True, suppress confidence wherever
                          the sampled token is EOS or EOT. Implements the
                          documented intent of the official flag -- the
                          fetched source's own line for this doesn't
                          appear to gate on the sampled token; see note
                          in the method body.
            eos_token_id, eot_token_id: special token ids (official defaults)

        Returns:
            x:                  [1, seq_len], only selected positions written
            selected_positions: 1D LongTensor of positions committed this
                                step (added beyond the official return
                                value, since callers here need to know
                                which position(s) were chosen)
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if logits_eos_inf:
            logits = logits.clone()
            logits[:, :, eos_token_id] = -torch.inf

        mask_index = (x == self.mask_id)

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [1, seq_len]

        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [1, seq_len]

        if confidence_eos_eot_inf:
            # Official source line doesn't appear to gate on the sampled
            # token; this implements the documented intent instead.
            x0_p = x0_p.clone()
            x0_p[x0 == eos_token_id] = -np.inf
            x0_p[x0 == eot_token_id] = -np.inf

        x0_p[:, block_end:] = -np.inf
        x0_p[:, :block_start] = -np.inf  # defensive

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        n_eligible = int((confidence[0] > -np.inf).sum().item())
        k = min(int(num_transfer_tokens), n_eligible)
        if k > 0:
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True

        x_new = x.clone()
        x_new[transfer_index] = x0[transfer_index]
        selected_positions = transfer_index[0].nonzero(as_tuple=True)[0]
        return x_new, selected_positions


def generate_llada(
    llada,
    x: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    steps: int,
    block_length: int = None,
    temperature: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    step_callback=None,
):
    """Port of ML-GSAI/LLaDA/generate.py's outer generation loop, built
    on LLADAWrapper and LLaDAPlanner.

    Args:
        llada:         LLADAWrapper instance
        x:             [1, prompt_len + gen_length], prompt + MASK tokens
        prompt_len:    number of prompt tokens
        gen_length:    number of response positions
        steps:         total diffusion steps across all blocks (must be
                       divisible by num_blocks)
        block_length:  block size; defaults to gen_length (single block)
        temperature:   Gumbel sampling temperature
        logits_eos_inf, confidence_eos_eot_inf: see LLaDAPlanner.step
        step_callback: optional callable(x, logits, block_idx, step_idx,
                       block_start, block_end), invoked before each
                       step's selection. If it returns a non-None x,
                       that replaces the sequence for this step and the
                       normal planner step is skipped -- return None for
                       natural generation.

    Returns:
        x: [1, prompt_len + gen_length]
    """
    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0, (
        f"block_length ({block_length}) must divide gen_length ({gen_length})"
    )
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, (
        f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
    )
    steps_per_block = steps // num_blocks

    planner = LLaDAPlanner(mask_id=llada.mask_token_id)
    x = x.clone()

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == llada.mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
            logits = llada.get_logits(x)

            if step_callback is not None:
                overridden = step_callback(x, logits, block_idx, step_idx, block_start, block_end)
                if overridden is not None:
                    x = overridden
                    continue

            x = planner.step(
                x=x,
                logits=logits,
                block_start=block_start,
                block_end=block_end,
                num_transfer_tokens=num_transfer_tokens[0, step_idx].item(),
                temperature=temperature,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
            )[0]

    return x
