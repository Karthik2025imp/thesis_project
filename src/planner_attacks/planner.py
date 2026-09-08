"""Confidence-based planner for LLaDA masked diffusion.

Selects which masked position to unmask next based on the highest
max-logit confidence across the vocabulary:
    f_theta(l) = argmax_i [ max_j l_ij ]   for i in masked positions
"""

import torch


class ConfidencePlanner:
    def __init__(self):
        pass

    def row_confidences(self, logits: torch.Tensor, mask_positions: torch.Tensor):
        masked_logits = logits[mask_positions]  # [num_masked, vocab_size]
        masked_probs = torch.softmax(masked_logits.float(), dim=-1)
        row_max_values, row_max_indices = torch.max(masked_probs, dim=-1)
        return row_max_values, row_max_indices

    def select_token(self, logits: torch.Tensor, mask_positions: torch.Tensor):
        """Select the masked position with highest confidence."""
        row_max_values, row_max_indices = self.row_confidences(logits, mask_positions)

        local_selected = torch.argmax(row_max_values).item()
        selected_position = mask_positions[local_selected].item()

        return {
            "selected_position": selected_position,
            "local_selected": local_selected,
            "selected_vocab_index": row_max_indices[local_selected].item(),
            "selected_confidence": row_max_values[local_selected].item(),
            "row_confidences": row_max_values,  # [num_masked]
            "row_indices": row_max_indices,     # [num_masked]
            "mask_positions": mask_positions,
        }

    def select_top_k_tokens(self, logits: torch.Tensor, mask_positions: torch.Tensor, k: int):
        """Select the top-k most confident masked positions to unmask this step."""
        row_max_values, _ = self.row_confidences(logits, mask_positions)

        k = min(k, len(mask_positions))
        _, top_local = torch.topk(row_max_values, k)
        selected_positions = mask_positions[top_local]
        return selected_positions
