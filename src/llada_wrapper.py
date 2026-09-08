"""LLaDA-8B-Instruct wrapper for masked discrete diffusion inference.

Reference: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
"""

import torch
from transformers import AutoTokenizer, AutoModel

MASK_TOKEN_ID = 126336  # LLaDA-8B-Instruct


class LLADAWrapper:
    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", device: str = "cuda"):
        self.device = device
        self.mask_token_id = MASK_TOKEN_ID

        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print(f"Loading model from {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("Model loaded.")

    def build_input(self, prompt: str, gen_length: int = 128, use_chat_template: bool = True):
        """Build [prompt tokens] + [MASK * gen_length].

        Returns:
            x:            [1, prompt_len + gen_length]
            prompt_index: [1, prompt_len + gen_length]  True = prompt position
        """
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        else:
            prompt_text = prompt

        encoded = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(self.device)

        x = torch.full(
            (1, prompt_ids.shape[1] + gen_length),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_ids.shape[1]] = prompt_ids
        prompt_index = (x != self.mask_token_id)

        return x, prompt_index

    @torch.no_grad()
    def get_logits_batch(self, x_batch: torch.Tensor):
        """Batched forward pass, for scoring many masked variants at once.
        Returns logits: [B, seq_len, vocab_size].
        """
        outputs = self.model(x_batch)
        return outputs.logits

    @torch.no_grad()
    def get_logits(self, x: torch.Tensor):
        """Single forward pass. Returns logits: [seq_len, vocab_size]."""
        outputs = self.model(x)
        return outputs.logits[0]

    def get_mask_positions(self, x: torch.Tensor):
        """Return indices of currently masked positions."""
        return (x[0] == self.mask_token_id).nonzero(as_tuple=True)[0]

    def unmask_positions(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        logits: torch.Tensor,
        temperature: float = 0.0,
        suppress_token_ids: list = None,
    ):
        """Fill `positions` with sampled (temperature > 0, Gumbel) or
        greedy (temperature == 0) tokens from `logits`.

        suppress_token_ids: optional vocab ids to zero out before
        sampling, in both branches.
        """
        x = x.clone()
        for pos in positions:
            pos_logits = logits[pos].to(torch.float64)

            if suppress_token_ids:
                for tid in suppress_token_ids:
                    pos_logits[tid] = -1e9

            if temperature > 0:
                # Gumbel noise sampling (official LLaDA inference method)
                noise = torch.rand_like(pos_logits)
                gumbel_noise = (-torch.log(noise)) ** temperature
                pos_logits = pos_logits.exp() / gumbel_noise
                token_id = torch.argmax(pos_logits).item()
            else:
                # Greedy argmax, masking out EOS/MASK to avoid collapse
                pos_logits[self.tokenizer.eos_token_id] = -1e9
                pos_logits[self.mask_token_id] = -1e9
                token_id = torch.argmax(pos_logits).item()

            x[0, pos] = token_id
        return x

    def extend_with_mask(self, x: torch.Tensor, num_new_tokens: int):
        """Append `num_new_tokens` fresh MASK slots to the end of the sequence."""
        new_mask = torch.full(
            (1, num_new_tokens), self.mask_token_id, dtype=x.dtype, device=x.device,
        )
        return torch.cat([x, new_mask], dim=1)

    def get_eos_like_token_ids(self):
        """Return vocab ids for EOS/EOT/termination-like special tokens
        (LLaDA has multiple distinct termination tokens, not just one).
        """
        ids = set(self.tokenizer.all_special_ids)
        ids.add(self.mask_token_id)
        return sorted(ids)

    def force_write_positions(self, x: torch.Tensor, positions: torch.Tensor, token_ids_by_position: dict):
        """Directly write specific token ids at specific positions,
        bypassing unmask_positions' sampling/greedy logic.
        """
        x = x.clone()
        for pos in positions:
            pos = int(pos.item()) if torch.is_tensor(pos) else int(pos)
            x[0, pos] = token_ids_by_position[pos]
        return x

    def decode_response(self, x: torch.Tensor, prompt_len: int):
        """Decode generated tokens (response only)."""
        response_ids = x[0, prompt_len:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)
