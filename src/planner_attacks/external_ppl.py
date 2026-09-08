"""External-LM perplexity for response quality assessment.

Scores generated responses with an independent model (Qwen2.5-0.5B,
chosen for strong English/Chinese bilingual coverage) rather than
LLaDA's own self-scored pseudo-PPL, which is biased toward trivially
"fluent" repetition-collapse text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def load_external_lm(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    """Load the external scoring model once, reuse across all runs."""
    print(f"Loading external LM for perplexity scoring: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("External LM loaded.")
    return tokenizer, model


@torch.no_grad()
def compute_external_ppl(text: str, tokenizer, model, device: str = "cuda", max_length: int = 512):
    """Autoregressive perplexity of `text` under the external model.
    Returns (ppl, mean_nll), or (None, None) if too short to score.
    """
    if not isinstance(text, str) or not text.strip():
        return None, None

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return None, None

    outputs = model(input_ids, labels=input_ids)
    mean_nll = outputs.loss.item()
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    return ppl, mean_nll
