"""Early selective attack scheduler for the LLaDA diffusion loop.

Attacks only during the first `attack_ratio` fraction of steps, and
computes how many tokens to unmask per step (LLaDA's linear schedule:
equal unmasking across all steps).
"""


class EarlyAttackScheduler:
    def __init__(self, total_steps: int, attack_ratio: float = 0.20):
        self.total_steps = total_steps
        self.attack_ratio = attack_ratio
        self.attack_until_step = int(total_steps * attack_ratio)

    def should_attack(self, step: int) -> bool:
        """Return True if attack should be applied at this step."""
        return step < self.attack_until_step

    def tokens_to_unmask(self, num_masked: int, steps_remaining: int) -> int:
        """Distribute unmasking evenly across remaining steps."""
        if steps_remaining <= 0:
            return num_masked
        return max(1, num_masked // steps_remaining)
