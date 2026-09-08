"""Utilities for detecting and filtering goal/target misalignment in
AdvBench (harmful_behaviors.csv).

Heuristic: a well-aligned (goal, target) pair should share most of its
distinctive vocabulary. Low content-word overlap flags a likely
row-alignment mismatch.
"""

import re
import pandas as pd

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
    "that", "this", "how", "someone", "some", "into", "using", "use",
    "provide", "write", "create", "develop", "design", "explain",
    "describe", "give", "detailed", "instructions", "step", "steps",
    "sure", "here", "is", "are", "which", "can", "be", "as", "their",
}


def content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in STOPWORDS}


def overlap_score(goal: str, target: str) -> float:
    goal_words = content_words(goal)
    if not goal_words:
        return 1.0  # nothing to compare, don't flag
    target_words = content_words(target)
    shared = goal_words & target_words
    return len(shared) / len(goal_words)


def filter_aligned_pairs(
    df: pd.DataFrame,
    goal_col: str = "goal",
    target_col: str = "target",
    threshold: float = 0.2,
):
    """Split an AdvBench dataframe into (clean_df, flagged_df) by
    goal/target content-word overlap. Adds an "orig_index" column to
    both, preserving the original row index for downstream tracing.
    """
    df = df.copy()
    df["orig_index"] = df.index
    df["overlap_score"] = df.apply(
        lambda r: overlap_score(r[goal_col], r[target_col]), axis=1
    )

    flagged_df = df[df["overlap_score"] < threshold].copy()
    clean_df = df[df["overlap_score"] >= threshold].copy().reset_index(drop=True)

    return clean_df, flagged_df
