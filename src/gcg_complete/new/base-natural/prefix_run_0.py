"""
prefix_run_0.py -- Process 0 of 5  (base-natural).

Model        : GSAI-ML/LLaDA-8B-Base
Prompt range : 0–10 (11 prompts)
Output CSV   : logs/gcg_complete/base-natural/insertion_accelerated_refinement_summary_p0.csv

Run with:
    cd ~/project
    python3 -u src/gcg_complete/new/base-natural/prefix_run_0.py 2>&1 | tee logs/gcg_complete/base-natural/prefix_run_0.log
"""

from prefix_run_worker import run_process

if __name__ == "__main__":
    run_process(process_id=0)