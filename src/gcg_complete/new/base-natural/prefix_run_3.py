"""
prefix_run_3.py -- Process 3 of 5  (base-natural).

Model        : GSAI-ML/LLaDA-8B-Base
Prompt range : 31–40 (10 prompts)
Output CSV   : logs/gcg_complete/base-natural/insertion_accelerated_refinement_summary_p3.csv

Run with:
    cd ~/project
    python3 -u src/gcg_complete/new/base-natural/prefix_run_3.py 2>&1 | tee logs/gcg_complete/base-natural/prefix_run_3.log
"""

from prefix_run_worker import run_process

if __name__ == "__main__":
    run_process(process_id=3)