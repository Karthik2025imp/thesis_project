"""
prefix_run_1.py -- Process 1 of 5  (instruct-natural).

Model        : GSAI-ML/LLaDA-8B-Instruct
Prompt range : 11–20 (10 prompts)
Output CSV   : logs/gcg_complete/instruct-natural/insertion_accelerated_refinement_summary_p1.csv

Run with:
    cd ~/project
    python3 -u src/gcg_complete/new/instruct-natural/prefix_run_1.py 2>&1 | tee logs/gcg_complete/instruct-natural/prefix_run_1.log
"""

from prefix_run_worker import run_process

if __name__ == "__main__":
    run_process(process_id=1)