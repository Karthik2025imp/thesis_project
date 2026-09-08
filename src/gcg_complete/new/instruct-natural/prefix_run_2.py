"""
prefix_run_2.py -- Process 2 of 5  (instruct-natural).

Model        : GSAI-ML/LLaDA-8B-Instruct
Prompt range : 21–30 (10 prompts)
Output CSV   : logs/gcg_complete/instruct-natural/insertion_accelerated_refinement_summary_p2.csv

Run with:
    cd ~/project
    python3 -u src/gcg_complete/new/instruct-natural/prefix_run_2.py 2>&1 | tee logs/gcg_complete/instruct-natural/prefix_run_2.log
"""

from prefix_run_worker import run_process

if __name__ == "__main__":
    run_process(process_id=2)