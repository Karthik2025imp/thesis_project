"""
prefix_run_4.py -- Process 4 of 5  (instruct-natural).

Model        : GSAI-ML/LLaDA-8B-Instruct
Prompt range : 41–50 (10 prompts)
Output CSV   : logs/gcg_complete/instruct-natural/insertion_accelerated_refinement_summary_p4.csv

Run with:
    cd ~/project
    python3 -u src/gcg_complete/new/instruct-natural/prefix_run_4.py 2>&1 | tee logs/gcg_complete/instruct-natural/prefix_run_4.log
"""

from prefix_run_worker import run_process

if __name__ == "__main__":
    run_process(process_id=4)