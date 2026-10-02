"""bench/run.py on Modal's 2x T4, for when Kaggle is unavailable.

    pip install modal
    modal setup                                   # once: log in
    modal run bench/modal_run.py                  # all sections
    modal run bench/modal_run.py --what contention

It clones the repository's main branch inside the container, so it runs
what is pushed, not local edits. A full run takes about 10 minutes of two
T4s, well inside Modal's free monthly credit. The JSON lines print as they
come and are saved to results-modal.jsonl here.
"""

import subprocess

import modal

REPO = "https://github.com/Shakhtar-Sankur/tandem"

image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("torch", "numpy")
app = modal.App("tandem-bench", image=image)


@app.function(gpu="T4:2", timeout=3600)
def bench(what: str) -> str:
    subprocess.run(["git", "clone", "--depth", "1", REPO, "/root/tandem"], check=True)
    subprocess.run(["python", "bench/run.py", what, "--out", "/root/results.jsonl"], cwd="/root/tandem", check=True)
    with open("/root/results.jsonl") as f:
        return f.read()


@app.local_entrypoint()
def main(what: str = "all"):
    out = bench.remote(what)
    with open("results-modal.jsonl", "w") as f:
        f.write(out)
    print(f"saved {len(out.splitlines())} lines to results-modal.jsonl")
