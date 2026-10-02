# tandem

Distributed training written from scratch on PyTorch tensors and autograd,
with its own collectives: no `torch.distributed`, no NCCL. Data
parallelism (DDP), ZeRO stages 1-3 (FSDP-style sharding) and pipeline
parallelism (GPipe, 1F1B), each checked bit for bit against torch's own
DDP or against single-process training.

Results on 2x T4 are being measured; see `bench/run.py`.

```sh
pip install torch pytest numpy
PYTHONPATH=.:tests python -m pytest -q tests     # 18 tests on CPU processes
python bench/run.py all                          # on a machine with 2 GPUs
```
