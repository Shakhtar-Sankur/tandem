"""tandem: distributed training from scratch on PyTorch tensors, with its own
collectives (no torch.distributed, no NCCL)."""

from .comm import CommError, Group, Work, launch

__all__ = ["CommError", "Group", "Work", "launch"]
