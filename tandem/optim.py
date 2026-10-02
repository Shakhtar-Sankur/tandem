"""AdamW on flat shards of the parameters, for the ZeRO stages.

The arithmetic is torch.optim.AdamW's single-tensor path (foreach=False,
not capturable), operation for operation. Every operation is elementwise, so
updating a rank's shard of a flat buffer gives exactly the bits torch's
AdamW gives for those elements of the whole tensors."""

import torch


class ShardedAdamW:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = {}  # key -> (step, exp_avg, exp_avg_sq)

    def update(self, key, param, grad):
        """One step for one shard: `param` and `grad` are same-shaped tensors;
        `key` names the shard's optimizer state."""
        beta1, beta2 = self.betas
        lr, wd, eps = self.lr, self.weight_decay, self.eps
        if key not in self.state:
            self.state[key] = [0.0, torch.zeros_like(param), torch.zeros_like(param)]
        st = self.state[key]
        st[0] += 1.0
        step, exp_avg, exp_avg_sq = st
        if param.numel() == 0:
            return
        if wd != 0:
            param.mul_(1 - lr * wd)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for _, a, b in self.state.values() for t in (a, b))
