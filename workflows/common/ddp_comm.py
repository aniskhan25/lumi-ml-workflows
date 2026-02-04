import time

import torch.distributed as dist


class DDPCommStats:
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.times_ms = []
        self.bytes_total = 0

    def set_step(self, step_idx):
        self.current_step = step_idx

    def should_measure(self):
        return self.current_step >= self.warmup_steps


def register_ddp_comm_hook(ddp_model, stats):
    if not hasattr(ddp_model, "register_comm_hook"):
        return False

    def hook(state, bucket):
        start = time.perf_counter()
        tensor = bucket.buffer()
        work = dist.all_reduce(tensor, async_op=True)
        fut = work.get_future()
        step_idx = state.current_step
        warmup = state.warmup_steps

        def callback(_):
            end = time.perf_counter()
            if step_idx >= warmup:
                state.times_ms.append((end - start) * 1000.0)
                state.bytes_total += tensor.numel() * tensor.element_size()
            return tensor

        return fut.then(callback)

    ddp_model.register_comm_hook(stats, hook)
    return True
