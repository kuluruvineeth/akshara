import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from akshara.train.loop import TrainConfig, adamw


def setup() -> tuple[int, int, torch.device]:
    """Join the process group described by torchrun's environment: NCCL on GPUs, gloo on CPU. Without torchrun, a
    single process on the best local device."""
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        return dist.get_rank(), dist.get_world_size(), torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size(), torch.device("cpu")


def unique_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(dict.fromkeys(parameter for parameter in model.parameters() if parameter.requires_grad))


class Single:
    """One device: nothing to synchronise."""

    def wrap(self, model: nn.Module) -> nn.Module:
        return model

    def optimizer(self, model: nn.Module, config: TrainConfig):
        return adamw(model, config)

    def no_sync(self, model: nn.Module):
        return nullcontext()

    def before_step(self) -> None:
        pass


class BucketedAllReduce(Single):
    """Data parallelism by hand. Every rank holds the whole model; gradients are averaged across ranks while backward
    is still running: parameters are grouped into buckets in the order their gradients arrive (last layer first), and
    a bucket is all-reduced asynchronously as soon as its last gradient is ready."""

    def __init__(self, bucket_mb: float = 25):
        self.bucket_bytes = bucket_mb * 2**20
        self.syncing = True

    def wrap(self, model: nn.Module) -> nn.Module:
        self.world = dist.get_world_size()
        parameters = unique_parameters(model)
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)
        self.buckets: list[list[nn.Parameter]] = [[]]
        size = 0
        for parameter in reversed(parameters):
            if size > self.bucket_bytes:
                self.buckets.append([])
                size = 0
            self.buckets[-1].append(parameter)
            size += parameter.numel() * parameter.element_size()
        self.bucket_of = {parameter: index for index, bucket in enumerate(self.buckets) for parameter in bucket}
        self.waiting = [len(bucket) for bucket in self.buckets]
        self.in_flight: list[tuple[int, torch.Tensor, dist.Work]] = []
        for parameter in parameters:
            parameter.register_post_accumulate_grad_hook(self.ready)
        return model

    def ready(self, parameter: nn.Parameter) -> None:
        if not self.syncing:
            return
        index = self.bucket_of[parameter]
        self.waiting[index] -= 1
        if self.waiting[index] == 0:
            flat = torch.cat([member.grad.flatten() for member in self.buckets[index]])
            self.in_flight.append((index, flat, dist.all_reduce(flat, async_op=True)))

    @contextmanager
    def no_sync(self, model: nn.Module):
        self.syncing = False
        try:
            yield
        finally:
            self.syncing = True

    def before_step(self) -> None:
        for index, flat, work in self.in_flight:
            work.wait()
            flat /= self.world
            offset = 0
            for member in self.buckets[index]:
                member.grad.copy_(flat[offset : offset + member.numel()].view_as(member.grad))
                offset += member.numel()
        self.in_flight.clear()
        self.waiting = [len(bucket) for bucket in self.buckets]


class TorchDDP(Single):
    """The same thing, done by PyTorch."""

    def wrap(self, model: nn.Module) -> nn.Module:
        return DistributedDataParallel(model)

    def no_sync(self, model: nn.Module):
        return model.no_sync()


STRATEGIES = {"single": Single, "hand-ddp": BucketedAllReduce, "ddp": TorchDDP}
