from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def save(path: str | Path, model: nn.Module, optimizer, step: int, loader=None) -> None:
    """A full resume state: model and optimizer through distributed checkpointing (each rank writes its own shards, and
    the result loads on any number of ranks), plus a small file per rank with the step, its data position and RNG.

    A ZeRO-1 optimizer is the exception: each rank's state covers different parameters, which distributed
    checkpointing would merge as if they were copies, so it goes into the rank's own file (same world size to resume).
    """
    path = Path(path)
    sharded = isinstance(optimizer, torch.optim.Optimizer)
    model_state, optimizer_state = get_state_dict(model, optimizer if sharded else [])
    dcp.save({"model": model_state, **({"optimizer": optimizer_state} if sharded else {})}, checkpoint_id=path)
    extra = {"step": step, "loader": loader.state_dict() if loader else None, "rng": torch.get_rng_state()}
    if not sharded:
        extra["optimizer"] = optimizer.state_dict()
    torch.save(extra, path / f"rank-{rank()}.pt")


def load(path: str | Path, model: nn.Module, optimizer, loader=None) -> int:
    """Restore what `save` wrote into an already wrapped model and its optimizer; returns the next step."""
    path = Path(path)
    sharded = isinstance(optimizer, torch.optim.Optimizer)
    model_state, optimizer_state = get_state_dict(model, optimizer if sharded else [])
    state = {"model": model_state, **({"optimizer": optimizer_state} if sharded else {})}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(
        model,
        optimizer if sharded else [],
        model_state_dict=state["model"],
        optim_state_dict=state.get("optimizer", {}),
    )
    extra = torch.load(path / f"rank-{rank()}.pt", weights_only=False)
    if not sharded:
        optimizer.load_state_dict(extra["optimizer"])
    if loader is not None:
        loader.load_state_dict(extra["loader"])
    torch.set_rng_state(extra["rng"])
    return extra["step"] + 1
