import json
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from akshara.train.schedule import wsd


@dataclass
class TrainConfig:
    steps: int = 1000
    peak_lr: float = 3e-3
    warmup_steps: int = 100
    decay_fraction: float = 0.2
    accumulation: int = 1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    clip: float = 1.0
    bf16: bool = True
    log_every: int = 10


def adamw(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only: no decay on embeddings (and the tied head) or on norm scales."""
    embedding_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)}
    decay, no_decay, seen = [], [], set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 and id(parameter) not in embedding_ids else no_decay).append(parameter)
    groups = [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config.peak_lr, betas=config.betas, eps=config.eps)


def next_token_loss(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))


def train(
    model: nn.Module,
    batches: Iterator[torch.Tensor],
    config: TrainConfig,
    device: str | torch.device = "cpu",
    log_path: str | Path | None = None,
    on_log: Callable[[dict], None] | None = None,
    strategy=None,
) -> list[dict]:
    """Training on one device or, with a `strategy` from `akshara.train.parallel`, on every rank of a process group.
    Each optimizer step consumes `accumulation` micro-batches of (batch, seq + 1) tokens per rank."""
    from akshara.train.parallel import Single

    strategy = strategy or Single()
    model = strategy.wrap(model.to(device))
    optimizer = strategy.optimizer(model, config)
    model.train()
    device_type = torch.device(device).type
    history, log_file = [], open(log_path, "w") if log_path else None
    started, tokens_seen = time.perf_counter(), 0
    try:
        for step in range(config.steps):
            lr = wsd(step, config.steps, config.peak_lr, config.warmup_steps, config.decay_fraction)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for micro in range(config.accumulation):
                tokens = next(batches).to(device)
                last = micro == config.accumulation - 1
                with strategy.no_sync(model) if not last else nullcontext():
                    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=config.bf16):
                        loss = next_token_loss(model, tokens)
                    (loss / config.accumulation).backward()
                loss_sum += loss.detach()
                tokens_seen += tokens[:, 1:].numel()
            strategy.before_step()
            if dist.is_initialized():
                dist.all_reduce(loss_sum)
                loss_sum /= dist.get_world_size()
                tokens_seen_all = tokens_seen * dist.get_world_size()
            else:
                tokens_seen_all = tokens_seen
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % config.log_every == 0 or step == config.steps - 1:
                elapsed = time.perf_counter() - started
                loss_value = loss_sum.item() / config.accumulation
                record = {"step": step, "loss": loss_value, "lr": lr, "grad_norm": grad_norm,
                          "tokens": tokens_seen_all, "tokens_per_second": tokens_seen_all / elapsed}  # fmt: skip
                history.append(record)
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                if on_log:
                    on_log(record)
    finally:
        if log_file:
            log_file.close()
    return history
