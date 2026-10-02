import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from akshara.data import PackedLoader
from akshara.model import TIERS, Akshara, ModelConfig
from akshara.train import checkpoint
from akshara.train.loop import TrainConfig, train
from akshara.train.parallel import STRATEGIES, setup

TINY = ModelConfig(d_model=256, n_layers=6, n_heads=4, n_kv_heads=2, ffn_dim=768, max_seq_len=512)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Akshara; run under torchrun for more than one rank")
    parser.add_argument("data", type=Path, help="folder of merged token files")
    parser.add_argument("output", type=Path, help="log and checkpoints")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="single")
    parser.add_argument("--tier", choices=[*TIERS, "tiny"], default="tiny")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4, help="rows per rank per micro-batch")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--fp32", action="store_true", help="no bf16 autocast")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="a checkpoint folder written by this command")
    parser.add_argument("--seed", type=int, default=0, help="the same on every rank, so all ranks start equal")
    args = parser.parse_args()

    rank, world, device = setup()
    torch.manual_seed(args.seed)
    config = TINY if args.tier == "tiny" else TIERS[args.tier]
    train_config = TrainConfig(
        steps=args.steps,
        peak_lr=args.lr,
        warmup_steps=args.warmup,
        accumulation=args.accumulation,
        bf16=not args.fp32,
        log_every=args.log_every,
    )
    loader = PackedLoader(args.data, args.batch_size, args.seq_len, rank=rank, world_size=world)
    strategy = STRATEGIES[args.strategy]()
    model, optimizer, start = Akshara(config), None, 0
    if args.resume:
        model = strategy.wrap(model.to(device))
        optimizer = strategy.optimizer(model, train_config)
        start = checkpoint.load(args.resume, model, optimizer, loader)

    def save(step: int, wrapped, current_optimizer) -> None:
        if args.save_every and (step + 1) % args.save_every == 0:
            checkpoint.save(args.output / f"step-{step:06d}", wrapped, current_optimizer, step, loader)

    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "train.jsonl" if rank == 0 else None
    train(
        model,
        loader,
        train_config,
        device,
        log,
        strategy=strategy,
        optimizer=optimizer,
        start_step=start,
        on_step=save,
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
