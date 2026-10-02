import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from akshara.model import Akshara, ModelConfig
from akshara.train import checkpoint
from akshara.train.loop import TrainConfig, train
from akshara.train.parallel import STRATEGIES

CONFIG = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=16)
TRAIN = TrainConfig(steps=6, peak_lr=1e-2, warmup_steps=2, accumulation=2, bf16=False, log_every=1)
WORLD = 2


def global_batches() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(0, CONFIG.vocab_size, (4, 17), generator=generator) for _ in range(TRAIN.steps * 2)]


def model() -> Akshara:
    torch.manual_seed(0)
    return Akshara(CONFIG)


def checksum(trained: torch.nn.Module) -> float:
    state = get_model_state_dict(trained, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return float(sum(tensor.double().sum() for tensor in state.values()))


def run_everything(rank: int, init_file: str, results: str, checkpoint_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    mine = [batch[rank::WORLD] for batch in global_batches()]
    report = {}
    for name in ("hand-ddp", "ddp", "zero1", "fsdp2"):
        trained = model()
        history = train(trained, iter(mine), TRAIN, strategy=STRATEGIES[name]())
        report[name] = {"losses": [record["loss"] for record in history], "checksum": checksum(trained)}

    for name in ("zero1", "fsdp2"):
        strategy = STRATEGIES[name]()
        folder = f"{checkpoint_dir}/{name}"

        def save_at_two(step, wrapped, optimizer, folder=folder):
            if step == 2:
                checkpoint.save(folder, wrapped, optimizer, step)

        first = train(model(), iter(mine), TRAIN, strategy=strategy, on_step=save_at_two)
        strategy = STRATEGIES[name]()
        resumed = strategy.wrap(model())
        optimizer = strategy.optimizer(resumed, TRAIN)
        start = checkpoint.load(folder, resumed, optimizer)
        rest = train(resumed, iter(mine[start * 2 :]), TRAIN, strategy=strategy, optimizer=optimizer, start_step=start)
        report[f"{name}-resume"] = {
            "straight": [record["loss"] for record in first][start:],
            "resumed": [record["loss"] for record in rest],
        }
    if rank == 0:
        with open(results, "w") as file:
            json.dump(report, file)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def distributed(tmp_path_factory) -> dict:
    folder = tmp_path_factory.mktemp("parallel")
    results = str(folder / "results.json")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    mp.spawn(run_everything, args=(str(folder / "init"), results, str(folder / "checkpoints")), nprocs=WORLD)
    with open(results) as file:
        return json.load(file)


@pytest.fixture(scope="module")
def reference() -> dict:
    trained = model()
    history = train(trained, iter(global_batches()), TRAIN)
    return {"losses": [record["loss"] for record in history], "checksum": checksum(trained)}


@pytest.mark.parametrize("name", ["hand-ddp", "ddp", "zero1", "fsdp2"])
def test_every_strategy_matches_one_device_on_the_same_global_batch(distributed, reference, name):
    assert distributed[name]["losses"] == pytest.approx(reference["losses"], rel=1e-5)
    assert distributed[name]["checksum"] == pytest.approx(reference["checksum"], rel=1e-5)


@pytest.mark.parametrize("name", ["zero1", "fsdp2"])
def test_a_resumed_run_continues_exactly(distributed, name):
    run = distributed[f"{name}-resume"]
    assert len(run["resumed"]) == TRAIN.steps - 3
    assert run["resumed"] == pytest.approx(run["straight"], rel=1e-6)
