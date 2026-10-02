import itertools
import json

import pytest
import torch

from akshara.model import Akshara, ModelConfig
from akshara.train import TrainConfig, adamw, next_token_loss, train, wsd

TINY = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=32)


def test_wsd_warms_up_holds_and_decays():
    lrs = [wsd(step, 100, 1.0, warmup_steps=10, decay_fraction=0.2) for step in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(lr == 1.0 for lr in lrs[10:80])
    assert lrs[80] < 1.0 and lrs[99] == pytest.approx(0.0)
    assert all(a >= b for a, b in itertools.pairwise(lrs[80:]))


def test_no_weight_decay_on_embeddings_or_norms():
    model = Akshara(TINY)
    decay, no_decay = adamw(model, TrainConfig()).param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert any(p is model.embed_tokens.weight for p in no_decay["params"])
    assert all(p.ndim == 2 for p in decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_gradient_accumulation_equals_one_big_batch():
    torch.manual_seed(0)
    batch = torch.randint(0, 64, (8, 17))
    big, small = Akshara(TINY), Akshara(TINY)
    small.load_state_dict(big.state_dict())
    next_token_loss(big, batch).backward()
    for part in batch.chunk(4):
        (next_token_loss(small, part) / 4).backward()
    for a, b in zip(big.parameters(), small.parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_a_tiny_model_memorises_a_batch_and_logs_its_curve(tmp_path):
    torch.manual_seed(0)
    batch = torch.randint(0, 64, (4, 17))
    config = TrainConfig(steps=150, peak_lr=1e-2, warmup_steps=10, bf16=False, log_every=10)
    history = train(Akshara(TINY), itertools.repeat(batch), config, log_path=tmp_path / "metrics.jsonl")
    assert history[0]["loss"] > 3.5 and history[-1]["loss"] < 0.1
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines == history and lines[-1]["step"] == 149 and lines[-1]["grad_norm"] > 0
