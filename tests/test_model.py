import pytest
import torch
import torch.nn.functional as F

from akshara.model import TIERS, Akshara, ModelConfig
from akshara.model.layers import apply_rope, rope_tables

TINY = ModelConfig(vocab_size=128, d_model=48, n_layers=2, n_heads=6, n_kv_heads=2, ffn_dim=96, max_seq_len=32)


def test_tier_s_has_the_planned_parameter_count():
    model = Akshara(TIERS["S"])
    d, kv, ffn = 576, 3 * 64, 1536
    per_layer = (d * d + 2 * d * kv + d * d) + 3 * d * ffn + 2 * d
    assert model.num_parameters() == 65_536 * d + 30 * per_layer + d == 143_952_192
    assert round(model.num_parameters() / 1e9, 3) == 0.144
    assert model.lm_head.weight is model.embed_tokens.weight


@pytest.mark.parametrize(("tier", "billions"), [("M", 0.805), ("L", 2.95)])
def test_larger_tiers_match_the_plan(tier, billions):
    with torch.device("meta"):
        model = Akshara(TIERS[tier])
    assert round(model.num_parameters() / 1e9, 3 if tier == "M" else 2) == billions


def test_qk_norm_adds_two_norms_per_layer():
    with torch.device("meta"):
        plain, normed = Akshara(TIERS["S"]), Akshara(ModelConfig(qk_norm=True))
    assert normed.num_parameters() - plain.num_parameters() == 30 * 2 * 64


def test_logits_have_the_right_shape_and_never_look_ahead():
    torch.manual_seed(0)
    model = Akshara(TINY).eval()
    ids = torch.randint(0, 128, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, 128)
    changed = ids.clone()
    changed[:, 6] = (changed[:, 6] + 1) % 128
    with torch.no_grad():
        assert torch.allclose(model(changed)[:, :6], logits[:, :6], atol=1e-6)
        assert not torch.allclose(model(changed)[:, 6:], logits[:, 6:])


def test_grouped_query_attention_equals_repeating_the_kv_heads():
    torch.manual_seed(0)
    attention = Akshara(TINY).layers[0].self_attn
    x = torch.randn(1, 7, 48)
    cos, sin = rope_tables(8, 7, 1e5)
    q = apply_rope(attention.q_proj(x).view(1, 7, 6, 8).transpose(1, 2), cos, sin)
    k = apply_rope(attention.k_proj(x).view(1, 7, 2, 8).transpose(1, 2), cos, sin)
    v = attention.v_proj(x).view(1, 7, 2, 8).transpose(1, 2)
    k, v = k.repeat_interleave(3, dim=1), v.repeat_interleave(3, dim=1)
    scores = (q @ k.transpose(-1, -2)) / 8**0.5
    scores = scores.masked_fill(torch.triu(torch.ones(7, 7, dtype=torch.bool), 1), float("-inf"))
    expected = attention.o_proj((F.softmax(scores, -1) @ v).transpose(1, 2).reshape(1, 7, 48))
    assert torch.allclose(attention(x, cos, sin), expected, atol=1e-5)


def test_rope_scores_depend_only_on_the_distance_between_positions():
    cos, sin = rope_tables(8, 32, 1e5)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(m, n):
        return (apply_rope(q, cos[m : m + 1], sin[m : m + 1]) * apply_rope(k, cos[n : n + 1], sin[n : n + 1])).sum()

    assert torch.allclose(score(3, 1), score(20, 18), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-5)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_logits_match_transformers_with_the_same_weights(qk_norm):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = ModelConfig(**{**TINY.__dict__, "qk_norm": qk_norm})
    ours = Akshara(config).eval()
    hf_config = dict(
        vocab_size=128, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=6,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=32, rope_theta=1e5, rms_norm_eps=1e-5,
        tie_word_embeddings=True, attention_bias=False,
    )  # fmt: skip
    if qk_norm:
        theirs = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**hf_config))
    else:
        theirs = transformers.LlamaForCausalLM(transformers.LlamaConfig(**hf_config, mlp_bias=False))
    state = {
        ("" if name.startswith("lm_head") else "model.") + name: value for name, value in ours.state_dict().items()
    }
    theirs.load_state_dict(state, strict=False)
    theirs.eval()
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        assert torch.allclose(ours(ids), theirs(ids).logits, atol=1e-4)
