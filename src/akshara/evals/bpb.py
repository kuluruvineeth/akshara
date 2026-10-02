import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


def token_bytes(tokenizer: Tokenizer) -> torch.Tensor:
    """UTF-8 bytes of every token id; 0 for special tokens. In a byte-level BPE vocabulary each character of a token's
    string stands for exactly one byte."""
    special = {token.content for token in tokenizer.get_added_tokens_decoder().values()}
    sizes = torch.zeros(tokenizer.get_vocab_size(), dtype=torch.int64)
    for token, token_id in tokenizer.get_vocab().items():
        sizes[token_id] = 0 if token in special else len(token)
    return sizes


@torch.no_grad()
def bits_per_byte(
    model: torch.nn.Module, batches: Iterable[torch.Tensor], sizes: torch.Tensor, device: str | torch.device
) -> float:
    """Next-token loss in bits per UTF-8 byte of the targets, so models with different tokenizers compare. Special
    tokens (end of text) carry no bytes and are left out of both sums."""
    model.eval()
    sizes = sizes.to(device)
    nats, total_bytes = 0.0, 0
    for tokens in batches:
        tokens = tokens.to(device)
        logits = model(tokens[:, :-1]).float()
        targets = tokens[:, 1:]
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
        target_bytes = sizes[targets.flatten()]
        nats += float((loss * (target_bytes > 0)).sum())
        total_bytes += int(target_bytes.sum())
    return nats / (math.log(2) * total_bytes)
