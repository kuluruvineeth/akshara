def wsd(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    decay_fraction: float = 0.2,
    final_ratio: float = 0.0,
) -> float:
    """Warmup-stable-decay: linear warmup, a long constant phase, then a linear decay over the last fraction."""
    decay_steps = max(1, round(total_steps * decay_fraction))
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return peak_lr
    progress = min(1.0, (step - decay_start + 1) / decay_steps)
    return peak_lr * (1 - progress * (1 - final_ratio))
