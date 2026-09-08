"""Lightweight chronological rules shared by the API and research engine."""

import math

import pandas as pd


def split_intervals(bars, train_ratio=.6, val_ratio=.2):
    """Share only boundary observations, never a reward or execution interval."""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one")
    rewards = len(bars) - 1
    train_end = math.floor(rewards * train_ratio)
    validation_end = train_end + math.floor(rewards * val_ratio)
    if train_end < 20 or validation_end - train_end < 5 or rewards - validation_end < 5:
        raise ValueError("Insufficient data: need at least 20 training, 5 validation and 5 test transitions")
    result = {}
    for name, start, end in (("train", 0, train_end),
                             ("validation", train_end, validation_end),
                             ("test", validation_end, rewards)):
        result[name] = {"start": start, "end": end, "reward_count": end - start,
                        "observation_start_date": pd.Timestamp(bars.index[start]).isoformat(),
                        "first_reward_date": pd.Timestamp(bars.index[start + 1]).isoformat(),
                        "last_reward_date": pd.Timestamp(bars.index[end]).isoformat()}
    return result
