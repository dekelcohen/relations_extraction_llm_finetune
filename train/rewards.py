"""GRPO reward functions (TRL signature): extra dataset columns arrive as kwargs lists.

Columns produced by `python -m re_synth export`: gt, decoys, text, slot_types (JSON strings).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from re_synth.score import parse_prediction, reward_from_metrics, sample_f1, score_sample  # noqa: E402


def _j(x):
    return json.loads(x) if isinstance(x, str) else x


def _completion_text(c) -> str:
    if isinstance(c, list):  # conversational
        return c[-1].get("content", "") if c and isinstance(c[-1], dict) else ""
    return c if isinstance(c, str) else ""


def extraction_reward(completions, gt, decoys, text, slot_types, **kwargs) -> list[float]:
    """Composite: 0.6 slot-F1 + 0.15 record-F1 + 0.15 null-acc + 0.1 (1-halluc) - 0.2 decoy-rate; 0 if invalid JSON."""
    out = []
    for c, g, d, t, st in zip(completions, gt, decoys, text, slot_types):
        try:
            recs, ok = parse_prediction(_completion_text(c))
            m = score_sample(recs, ok, _j(g), _j(d), t, _j(st))
            out.append(float(reward_from_metrics(m)))
        except Exception:  # never crash training on a weird sample
            out.append(0.0)
    return out


def slot_f1_reward(completions, gt, decoys, text, slot_types, **kwargs) -> list[float]:
    """Plain slot-F1 (logged as a separate reward; give it weight 0 in reward_weights to only monitor)."""
    out = []
    for c, g, d, t, st in zip(completions, gt, decoys, text, slot_types):
        try:
            recs, ok = parse_prediction(_completion_text(c))
            out.append(float(sample_f1(score_sample(recs, ok, _j(g), _j(d), t, _j(st)))
                             if ok else 0.0))
        except Exception:
            out.append(0.0)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    return [1.0 if parse_prediction(_completion_text(c))[1] else 0.0 for c in completions]
