"""马拉松赛事计时分析平台核心包."""

from .config import (
    TIMING_POINTS,
    TIMING_POINT_DISTANCES,
    TIMING_POINT_ORDER,
    CUTOFF_TIMES,
    GUN_START_TIME,
    WAVE_INTERVAL_MINUTES,
    AGE_GROUPS,
    MAX_PACE_KMH,
    MIN_SEGMENT_RATIO,
)
from .models import Runner, ChipRead, RaceResult, Suspicion, WaveAssignment

__all__ = [
    "TIMING_POINTS",
    "TIMING_POINT_DISTANCES",
    "TIMING_POINT_ORDER",
    "CUTOFF_TIMES",
    "GUN_START_TIME",
    "WAVE_INTERVAL_MINUTES",
    "AGE_GROUPS",
    "MAX_PACE_KMH",
    "MIN_SEGMENT_RATIO",
    "Runner",
    "ChipRead",
    "RaceResult",
    "Suspicion",
    "WaveAssignment",
]
