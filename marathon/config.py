"""全局配置常量."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

GUN_START_TIME = datetime(2026, 3, 15, 7, 30, 0)

TIMING_POINTS: List[str] = ["START", "KM5", "KM10", "HALF", "KM30", "FINISH"]

TIMING_POINT_DISTANCES: Dict[str, float] = {
    "START": 0.0,
    "KM5": 5.0,
    "KM10": 10.0,
    "HALF": 21.0975,
    "KM30": 30.0,
    "FINISH": 42.195,
}

TIMING_POINT_ORDER: Dict[str, int] = {p: i for i, p in enumerate(TIMING_POINTS)}

CUTOFF_TIMES: Dict[str, timedelta] = {
    "START": timedelta(hours=1),
    "KM5": timedelta(hours=1, minutes=15),
    "KM10": timedelta(hours=2),
    "HALF": timedelta(hours=3),
    "KM30": timedelta(hours=4, minutes=15),
    "FINISH": timedelta(hours=6),
}

AGE_GROUPS: List[Tuple[str, int, int]] = [
    ("18-24", 18, 24),
    ("25-29", 25, 29),
    ("30-34", 30, 34),
    ("35-39", 35, 39),
    ("40-44", 40, 44),
    ("45-49", 45, 49),
    ("50-54", 50, 54),
    ("55-59", 55, 59),
    ("60-64", 60, 64),
    ("65+", 65, 120),
]

MAX_PACE_KMH: float = 25.0
MIN_SEGMENT_RATIO: float = 0.7

WAVE_INTERVAL_MINUTES: int = 5

DEFAULT_WAVE_CAPACITY: int = 500
DEFAULT_ZONE_CAPACITY: int = 2000

BOUNCE_WINDOW_SECONDS: int = 5
