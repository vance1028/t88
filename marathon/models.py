"""数据模型定义."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class Runner:
    bib: str
    name: str
    gender: str
    age: int
    estimated_finish_hours: float
    start_zone: Optional[str] = None
    wave_id: Optional[int] = None
    is_elite: bool = False
    is_wheelchair: bool = False


@dataclass
class ChipRead:
    bib: str
    timing_point: str
    timestamp: datetime


@dataclass
class Suspicion:
    bib: str
    reason: str
    details: str = ""


@dataclass
class SegmentResult:
    timing_point: str
    timestamp: Optional[datetime]
    distance: float
    gun_time: Optional[float]
    net_time: Optional[float]
    segment_time: Optional[float]
    segment_pace: Optional[float]
    is_interpolated: bool = False
    is_missing: bool = False
    is_cutoff: bool = False


@dataclass
class RaceResult:
    bib: str
    status: str
    gun_time: Optional[float]
    net_time: Optional[float]
    average_pace: Optional[float]
    segments: Dict[str, SegmentResult]
    suspicions: List[Suspicion] = field(default_factory=list)


@dataclass
class WaveAssignment:
    wave_id: int
    zone_id: str
    start_time: datetime
    bibs: List[str]


@dataclass
class WaveConfig:
    num_zones: int = 5
    runners_per_zone: int = 2000
    runners_per_wave: int = 500
    wave_interval_minutes: int = 5


@dataclass
class ValidationReport:
    all_valid: bool
    zone_capacity_violations: List[str] = field(default_factory=list)
    wave_capacity_violations: List[str] = field(default_factory=list)
    elite_in_priority_violations: List[str] = field(default_factory=list)
    wheelchair_in_priority_violations: List[str] = field(default_factory=list)
