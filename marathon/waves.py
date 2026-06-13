"""分区与波次编排，含约束验证."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import GUN_START_TIME, WAVE_INTERVAL_MINUTES
from .models import ValidationReport, WaveAssignment, WaveConfig


def _wave_start_time(wave_id: int) -> datetime:
    return GUN_START_TIME + timedelta(minutes=wave_id * WAVE_INTERVAL_MINUTES)


def assign_waves(
    roster_df: pd.DataFrame,
    config: WaveConfig = WaveConfig(),
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[WaveAssignment]]:
    """按预估成绩从快到慢分区、再分波，精英/轮椅优先."""
    df = roster_df.copy()

    if "wave_id" not in df.columns:
        df["wave_id"] = None
    if "start_zone" not in df.columns:
        df["start_zone"] = None

    df["_priority"] = 0
    df.loc[df["is_wheelchair"], "_priority"] = 2
    df.loc[df["is_elite"], "_priority"] = 3

    df["_sort_key"] = (
        -df["_priority"].astype(int) * 1000000
        + df["estimated_finish_hours"].astype(float) * 1.0
    )
    df = df.sort_values("_sort_key", kind="stable").reset_index(drop=True)

    zone_names = [f"ZONE_{chr(65 + i)}" for i in range(config.num_zones)]

    elite_wheelchair = df[df["_priority"] >= 2].copy()
    normal = df[df["_priority"] < 2].copy()

    assignment: Dict[str, Tuple[str, int]] = {}
    wave_bibs: Dict[int, List[str]] = {}
    zone_bibs: Dict[str, List[str]] = {z: [] for z in zone_names}

    wave_counter = 0
    current_wave: List[str] = []

    for _, r in elite_wheelchair.iterrows():
        bib = r["bib"]
        zone = zone_names[0]
        if len(current_wave) >= config.runners_per_wave:
            wave_bibs[wave_counter] = current_wave
            wave_counter += 1
            current_wave = []
        current_wave.append(bib)
        assignment[bib] = (zone, wave_counter)
        zone_bibs[zone].append(bib)

    normal = normal.sort_values("estimated_finish_hours", kind="stable").reset_index(drop=True)
    zone_size = max(len(normal) // config.num_zones, 1)

    for idx, (_, r) in enumerate(normal.iterrows()):
        bib = r["bib"]
        zone_idx = min(idx // max(zone_size, 1), config.num_zones - 1)
        zone = zone_names[zone_idx]

        if len(zone_bibs[zone]) >= config.runners_per_zone:
            for zi in range(config.num_zones):
                z = zone_names[zi]
                if len(zone_bibs[z]) < config.runners_per_zone:
                    zone = z
                    break

        current_wave_id = wave_counter + (len(current_wave) >= config.runners_per_wave)
        if len(current_wave) >= config.runners_per_wave:
            wave_bibs[wave_counter] = current_wave
            wave_counter += 1
            current_wave = []

        if zone_idx > 0 and len(current_wave) == 0 and wave_counter <= zone_idx * 2:
            wave_counter = zone_idx * 2

        current_wave.append(bib)
        assignment[bib] = (zone, wave_counter)
        zone_bibs[zone].append(bib)

    if current_wave:
        wave_bibs[wave_counter] = current_wave

    wave_assignments: List[WaveAssignment] = []
    zone_of_wave: Dict[int, str] = {}
    for wave_id, bibs in wave_bibs.items():
        zone_counts: Dict[str, int] = {}
        for b in bibs:
            z = assignment[b][0]
            zone_counts[z] = zone_counts.get(z, 0) + 1
        if zone_counts:
            dominant_zone = max(zone_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_zone = zone_names[0]
        zone_of_wave[wave_id] = dominant_zone

        wave_assignments.append(WaveAssignment(
            wave_id=wave_id,
            zone_id=dominant_zone,
            start_time=_wave_start_time(wave_id),
            bibs=list(bibs),
        ))

    def _get_zone(bib: str) -> str:
        return assignment.get(bib, (zone_names[0], 0))[0]

    def _get_wave(bib: str) -> int:
        return assignment.get(bib, (zone_names[0], 0))[1]

    df["start_zone"] = df["bib"].apply(_get_zone)
    df["wave_id"] = df["bib"].apply(_get_wave)

    return df, wave_assignments


def validate_assignments(
    roster_df: pd.DataFrame,
    wave_assignments: List[WaveAssignment],
    config: WaveConfig = WaveConfig(),
) -> ValidationReport:
    """验证容量、优先波次等约束."""
    report = ValidationReport(all_valid=True)

    zone_counts = roster_df.groupby("start_zone").size().to_dict()
    for z, cnt in zone_counts.items():
        if cnt > config.runners_per_zone:
            report.all_valid = False
            report.zone_capacity_violations.append(
                f"分区 {z} 容量 {cnt} 超过上限 {config.runners_per_zone}"
            )

    for wa in wave_assignments:
        if len(wa.bibs) > config.runners_per_wave:
            report.all_valid = False
            report.wave_capacity_violations.append(
                f"波次 {wa.wave_id} 人数 {len(wa.bibs)} 超过上限 {config.runners_per_wave}"
            )

    if wave_assignments:
        priority_wave_ids = sorted([wa.wave_id for wa in wave_assignments[:2]])
    else:
        priority_wave_ids = []

    elite_bibs = set(roster_df[roster_df["is_elite"]]["bib"].tolist())
    wheelchair_bibs = set(roster_df[roster_df["is_wheelchair"]]["bib"].tolist())

    priority_bibs_seen: set = set()
    for wa in wave_assignments:
        if wa.wave_id in priority_wave_ids:
            priority_bibs_seen.update(wa.bibs)

    elite_not_in_priority = elite_bibs - priority_bibs_seen
    for bib in elite_not_in_priority:
        w = roster_df[roster_df["bib"] == bib]["wave_id"].iloc[0]
        report.all_valid = False
        report.elite_in_priority_violations.append(
            f"精英选手 {bib} 被分到波次 {w}，未在优先波 {priority_wave_ids}"
        )

    wheelchair_not_in_priority = wheelchair_bibs - priority_bibs_seen
    for bib in wheelchair_not_in_priority:
        w = roster_df[roster_df["bib"] == bib]["wave_id"].iloc[0]
        report.all_valid = False
        report.wheelchair_in_priority_violations.append(
            f"轮椅选手 {bib} 被分到波次 {w}，未在优先波 {priority_wave_ids}"
        )

    return report


def wave_assignment_table(wave_assignments: List[WaveAssignment]) -> pd.DataFrame:
    """把WaveAssignment列表转成DataFrame方便展示."""
    rows = []
    for wa in wave_assignments:
        rows.append({
            "wave_id": wa.wave_id,
            "zone_id": wa.zone_id,
            "start_time": wa.start_time,
            "runner_count": len(wa.bibs),
        })
    return pd.DataFrame(rows)
