"""作弊与异常检测：抄近道、配速异常、抢跑."""

from __future__ import annotations

from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import (
    GUN_START_TIME,
    MAX_PACE_KMH,
    MIN_SEGMENT_RATIO,
    TIMING_POINT_DISTANCES,
    TIMING_POINT_ORDER,
    TIMING_POINTS,
    WAVE_INTERVAL_MINUTES,
)


def detect_shortcut(
    seg_df: pd.DataFrame,
    bib: str,
) -> List[Tuple[str, str]]:
    """检测疑似抄近道：有终点成绩但存在连续的中间点缺失."""
    rows = seg_df[seg_df["bib"] == bib].sort_values("timing_point", key=lambda s: s.map(TIMING_POINT_ORDER))
    if rows.empty:
        return []

    finish_row = rows[rows["timing_point"] == "FINISH"].iloc[0]
    if finish_row["is_missing"] or pd.isna(finish_row["timestamp"]):
        return []

    flags = []
    mid_points = TIMING_POINTS[1:-1]
    consecutive_missing_streak = 0
    first_missing_in_streak = None

    for p in mid_points:
        r = rows[rows["timing_point"] == p].iloc[0]
        not_actual = bool(r["is_missing"]) or bool(r["is_interpolated"])
        if not_actual:
            if consecutive_missing_streak == 0:
                first_missing_in_streak = p
            consecutive_missing_streak += 1
        else:
            if consecutive_missing_streak >= 2:
                last_missing = TIMING_POINTS[TIMING_POINT_ORDER[p] - 1]
                flags.append((
                    "SHORTCUT",
                    f"计时点 {first_missing_in_streak} 至 {last_missing} 共 {consecutive_missing_streak} 个点缺失，疑似抄近道",
                ))
            consecutive_missing_streak = 0
            first_missing_in_streak = None

    if consecutive_missing_streak >= 2:
        last_missing = TIMING_POINTS[-2]
        flags.append((
            "SHORTCUT",
            f"计时点 {first_missing_in_streak} 至 {last_missing} 共 {consecutive_missing_streak} 个点缺失，疑似抄近道",
        ))

    return flags


def detect_pace_anomaly(
    seg_df: pd.DataFrame,
    bib: str,
) -> List[Tuple[str, str]]:
    """检测分段配速异常：某段速度远超本人其它分段、或越过物理上限."""
    rows = seg_df[seg_df["bib"] == bib].sort_values("timing_point", key=lambda s: s.map(TIMING_POINT_ORDER))
    if rows.empty:
        return []

    flags: List[Tuple[str, str]] = []

    with_seg = rows[
        rows["segment_time_sec"].notna()
        & (~rows["is_interpolated"])
        & (~rows["is_missing"])
    ]

    if len(with_seg) < 2:
        return flags

    paces_kmh: List[float] = []
    point_pace_map: List[Tuple[str, float]] = []

    for _, r in with_seg.iterrows():
        seg_sec = r["segment_time_sec"]
        seg_km = 0.0
        idx = TIMING_POINT_ORDER[r["timing_point"]]
        if idx > 0:
            prev_p = TIMING_POINTS[idx - 1]
            seg_km = TIMING_POINT_DISTANCES[r["timing_point"]] - TIMING_POINT_DISTANCES[prev_p]
        if seg_sec > 0 and seg_km > 0:
            kmh = seg_km / (seg_sec / 3600.0)
            paces_kmh.append(kmh)
            point_pace_map.append((r["timing_point"], kmh))

    if not paces_kmh:
        return flags

    median = float(np.median(paces_kmh))
    if median <= 0:
        return flags

    for point, kmh in point_pace_map:
        if kmh > MAX_PACE_KMH:
            flags.append((
                "PACE_IMPOSSIBLE",
                f"{point} 段速度 {kmh:.2f} km/h，超过人类合理上限 {MAX_PACE_KMH} km/h",
            ))
        ratio = kmh / median
        if ratio > 1.0 / MIN_SEGMENT_RATIO:
            flags.append((
                "PACE_ANOMALY",
                f"{point} 段速度 {kmh:.2f} km/h 为本人中位 {median:.2f} km/h 的 {ratio:.2f} 倍，疑似异常",
            ))

    return flags


def detect_false_start(
    result_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    bib: str,
) -> List[Tuple[str, str]]:
    """检测抢跑：净成绩比本波次理论最快还快."""
    flags: List[Tuple[str, str]] = []

    rr = result_df[result_df["bib"] == bib]
    if rr.empty:
        return flags
    rr = rr.iloc[0]
    if rr["status"] != "FINISHED" or pd.isna(rr["net_time_sec"]):
        return flags

    runner = roster_df[roster_df["bib"] == bib]
    if runner.empty:
        return flags
    runner = runner.iloc[0]
    wave_id = runner.get("wave_id", 0) or 0

    same_wave = roster_df[
        (roster_df["wave_id"] == wave_id)
        & (~roster_df["is_elite"])
        & (~roster_df["is_wheelchair"])
    ]
    if same_wave.empty:
        return flags

    est_times = same_wave["estimated_finish_hours"].values
    est_times = est_times[~np.isnan(est_times)]
    if len(est_times) < 3:
        return flags

    fastest_est_h = float(np.percentile(est_times, 5))
    fastest_est_sec = fastest_est_h * 3600.0

    start_row = seg_df[(seg_df["bib"] == bib) & (seg_df["timing_point"] == "START")]
    if start_row.empty:
        return flags
    start_row = start_row.iloc[0]
    start_ts = start_row["timestamp"]
    if pd.isna(start_ts):
        return flags

    wave_gun_offset_sec = int(wave_id) * WAVE_INTERVAL_MINUTES * 60
    earliest_expected_start = GUN_START_TIME + timedelta(seconds=int(wave_gun_offset_sec))
    start_delta = (start_ts - earliest_expected_start).total_seconds()

    net_sec = rr["net_time_sec"]
    suspicious_margin = 300.0
    if start_delta < -1 and (fastest_est_sec - net_sec) > suspicious_margin:
        flags.append((
            "FALSE_START",
            f"过起点时间 {start_ts} 早于本波理论最早出发 {earliest_expected_start} {-start_delta:.0f} 秒，净成绩 {net_sec:.0f}s 快于波次前5%预估 {fastest_est_sec:.0f}s",
        ))

    return flags


def detect_all_cheating(
    result_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    roster_df: pd.DataFrame,
) -> pd.DataFrame:
    """对所有选手做完整的作弊检测，返回可疑名单."""
    records: List[dict] = []
    bibs = sorted(set(result_df["bib"].tolist()))

    for bib in bibs:
        checks = []
        checks.extend(detect_shortcut(seg_df, bib))
        checks.extend(detect_pace_anomaly(seg_df, bib))
        checks.extend(detect_false_start(result_df, roster_df, seg_df, bib))

        for code, detail in checks:
            records.append({
                "bib": bib,
                "suspicion_code": code,
                "suspicion_detail": detail,
            })

    return pd.DataFrame(records)
