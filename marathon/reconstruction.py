"""成绩重建核心逻辑：去抖、漏读插值、枪声/净成绩、分段配速."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BOUNCE_WINDOW_SECONDS,
    CUTOFF_TIMES,
    GUN_START_TIME,
    TIMING_POINT_DISTANCES,
    TIMING_POINT_ORDER,
    TIMING_POINTS,
)


def dedup_reads(reads_df: pd.DataFrame) -> pd.DataFrame:
    """对同一选手同一点的连刷记录去抖，只保留最早的一条."""
    if reads_df.empty:
        return reads_df.copy()

    df = reads_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["bib", "timing_point", "timestamp"]).reset_index(drop=True)

    keep_mask = np.ones(len(df), dtype=bool)
    last_key: Optional[Tuple[str, str]] = None
    last_ts: Optional[datetime] = None
    window = timedelta(seconds=BOUNCE_WINDOW_SECONDS)

    for i in range(len(df)):
        row = df.iloc[i]
        key = (row["bib"], row["timing_point"])
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        if key == last_key and last_ts is not None and (ts - last_ts) <= window:
            keep_mask[i] = False
        else:
            last_key = key
            last_ts = ts

    return df[keep_mask].reset_index(drop=True)


def build_runner_times(deduped_df: pd.DataFrame) -> Dict[str, Dict[str, datetime]]:
    """把去抖后的DataFrame转成 {bib: {point: datetime}} 结构."""
    result: Dict[str, Dict[str, datetime]] = {}
    if deduped_df.empty:
        return result

    for _, row in deduped_df.iterrows():
        bib = row["bib"]
        point = row["timing_point"]
        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if bib not in result:
            result[bib] = {}
        result[bib][point] = ts
    return result


def interpolate_missing(
    runner_times: Dict[str, datetime]
) -> Tuple[Dict[str, Optional[datetime]], Dict[str, str]]:
    """对缺失的中间计时点做插值估计；返回 {point: ts} 与 {point: status}.

    status: 'actual' | 'interpolated' | 'missing'
    """
    filled: Dict[str, Optional[datetime]] = {}
    status: Dict[str, str] = {}

    actual_points = sorted(
        [(TIMING_POINT_ORDER[p], p, t) for p, t in runner_times.items()],
        key=lambda x: x[0],
    )

    for _, p, t in actual_points:
        filled[p] = t
        status[p] = "actual"

    if not actual_points:
        for p in TIMING_POINTS:
            filled[p] = None
            status[p] = "missing"
        return filled, status

    actual_orders = [o for o, _, _ in actual_points]

    for i in range(len(actual_points) - 1):
        o1, p1, t1 = actual_points[i]
        o2, p2, t2 = actual_points[i + 1]
        if o2 - o1 > 1:
            total_d = TIMING_POINT_DISTANCES[p2] - TIMING_POINT_DISTANCES[p1]
            total_t = (t2 - t1).total_seconds()
            for mid_o in range(o1 + 1, o2):
                mid_p = TIMING_POINTS[mid_o]
                if mid_p not in filled:
                    d_ratio = (TIMING_POINT_DISTANCES[mid_p] - TIMING_POINT_DISTANCES[p1]) / max(total_d, 1e-9)
                    mid_ts = t1 + timedelta(seconds=total_t * d_ratio)
                    filled[mid_p] = mid_ts
                    status[mid_p] = "interpolated"

    first_o = actual_orders[0]
    for o in range(0, first_o):
        p = TIMING_POINTS[o]
        if p not in filled:
            filled[p] = None
            status[p] = "missing"

    last_o = actual_orders[-1]
    for o in range(last_o + 1, len(TIMING_POINTS)):
        p = TIMING_POINTS[o]
        if p not in filled:
            filled[p] = None
            status[p] = "missing"

    return filled, status


def apply_cutoff(
    filled_times: Dict[str, Optional[datetime]],
    status: Dict[str, str],
) -> Tuple[Dict[str, Optional[datetime]], Dict[str, str], Optional[str]]:
    """按关门时间裁剪；返回 (裁剪后的时间, 状态更新, DNF触发点或None)."""
    trimmed = dict(filled_times)
    new_status = dict(status)
    cutoff_point: Optional[str] = None

    first_past = None
    for p in TIMING_POINTS:
        ts = trimmed.get(p)
        if ts is None:
            continue
        deadline = GUN_START_TIME + CUTOFF_TIMES[p]
        if ts > deadline:
            first_past = p
            break

    if first_past is not None:
        cutoff_point = first_past
        cutoff_order = TIMING_POINT_ORDER[first_past]
        for p in TIMING_POINTS:
            if TIMING_POINT_ORDER[p] > cutoff_order:
                trimmed[p] = None
                new_status[p] = "missing"
        if new_status[first_past] == "actual":
            new_status[first_past] = "cutoff_actual"
        elif new_status[first_past] == "interpolated":
            new_status[first_past] = "cutoff_interpolated"

    return trimmed, new_status, cutoff_point


def compute_segment_table(
    bib: str,
    trimmed_times: Dict[str, Optional[datetime]],
    status: Dict[str, str],
) -> pd.DataFrame:
    """计算分段表（枪声、净时间、分段用时、配速）."""
    rows: List[Dict] = []
    start_ts = trimmed_times.get("START")
    finish_ts = trimmed_times.get("FINISH")
    prev_ts: Optional[datetime] = None
    prev_dist: float = 0.0

    for point in TIMING_POINTS:
        ts = trimmed_times.get(point)
        dist = TIMING_POINT_DISTANCES[point]
        st = status.get(point, "missing")
        is_interp = st == "interpolated" or st == "cutoff_interpolated"
        is_missing = st == "missing" and ts is None
        is_cutoff = st.startswith("cutoff")

        gun_sec: Optional[float] = None
        net_sec: Optional[float] = None
        seg_sec: Optional[float] = None
        seg_pace: Optional[float] = None

        if ts is not None:
            gun_sec = (ts - GUN_START_TIME).total_seconds()
            if start_ts is not None:
                net_sec = (ts - start_ts).total_seconds()
            if prev_ts is not None and ts >= prev_ts:
                seg_sec = (ts - prev_ts).total_seconds()
                seg_km = dist - prev_dist
                if seg_sec > 0 and seg_km > 0:
                    seg_pace = seg_sec / 60.0 / seg_km

        rows.append({
            "bib": bib,
            "timing_point": point,
            "distance": dist,
            "timestamp": ts,
            "gun_time_sec": gun_sec,
            "net_time_sec": net_sec,
            "segment_time_sec": seg_sec,
            "segment_pace_min_per_km": seg_pace,
            "is_interpolated": is_interp,
            "is_missing": is_missing,
            "is_cutoff": is_cutoff,
            "status": st,
        })

        if ts is not None:
            prev_ts = ts
            prev_dist = dist

    return pd.DataFrame(rows)


def build_result_row(
    bib: str,
    seg_df: pd.DataFrame,
    cutoff_point: Optional[str],
) -> Dict:
    """把分段表汇总成一条选手结果记录."""
    start_row = seg_df[seg_df["timing_point"] == "START"].iloc[0]
    finish_row = seg_df[seg_df["timing_point"] == "FINISH"].iloc[0]

    start_ts = start_row["timestamp"]
    finish_ts = finish_row["timestamp"]

    if pd.isna(start_ts) or start_ts is None:
        status = "DNS"
        gun_time_sec = None
        net_time_sec = None
        avg_pace = None
    elif pd.isna(finish_ts) or finish_ts is None:
        if cutoff_point is not None:
            status = "DNF_CUTOFF"
        else:
            status = "DNF"
        gun_time_sec = None
        net_time_sec = None
        avg_pace = None
    else:
        status = "FINISHED"
        gun_time_sec = (finish_ts - GUN_START_TIME).total_seconds()
        net_time_sec = (finish_ts - start_ts).total_seconds()
        total_km = TIMING_POINT_DISTANCES["FINISH"]
        if net_time_sec > 0:
            avg_pace = (net_time_sec / 60.0) / total_km
        else:
            avg_pace = None

    return {
        "bib": bib,
        "status": status,
        "gun_time_sec": gun_time_sec,
        "net_time_sec": net_time_sec,
        "average_pace_min_per_km": avg_pace,
        "cutoff_point": cutoff_point,
    }


def reconstruct_race(
    reads_df: pd.DataFrame,
    roster_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """主入口：重建比赛成绩，返回 (结果表, 分段明细表)."""
    deduped = dedup_reads(reads_df)
    runner_times = build_runner_times(deduped)

    all_results: List[Dict] = []
    all_segments: List[pd.DataFrame] = []

    all_bibs = set(roster_df["bib"].tolist()) | set(runner_times.keys())

    for bib in sorted(all_bibs):
        rt = runner_times.get(bib, {})
        filled, status = interpolate_missing(rt)
        trimmed, new_status, cutoff_p = apply_cutoff(filled, status)
        seg_df = compute_segment_table(bib, trimmed, new_status)
        result = build_result_row(bib, seg_df, cutoff_p)
        all_results.append(result)
        all_segments.append(seg_df)

    result_df = pd.DataFrame(all_results)
    if all_segments:
        ref_cols = [
            "bib", "timing_point", "distance", "timestamp",
            "gun_time_sec", "net_time_sec", "segment_time_sec",
            "segment_pace_min_per_km", "is_interpolated", "is_missing",
            "is_cutoff", "status",
        ]
        normalized = [s[ref_cols] for s in all_segments]
        for s in normalized:
            s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            segments_df = pd.concat(normalized, ignore_index=True)
    else:
        segments_df = pd.DataFrame(columns=[
            "bib", "timing_point", "distance", "timestamp",
            "gun_time_sec", "net_time_sec", "segment_time_sec",
            "segment_pace_min_per_km", "is_interpolated", "is_missing",
            "is_cutoff", "status",
        ])
    return result_df, segments_df


def format_duration(sec: Optional[float]) -> str:
    """秒数格式化为 HH:MM:SS."""
    if sec is None or pd.isna(sec):
        return "--"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
