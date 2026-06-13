"""模拟数据生成器：选手名册 + 刷卡流水，含异常样本."""

import random
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import (
    CUTOFF_TIMES,
    GUN_START_TIME,
    TIMING_POINT_DISTANCES,
    TIMING_POINTS,
    WAVE_INTERVAL_MINUTES,
)
from .models import ChipRead, Runner

_FAMILY_NAMES = [
    "王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗",
]
_GIVEN_NAMES = [
    "伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
    "勇", "艳", "杰", "娟", "涛", "明", "超", "秀英", "霞", "平",
    "刚", "桂英", "文", "辉", "玲", "健", "峰", "华", "飞", "鑫",
]


def _random_name(rng: random.Random) -> str:
    return rng.choice(_FAMILY_NAMES) + rng.choice(_GIVEN_NAMES)


def generate_roster(
    num_runners: int = 8000,
    seed: int = 42,
    elite_count: int = 30,
    wheelchair_count: int = 20,
) -> pd.DataFrame:
    """生成选手名册DataFrame."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    records = []
    bib_counter = 1

    elite_bibs = list(range(1, elite_count + 1))
    wheelchair_bibs = list(range(elite_count + 1, elite_count + wheelchair_count + 1))
    bib_counter = elite_count + wheelchair_count + 1

    for i in range(elite_count):
        bib = f"A{elite_bibs[i]:05d}"
        records.append({
            "bib": bib,
            "name": _random_name(rng),
            "gender": rng.choices(["M", "F"], weights=[0.7, 0.3])[0],
            "age": int(np_rng.integers(22, 40)),
            "estimated_finish_hours": float(np_rng.uniform(2.05, 2.5)),
            "is_elite": True,
            "is_wheelchair": False,
            "start_zone": "ZONE_A",
            "wave_id": 0,
        })

    for i in range(wheelchair_count):
        bib = f"W{wheelchair_bibs[i]:05d}"
        records.append({
            "bib": bib,
            "name": _random_name(rng),
            "gender": rng.choices(["M", "F"], weights=[0.6, 0.4])[0],
            "age": int(np_rng.integers(25, 60)),
            "estimated_finish_hours": float(np_rng.uniform(1.3, 2.2)),
            "is_elite": False,
            "is_wheelchair": True,
            "start_zone": "ZONE_A",
            "wave_id": 0,
        })

    remaining = num_runners - elite_count - wheelchair_count
    est_mean, est_std = 4.2, 0.8
    est_times = np_rng.normal(est_mean, est_std, remaining)
    est_times = np.clip(est_times, 2.5, 6.0)

    genders = rng.choices(["M", "F"], weights=[0.62, 0.38], k=remaining)
    ages_male = np_rng.normal(38, 10, remaining)
    ages_female = np_rng.normal(35, 9, remaining)
    ages = np.where(np.array(genders) == "M", ages_male, ages_female)
    ages = np.clip(ages, 18, 75).astype(int)

    zone_names = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E"]
    est_sorted_idx = np.argsort(est_times)
    zone_size = remaining // len(zone_names)

    for local_i in range(remaining):
        global_rank = int(est_sorted_idx[local_i])
        zone_idx = min(local_i // max(zone_size, 1), len(zone_names) - 1)
        zone = zone_names[zone_idx]
        wave_id = zone_idx * 2 + (local_i % 2)

        bib = f"B{bib_counter:05d}"
        bib_counter += 1
        records.append({
            "bib": bib,
            "name": _random_name(rng),
            "gender": genders[global_rank],
            "age": int(ages[global_rank]),
            "estimated_finish_hours": float(est_times[global_rank]),
            "is_elite": False,
            "is_wheelchair": False,
            "start_zone": zone,
            "wave_id": wave_id,
        })

    return pd.DataFrame(records)


def _pace_from_est(est_hours: float, is_wheelchair: bool) -> float:
    total_km = 42.195
    return est_hours / total_km


def _wave_start_time(wave_id: int) -> datetime:
    return GUN_START_TIME + timedelta(minutes=wave_id * WAVE_INTERVAL_MINUTES)


def generate_chip_reads(
    roster_df: pd.DataFrame,
    seed: int = 42,
    duplicate_rate: float = 0.08,
    missing_rate: float = 0.04,
    cheater_count: int = 25,
    cutoff_dnf_count: int = 120,
    dns_count: int = 80,
) -> pd.DataFrame:
    """生成刷卡流水DataFrame，含去抖重复、漏读、作弊、关门、DNS."""
    rng = random.Random(seed + 1000)
    np_rng = np.random.default_rng(seed + 1001)

    reads = []

    runner_dicts = roster_df.to_dict("records")
    rng.shuffle(runner_dicts)

    dns_bibs = set()
    non_special = [r for r in runner_dicts if not r["is_elite"] and not r["is_wheelchair"]]
    for r in non_special[:dns_count]:
        dns_bibs.add(r["bib"])

    cheater_targets = non_special[dns_count:dns_count + cheater_count]
    cheater_bibs = {r["bib"] for r in cheater_targets}

    dnf_targets = non_special[dns_count + cheater_count:dns_count + cheater_count + cutoff_dnf_count]
    dnf_bibs = {r["bib"] for r in dnf_targets}

    missing_idx_map = {}
    for r in runner_dicts:
        if r["bib"] in dns_bibs:
            continue
        n_mid = len(TIMING_POINTS) - 2
        missing_idx_map[r["bib"]] = []
        for j in range(1, len(TIMING_POINTS) - 1):
            if rng.random() < missing_rate:
                missing_idx_map[r["bib"]].append(j)

    cutoff_point_idx = {}
    for r in dnf_targets:
        cut_idx = rng.randint(1, len(TIMING_POINTS) - 2)
        cutoff_point_idx[r["bib"]] = cut_idx

    for runner in runner_dicts:
        bib = runner["bib"]
        if bib in dns_bibs:
            continue

        wave_id = runner.get("wave_id", 0) or 0
        actual_start = _wave_start_time(wave_id) + timedelta(
            seconds=float(np_rng.exponential(25))
        )

        est_h = runner["estimated_finish_hours"]
        base_pace = _pace_from_est(est_h, runner["is_wheelchair"])
        pace_jitter = np_rng.normal(1.0, 0.05)

        is_cheater_shortcut = bib in cheater_bibs and rng.random() < 0.5
        is_cheater_fast = bib in cheater_bibs and not is_cheater_shortcut
        is_cheater_false_start = bib in cheater_bibs and rng.random() < 0.3

        runner_times = {}

        last_t = actual_start
        runner_times["START"] = last_t

        if is_cheater_false_start:
            runner_times["START"] = actual_start - timedelta(
                seconds=float(np_rng.uniform(30, 120))
            )
            last_t = runner_times["START"]

        skip_idx = -1
        if is_cheater_shortcut:
            skip_idx = rng.randint(1, len(TIMING_POINTS) - 2)

        for p_idx in range(1, len(TIMING_POINTS)):
            point = TIMING_POINTS[p_idx]
            prev_point = TIMING_POINTS[p_idx - 1]

            if bib in dnf_bibs and p_idx > cutoff_point_idx.get(bib, 999):
                break

            seg_km = TIMING_POINT_DISTANCES[point] - TIMING_POINT_DISTANCES[prev_point]
            seg_pace = base_pace * pace_jitter * float(np_rng.normal(1.0, 0.06))

            if is_cheater_fast and p_idx == len(TIMING_POINTS) - 1:
                seg_pace *= 0.4
            if p_idx == skip_idx:
                missing_idx_map[bib].append(p_idx)
                continue

            seg_seconds = seg_km * seg_pace * 3600
            last_t = last_t + timedelta(seconds=float(seg_seconds))

            if bib in dnf_bibs and p_idx == cutoff_point_idx.get(bib, 999):
                cutoff_deadline = GUN_START_TIME + CUTOFF_TIMES[point]
                if last_t < cutoff_deadline:
                    last_t = cutoff_deadline + timedelta(seconds=rng.randint(1, 300))

            runner_times[point] = last_t

        for p_idx, point in enumerate(TIMING_POINTS):
            if point not in runner_times:
                continue
            if p_idx in missing_idx_map.get(bib, []):
                continue
            ts = runner_times[point]

            n_dup = 1
            if rng.random() < duplicate_rate:
                n_dup = rng.randint(2, 4)

            for d_i in range(n_dup):
                offset = timedelta(
                    milliseconds=int(np_rng.uniform(0, 3500))
                ) if d_i > 0 else timedelta(0)
                reads.append({
                    "bib": bib,
                    "timing_point": point,
                    "timestamp": ts + offset,
                })

    df = pd.DataFrame(reads)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sample(frac=1.0, random_state=seed + 5000).reset_index(drop=True)
    return df
