"""核心逻辑单元测试：手工构造可验算数据."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from marathon.config import CUTOFF_TIMES, GUN_START_TIME, TIMING_POINT_DISTANCES, TIMING_POINTS
from marathon.reconstruction import (
    apply_cutoff,
    build_runner_times,
    dedup_reads,
    interpolate_missing,
    reconstruct_race,
)
from marathon.cheating import (
    detect_all_cheating,
    detect_false_start,
    detect_pace_anomaly,
    detect_shortcut,
)
from marathon.waves import assign_waves, validate_assignments, WaveConfig


T0 = GUN_START_TIME


def _ts(offset_seconds: float) -> datetime:
    return T0 + timedelta(seconds=offset_seconds)


def _df_from_tuples(tuples):
    return pd.DataFrame(
        [(bib, pt, _ts(sec)) for (bib, pt, sec) in tuples],
        columns=["bib", "timing_point", "timestamp"],
    )


def _make_roster(bib_list, **defaults):
    rows = []
    for i, bib in enumerate(bib_list):
        rows.append({
            "bib": bib,
            "name": f"选手{i}",
            "gender": "M",
            "age": 30,
            "estimated_finish_hours": 4.0,
            "is_elite": False,
            "is_wheelchair": False,
            "start_zone": "ZONE_A",
            "wave_id": 0,
        })
    df = pd.DataFrame(rows)
    for k, v in defaults.items():
        df[k] = v
    return df


# =========================================================
# 1. 去抖测试
# =========================================================
class TestDedup:
    def test_no_duplicates_unchanged(self):
        df = _df_from_tuples([
            ("B001", "START", 30),
            ("B001", "KM5", 1800),
            ("B002", "START", 45),
        ])
        out = dedup_reads(df)
        assert len(out) == 3

    def test_bounce_within_window_removed(self):
        df = _df_from_tuples([
            ("B001", "START", 30),
            ("B001", "START", 31),
            ("B001", "START", 33),
            ("B001", "KM5", 1800),
        ])
        out = dedup_reads(df)
        start_rows = out[(out["bib"] == "B001") & (out["timing_point"] == "START")]
        assert len(start_rows) == 1
        assert start_rows.iloc[0]["timestamp"] == _ts(30)

    def test_outside_window_kept(self):
        df = _df_from_tuples([
            ("B001", "START", 30),
            ("B001", "START", 30 + 10),
        ])
        out = dedup_reads(df)
        assert len(out[(out["bib"] == "B001") & (out["timing_point"] == "START")]) == 2


# =========================================================
# 2. 漏读插值测试
# =========================================================
class TestInterpolation:
    def test_perfect_run_all_actual(self):
        rt = {p: _ts(1800 * i) for i, p in enumerate(TIMING_POINTS)}
        filled, status = interpolate_missing(rt)
        for p in TIMING_POINTS:
            assert status[p] == "actual"
            assert filled[p] is not None

    def test_single_mid_missing_interpolated(self):
        rt = {}
        offsets = [60, 1860, None, 7500, 10500, 15120]
        for p, off in zip(TIMING_POINTS, offsets):
            if off is not None:
                rt[p] = _ts(off)

        filled, status = interpolate_missing(rt)
        assert status["KM10"] == "interpolated"
        t5 = (filled["KM5"] - T0).total_seconds()
        t10 = (filled["KM10"] - T0).total_seconds()
        t_half = (filled["HALF"] - T0).total_seconds()
        # 线性插值: (10-5)/(21.0975-5) 比例
        ratio = (10.0 - 5.0) / (21.0975 - 5.0)
        expected = t5 + (t_half - t5) * ratio
        assert abs(t10 - expected) < 0.5

    def test_no_start_all_missing_before(self):
        rt = {
            "KM5": _ts(2000),
            "KM10": _ts(4000),
            "FINISH": _ts(15000),
        }
        filled, status = interpolate_missing(rt)
        assert status["START"] == "missing"
        assert filled["START"] is None


# =========================================================
# 3. 关门与DNF
# =========================================================
class TestCutoff:
    def test_finish_before_cutoff_no_change(self):
        times = {p: _ts(1800 * i + 60) for i, p in enumerate(TIMING_POINTS)}
        st = {p: "actual" for p in TIMING_POINTS}
        trimmed, new_st, cutoff_p = apply_cutoff(times, st)
        assert cutoff_p is None
        assert trimmed["FINISH"] is not None

    def test_km10_past_cutoff_truncated(self):
        t = {p: _ts(0) for p in TIMING_POINTS}
        st = {p: "actual" for p in TIMING_POINTS}
        km10_deadline = (GUN_START_TIME + CUTOFF_TIMES["KM10"] - T0).total_seconds()
        t["KM10"] = _ts(km10_deadline + 500)
        t["KM5"] = _ts(500)
        t["START"] = _ts(30)
        t["HALF"] = _ts(km10_deadline + 3000)
        t["KM30"] = _ts(km10_deadline + 6000)
        t["FINISH"] = _ts(km10_deadline + 10000)
        trimmed, new_st, cutoff_p = apply_cutoff(t, st)
        assert cutoff_p == "KM10"
        assert trimmed["HALF"] is None
        assert trimmed["KM30"] is None
        assert trimmed["FINISH"] is None
        assert new_st["KM10"].startswith("cutoff")


# =========================================================
# 4. 枪声成绩 vs 净成绩 & 分段配速
# =========================================================
class TestScoreReconstruction:
    def test_gun_vs_net(self):
        # 起点过毯晚60秒，到终点总共用了15000秒（枪）
        reads = _df_from_tuples([
            ("B001", "START", 60),
            ("B001", "KM5", 2100),
            ("B001", "KM10", 4140),
            ("B001", "HALF", 8340),
            ("B001", "KM30", 11700),
            ("B001", "FINISH", 16500),
        ])
        roster = _make_roster(["B001"])
        result_df, seg_df = reconstruct_race(reads, roster)
        r = result_df.iloc[0]
        assert r["status"] == "FINISHED"
        # 枪成绩 = 16500 - 0 = 16500
        assert abs(r["gun_time_sec"] - 16500) < 0.5
        # 净成绩 = 16500 - 60 = 16440
        assert abs(r["net_time_sec"] - 16440) < 0.5

        # 分段KM5: 2100-60 = 2040s for 5km -> pace = (2040/60)/5 = 6.8 min/km
        km5 = seg_df[(seg_df["bib"] == "B001") & (seg_df["timing_point"] == "KM5")].iloc[0]
        assert abs(km5["segment_time_sec"] - 2040) < 0.5
        expected_pace = (2040 / 60.0) / 5.0
        assert abs(km5["segment_pace_min_per_km"] - expected_pace) < 0.001

    def test_dns_no_start(self):
        reads = _df_from_tuples([
            ("B001", "KM5", 2000),
        ])
        roster = _make_roster(["B001"])
        rdf, _ = reconstruct_race(reads, roster)
        r = rdf[rdf["bib"] == "B001"].iloc[0]
        assert r["status"] == "DNS"

    def test_dnf_no_finish(self):
        reads = _df_from_tuples([
            ("B001", "START", 60),
            ("B001", "KM5", 2100),
        ])
        roster = _make_roster(["B001"])
        rdf, _ = reconstruct_race(reads, roster)
        r = rdf[rdf["bib"] == "B001"].iloc[0]
        assert r["status"] == "DNF"
        assert r["cutoff_point"] is None


# =========================================================
# 5. 作弊检测
# =========================================================
class TestCheating:
    def test_shortcut_two_consecutive_missing(self):
        reads = _df_from_tuples([
            ("B001", "START", 60),
            ("B001", "KM5", 2000),
            # KM10 和 HALF 缺失
            ("B001", "KM30", 11000),
            ("B001", "FINISH", 16000),
        ])
        roster = _make_roster(["B001"])
        rdf, sdf = reconstruct_race(reads, roster)
        flags = detect_shortcut(sdf, "B001")
        assert any(code == "SHORTCUT" for code, _ in flags)

    def test_pace_impossible_triggers(self):
        # KM5 段 5公里只用了5分钟=300秒 (60km/h)
        reads = _df_from_tuples([
            ("B001", "START", 60),
            ("B001", "KM5", 60 + 300),
            ("B001", "KM10", 60 + 300 + 2700),
            ("B001", "HALF", 60 + 300 + 2700 + 6000),
            ("B001", "KM30", 60 + 300 + 2700 + 6000 + 3200),
            ("B001", "FINISH", 60 + 300 + 2700 + 6000 + 3200 + 5400),
        ])
        roster = _make_roster(["B001"])
        rdf, sdf = reconstruct_race(reads, roster)
        flags = detect_pace_anomaly(sdf, "B001")
        codes = {c for c, _ in flags}
        assert "PACE_IMPOSSIBLE" in codes

    def test_deterministic_same_input_same_output(self):
        """同一份输入多跑几遍结果完全一致."""
        reads = _df_from_tuples([
            ("B001", "START", 60), ("B001", "START", 62),
            ("B001", "KM5", 2100),
            ("B001", "KM10", 4140),
            ("B001", "HALF", 8340),
            ("B001", "KM30", 11700),
            ("B001", "FINISH", 16500),
            ("B002", "START", 90),
            ("B002", "KM5", 2500),
            ("B002", "KM10", 5000),
            ("B002", "HALF", 9900),
            ("B002", "KM30", 14000),
            ("B002", "FINISH", 20000),
        ])
        roster = _make_roster(["B001", "B002"])
        results = []
        for _ in range(3):
            rdf, sdf = reconstruct_race(reads, roster)
            cdf = detect_all_cheating(rdf, sdf, roster)
            if len(cdf) == 0:
                cdf_sorted_lines = tuple()
            else:
                cdf_sorted_lines = tuple(
                    cdf.sort_values(["bib", "suspicion_code"]).to_csv(index=False).splitlines()
                )
            results.append((
                tuple(rdf.sort_values("bib").to_csv(index=False).splitlines()),
                tuple(sdf.sort_values(["bib", "timing_point"]).to_csv(index=False).splitlines()),
                cdf_sorted_lines,
            ))
        for i in range(1, len(results)):
            assert results[i] == results[0]


# =========================================================
# 6. 波次编排与约束
# =========================================================
class TestWaveAssignment:
    def test_elite_and_wheelchair_in_priority_wave(self):
        bibs = [f"B{i:03d}" for i in range(100)]
        roster = _make_roster(bibs)
        roster.loc[0:4, "is_elite"] = True
        roster.loc[5:9, "is_wheelchair"] = True
        roster["estimated_finish_hours"] = np.linspace(2.2, 6.0, len(roster))

        cfg = WaveConfig(num_zones=3, runners_per_zone=50, runners_per_wave=30)
        assigned, waves = assign_waves(roster, cfg)
        report = validate_assignments(assigned, waves, cfg)
        assert report.all_valid, f"约束未满足: {report}"

        elite_bibs = set(assigned[assigned["is_elite"]]["bib"])
        wheelchair_bibs = set(assigned[assigned["is_wheelchair"]]["bib"])
        all_priority_bibs = set()
        for wa in waves[:2]:
            all_priority_bibs.update(wa.bibs)
        assert elite_bibs.issubset(all_priority_bibs)
        assert wheelchair_bibs.issubset(all_priority_bibs)

    def test_capacity_constraints_respected(self):
        bibs = [f"B{i:03d}" for i in range(500)]
        roster = _make_roster(bibs)
        roster["estimated_finish_hours"] = np.linspace(3.0, 5.5, len(roster))

        cfg = WaveConfig(num_zones=3, runners_per_zone=200, runners_per_wave=80)
        assigned, waves = assign_waves(roster, cfg)
        report = validate_assignments(assigned, waves, cfg)
        assert report.all_valid, f"容量违规: {report}"

    def test_wave_interval_monotonic_start_times(self):
        bibs = [f"B{i:03d}" for i in range(300)]
        roster = _make_roster(bibs)
        roster["estimated_finish_hours"] = np.linspace(3.0, 5.5, len(roster))

        cfg = WaveConfig(num_zones=2, runners_per_zone=200, runners_per_wave=50)
        assigned, waves = assign_waves(roster, cfg)
        sorted_waves = sorted(waves, key=lambda w: w.wave_id)
        for i in range(1, len(sorted_waves)):
            delta = (sorted_waves[i].start_time - sorted_waves[i - 1].start_time).total_seconds()
            assert delta > 0
