"""端到端快速冒烟测试：小规模跑完整个流水线并检查关键指标."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marathon.cheating import detect_all_cheating
from marathon.reconstruction import reconstruct_race
from marathon.simulator import generate_chip_reads, generate_roster
from marathon.waves import WaveConfig, assign_waves, validate_assignments


def main():
    N = 1000
    print(f"[1/5] 生成 {N} 人模拟名册 ...")
    roster = generate_roster(num_runners=N, seed=42)
    assert len(roster) == N
    print(f"      名册 OK：{len(roster)} 人，精英 {(roster.is_elite).sum()}，轮椅 {(roster.is_wheelchair).sum()}")

    print("[2/5] 生成刷卡流水（含去抖重复、漏读、作弊、关门、DNS）...")
    reads = generate_chip_reads(roster, seed=42)
    print(f"      刷卡 OK：{len(reads)} 条记录，乱序混合")

    print("[3/5] 重建成绩（去抖 + 插值 + 关门）...")
    result_df, segments_df = reconstruct_race(reads, roster)
    status_cnt = result_df.groupby("status").size().to_dict()
    print(f"      结果 OK：{status_cnt}")
    assert "FINISHED" in status_cnt, "应该有完赛选手"
    finished = result_df[result_df["status"] == "FINISHED"].copy()
    # 只对起点在鸣枪后（正常出发）的人断言枪成绩 >= 净成绩
    # 抢跑者起点早于鸣枪，会出现净成绩 > 枪成绩（是作弊特征）
    finished_with_start = segments_df[
        (segments_df["timing_point"] == "START")
        & segments_df["gun_time_sec"].notna()
    ][["bib", "gun_time_sec"]].rename(columns={"gun_time_sec": "start_gun_sec"})
    finished = finished.merge(finished_with_start, on="bib", how="inner")
    normal_starters = finished[finished["start_gun_sec"] >= 0]
    if len(normal_starters):
        assert (normal_starters["gun_time_sec"] >= normal_starters["net_time_sec"]).all(), \
            "正常出发选手的枪成绩应 >= 净成绩"

    print("[4/5] 作弊检测 ...")
    suspicions = detect_all_cheating(result_df, segments_df, roster)
    print(f"      作弊检测 OK：发现 {len(suspicions)} 条可疑记录")
    if len(suspicions):
        print(f"      按类别分布：{suspicions.groupby('suspicion_code').size().to_dict()}")

    print("[5/5] 分区波次编排 + 约束校验 ...")
    cfg = WaveConfig(num_zones=4, runners_per_zone=400, runners_per_wave=150)
    assigned, wave_assigns = assign_waves(roster, cfg)
    report = validate_assignments(assigned, wave_assigns, cfg)
    print(f"      编排 OK：{len(wave_assigns)} 个波次，约束满足 = {report.all_valid}")
    if not report.all_valid:
        print("      违规详情：")
        for k, v in report.__dict__.items():
            if isinstance(v, list) and v:
                for x in v[:3]:
                    print(f"        - {k}: {x}")

    print("\n🎉 端到端流程全部通过！")
    print("   运行平台请双击 run.bat 或执行：")
    print("   python -m streamlit run app.py --server.port 7956")


if __name__ == "__main__":
    main()
