"""Streamlit 可视化看板：总榜、分组排名、各类图表、筛选联动."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marathon.config import (
    AGE_GROUPS,
    CUTOFF_TIMES,
    GUN_START_TIME,
    TIMING_POINT_DISTANCES,
    TIMING_POINT_ORDER,
    TIMING_POINTS,
)
from marathon.cheating import detect_all_cheating
from marathon.reconstruction import format_duration, reconstruct_race
from marathon.simulator import generate_chip_reads, generate_roster
from marathon.waves import (
    WaveConfig,
    assign_waves,
    validate_assignments,
    wave_assignment_table,
)


# =========================================================
# 数据层：session_state 缓存，全链路一次跑完
# =========================================================
@st.cache_data(show_spinner=False)
def build_dataset(num_runners: int, seed: int):
    roster = generate_roster(num_runners=num_runners, seed=seed)
    reads = generate_chip_reads(roster, seed=seed)
    return roster, reads


def run_pipeline(roster: pd.DataFrame, reads: pd.DataFrame):
    results, segments = reconstruct_race(reads, roster)
    merged = results.merge(roster, on="bib", how="left")
    suspicions = detect_all_cheating(results, segments, roster)
    return merged, segments, suspicions


def _get_age_group(age: int) -> str:
    for name, lo, hi in AGE_GROUPS:
        if lo <= age <= hi:
            return name
    return "65+"


def _rank_by(df: pd.DataFrame, group_cols=None, sort_col="net_time_sec"):
    g = df.copy()
    g = g[g["status"] == "FINISHED"].copy()
    if g.empty:
        return g
    group_cols = group_cols or []
    if group_cols:
        g["rank"] = g.groupby(group_cols)[sort_col].rank(method="min", na_option="bottom").astype(int)
    else:
        g = g.sort_values(sort_col, na_position="last").reset_index(drop=True)
        g["rank"] = g.index + 1
    return g.sort_values(group_cols + [sort_col] if group_cols else [sort_col])


# =========================================================
# 页面：总览 KPI
# =========================================================
def page_overview(roster, reads, merged, segments, suspicions):
    st.header("赛事总览")

    total = len(roster)
    finished = int((merged["status"] == "FINISHED").sum())
    dns = int((merged["status"] == "DNS").sum())
    dnf = int((merged["status"] == "DNF").sum())
    dnf_cutoff = int((merged["status"] == "DNF_CUTOFF").sum())
    suspicious_bibs = int(suspicions["bib"].nunique()) if len(suspicions) else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("总选手", f"{total:,}")
    c2.metric("完赛 (FINISHED)", f"{finished:,}")
    c3.metric("未出发 (DNS)", f"{dns:,}")
    c4.metric("未完赛 (DNF)", f"{dnf + dnf_cutoff:,}")
    c5.metric("完赛率", f"{finished / max(total, 1) * 100:.1f}%")
    c6.metric("可疑选手", f"{suspicious_bibs:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("完赛状态分布")
        status_cnt = merged.groupby("status").size().reset_index(name="count")
        fig = px.pie(status_cnt, names="status", values="count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("性别分布")
        gender_cnt = roster.groupby("gender").size().reset_index(name="count")
        fig2 = px.pie(gender_cnt, names="gender", values="count", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("完赛时间分布（净成绩）")
    finished_df = merged[merged["status"] == "FINISHED"].copy()
    if not finished_df.empty:
        fig3 = px.histogram(
            finished_df,
            x="net_time_sec",
            color="gender",
            marginal="box",
            nbins=50,
            labels={"net_time_sec": "净完赛时间 (秒)"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("各分段通过人数（实际刷卡）")
        actual_seg = segments[segments["status"] == "actual"]
        pcount = actual_seg.groupby("timing_point").size().reindex(TIMING_POINTS).fillna(0)
        pcount_df = pcount.reset_index()
        pcount_df.columns = ["timing_point", "count"]
        fig4 = px.bar(pcount_df, x="timing_point", y="count", text="count")
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("平均分段配速 (min/km)")
        finished_segs = segments[
            segments["segment_pace_min_per_km"].notna()
            & (~segments["is_interpolated"])
        ]
        pace_by_point = (
            finished_segs.groupby("timing_point")["segment_pace_min_per_km"].mean().reset_index()
        )
        fig5 = px.line(
            pace_by_point,
            x="timing_point",
            y="segment_pace_min_per_km",
            markers=True,
            labels={"segment_pace_min_per_km": "平均配速 (min/km)"},
        )
        st.plotly_chart(fig5, use_container_width=True)


# =========================================================
# 页面：榜单（带筛选、按性别/年龄组分组）
# =========================================================
def page_rankings(roster, merged, segments, suspicions):
    st.header("赛事榜单")

    with st.sidebar.expander("筛选条件", expanded=True):
        gender = st.multiselect("性别", ["M", "F"], default=["M", "F"])
        groups = [g[0] for g in AGE_GROUPS]
        age_sel = st.multiselect("年龄组", groups, default=groups)
        waves = sorted(merged["wave_id"].dropna().unique().tolist())
        wave_sel = st.multiselect("出发波次", waves, default=waves)
        status_sel = st.multiselect(
            "状态", ["FINISHED", "DNF", "DNF_CUTOFF", "DNS"], default=["FINISHED"]
        )

    merged2 = merged.copy()
    merged2["age_group"] = merged2["age"].apply(_get_age_group)

    mask = (
        merged2["gender"].isin(gender)
        & merged2["age_group"].isin(age_sel)
        & merged2["wave_id"].isin(wave_sel)
        & merged2["status"].isin(status_sel)
    )
    df = merged2[mask].copy()

    tab_overall, tab_gender, tab_age, tab_indiv = st.tabs(
        ["总榜", "性别榜", "年龄组榜", "选手详情"]
    )

    with tab_overall:
        st.subheader("总榜（净成绩排名）")
        ranked = _rank_by(df, sort_col="net_time_sec")
        if not ranked.empty:
            ranked = ranked.sort_values("net_time_sec").reset_index(drop=True)
            ranked["rank"] = ranked.index + 1
            show = ranked[
                [
                    "rank", "bib", "name", "gender", "age", "age_group",
                    "wave_id", "gun_time_sec", "net_time_sec",
                    "average_pace_min_per_km",
                ]
            ].copy()
            show["gun_time"] = show["gun_time_sec"].apply(format_duration)
            show["net_time"] = show["net_time_sec"].apply(format_duration)
            show["pace"] = show["average_pace_min_per_km"].apply(
                lambda v: f"{v:.2f} min/km" if pd.notna(v) else "--"
            )
            st.dataframe(
                show.drop(columns=["gun_time_sec", "net_time_sec", "average_pace_min_per_km"]),
                use_container_width=True,
                hide_index=True,
                height=520,
            )
        else:
            st.info("没有符合条件的完赛选手")

    with tab_gender:
        st.subheader("分性别榜")
        for g in gender:
            st.markdown(f"**{g} 组**")
            sub = df[df["gender"] == g].copy()
            rk = _rank_by(sub, sort_col="net_time_sec")
            if not rk.empty:
                rk = rk.sort_values("net_time_sec").reset_index(drop=True)
                rk["rank"] = rk.index + 1
                rk["gun_time"] = rk["gun_time_sec"].apply(format_duration)
                rk["net_time"] = rk["net_time_sec"].apply(format_duration)
                rk["pace"] = rk["average_pace_min_per_km"].apply(
                    lambda v: f"{v:.2f} min/km" if pd.notna(v) else "--"
                )
                st.dataframe(
                    rk[["rank", "bib", "name", "age", "gun_time", "net_time", "pace"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_age:
        st.subheader("分年龄组榜")
        for ag in age_sel:
            st.markdown(f"**{ag} 组**")
            sub = df[df["age_group"] == ag].copy()
            rk = _rank_by(sub, sort_col="net_time_sec")
            if not rk.empty:
                rk = rk.sort_values("net_time_sec").reset_index(drop=True)
                rk["rank"] = rk.index + 1
                rk["gun_time"] = rk["gun_time_sec"].apply(format_duration)
                rk["net_time"] = rk["net_time_sec"].apply(format_duration)
                st.dataframe(
                    rk[["rank", "bib", "name", "gender", "gun_time", "net_time"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_indiv:
        st.subheader("选手明细查询")
        bib_list = sorted(df["bib"].tolist())
        if bib_list:
            chosen = st.selectbox("选择选手号码布", bib_list)
            if chosen:
                indiv = df[df["bib"] == chosen].iloc[0]
                st.write(f"姓名: {indiv['name']}  |  性别: {indiv['gender']}  |  年龄: {indiv['age']}  |  波次: {indiv['wave_id']}")
                st.write(f"状态: **{indiv['status']}**  |  枪成绩: {format_duration(indiv['gun_time_sec'])}  |  净成绩: {format_duration(indiv['net_time_sec'])}")
                my_segs = segments[segments["bib"] == chosen].copy()
                my_segs = my_segs.sort_values("timing_point", key=lambda s: s.map(TIMING_POINT_ORDER))
                my_segs["到达时间"] = my_segs["timestamp"].astype(str)
                my_segs["分段用时"] = my_segs["segment_time_sec"].apply(format_duration)
                my_segs["分段配速(min/km)"] = my_segs["segment_pace_min_per_km"].apply(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "--"
                )
                disp = my_segs[["timing_point", "到达时间", "分段用时", "分段配速(min/km)", "is_interpolated", "is_missing", "is_cutoff"]]
                disp.columns = ["计时点", "到达时间", "分段用时", "分段配速", "插值", "缺失", "关门"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

                ind_sus = suspicions[suspicions["bib"] == chosen]
                if len(ind_sus):
                    st.markdown("**⚠️ 可疑记录**")
                    st.dataframe(ind_sus, use_container_width=True, hide_index=True)


# =========================================================
# 页面：分段流量 & 波次拥堵
# =========================================================
def page_flow(roster, merged, segments):
    st.header("分段流量与波次拥堵")

    st.subheader("各计时点到达时间分布（按波次）")
    segs2 = segments[segments["timestamp"].notna()].copy()
    segs2 = segs2.merge(
        merged[["bib", "wave_id", "gender"]], on="bib", how="left"
    )
    segs2["hour"] = segs2["timestamp"].dt.hour + segs2["timestamp"].dt.minute / 60.0

    if not segs2.empty:
        fig = px.box(
            segs2,
            x="timing_point",
            y="hour",
            color="wave_id",
            labels={"hour": "到达时间 (24h)", "timing_point": "计时点"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("起点拥堵：每波选手通过起点的耗时分布")
    start_seg = segs2[segs2["timing_point"] == "START"].copy()
    if not start_seg.empty:
        start_seg["wave_start"] = start_seg["wave_id"].apply(
            lambda w: GUN_START_TIME + timedelta(minutes=(w or 0) * 5)
        )
        start_seg["delay_sec"] = (
            start_seg["timestamp"] - start_seg["wave_start"]
        ).dt.total_seconds()
        fig2 = px.histogram(
            start_seg,
            x="delay_sec",
            color="wave_id",
            nbins=60,
            labels={"delay_sec": "过起点相对波次出发的延迟 (秒)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("配速分布（净平均配速）")
    finished = merged[merged["status"] == "FINISHED"].copy()
    if not finished.empty:
        fig3 = px.violin(
            finished,
            x="gender",
            y="average_pace_min_per_km",
            color="gender",
            box=True,
            labels={"average_pace_min_per_km": "平均配速 (min/km)"},
        )
        st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# 页面：可疑名单
# =========================================================
def page_suspicion(roster, merged, suspicions):
    st.header("可疑名单 & 作弊判定")

    if len(suspicions) == 0:
        st.success("没有检测到可疑记录 🎉")
        return

    merged2 = merged.copy()
    merged2["age_group"] = merged2["age"].apply(_get_age_group)
    df = suspicions.merge(merged2, on="bib", how="left")

    st.metric("可疑记录总数", len(df))

    col1, col2 = st.columns(2)
    with col1:
        by_code = df.groupby("suspicion_code").size().reset_index(name="count")
        fig = px.bar(by_code, x="suspicion_code", y="count", text="count", title="按可疑类别")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_gender = df.groupby("gender").size().reset_index(name="count")
        fig2 = px.pie(by_gender, names="gender", values="count", hole=0.4, title="按性别")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("详细可疑记录（保留选手成绩，未直接抹除）")
    display_cols = [
        "bib", "name", "gender", "age", "status", "suspicion_code", "suspicion_detail",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, hide_index=True, height=520)


# =========================================================
# 页面：分区波次编排
# =========================================================
def page_waves(roster):
    st.header("分区 & 波次编排")

    st.subheader("编排参数")
    with st.form("wave_cfg"):
        c1, c2, c3, c4 = st.columns(4)
        nz = c1.number_input("分区数量", min_value=1, max_value=10, value=5)
        rpz = c2.number_input("分区人数上限", min_value=100, max_value=5000, value=2000, step=100)
        rpw = c3.number_input("单波人数上限", min_value=50, max_value=1000, value=500, step=50)
        wiv = c4.number_input("波次间隔(分钟)", min_value=1, max_value=30, value=5)
        submitted = st.form_submit_button("重新编排")

    cfg = WaveConfig(
        num_zones=int(nz),
        runners_per_zone=int(rpz),
        runners_per_wave=int(rpw),
        wave_interval_minutes=int(wiv),
    )
    assigned, wa_list = assign_waves(roster, cfg)
    report = validate_assignments(assigned, wa_list, cfg)
    wa_df = wave_assignment_table(wa_list)

    st.subheader("编排结果")
    if report.all_valid:
        st.success("✅ 所有约束均满足")
    else:
        st.error("❌ 存在约束违规")
        if report.zone_capacity_violations:
            st.markdown("**分区容量违规：**")
            for v in report.zone_capacity_violations:
                st.warning(v)
        if report.wave_capacity_violations:
            st.markdown("**波次容量违规：**")
            for v in report.wave_capacity_violations:
                st.warning(v)
        if report.elite_in_priority_violations:
            st.markdown("**精英优先波违规：**")
            for v in report.elite_in_priority_violations[:5]:
                st.warning(v)
        if report.wheelchair_in_priority_violations:
            st.markdown("**轮椅优先波违规：**")
            for v in report.wheelchair_in_priority_violations[:5]:
                st.warning(v)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**分区人数**")
        zone_cnt = assigned.groupby("start_zone").size().reset_index(name="count")
        fig_z = px.bar(zone_cnt, x="start_zone", y="count", text="count")
        st.plotly_chart(fig_z, use_container_width=True)
    with col2:
        st.markdown("**波次人数与出发时间**")
        wa_dfp = wa_df.copy()
        wa_dfp["start_time_str"] = wa_dfp["start_time"].astype(str)
        fig_w = px.bar(
            wa_dfp,
            x="wave_id",
            y="runner_count",
            text="runner_count",
            hover_data=["start_time_str", "zone_id"],
            color="zone_id",
        )
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("**波次明细表**")
    st.dataframe(wa_dfp.drop(columns=["start_time_str"], errors="ignore"), use_container_width=True, hide_index=True)

    st.markdown("**分区后预估成绩分布**")
    fig3 = px.violin(
        assigned,
        x="start_zone",
        y="estimated_finish_hours",
        color="start_zone",
        box=True,
        labels={"estimated_finish_hours": "预估完赛时间 (小时)"},
    )
    st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# 主程序
# =========================================================
def main():
    st.set_page_config(page_title="马拉松赛事计时分析平台", layout="wide")
    st.title("🏃 城市马拉松赛事计时分析平台")
    st.caption(f"鸣枪时间：{GUN_START_TIME}  |  全马 42.195km  |  全程本地运行，无外部连接")

    with st.sidebar:
        st.header("数据源 & 参数")
        num = st.slider("选手规模", min_value=500, max_value=15000, value=8000, step=500)
        seed = st.number_input("随机种子", min_value=0, max_value=99999, value=42)
        st.markdown("---")
        st.markdown("**关门时间**")
        for p in TIMING_POINTS:
            td = CUTOFF_TIMES[p]
            dead = GUN_START_TIME + td
            st.caption(f"{p}: 枪后 {int(td.total_seconds()//3600)}h{int((td.total_seconds()%3600)//60):02d}m ({dead.strftime('%H:%M')})")

    roster, reads = build_dataset(num, seed)

    with st.spinner("正在重建成绩、判定关门与可疑行为..."):
        merged, segments, suspicions = run_pipeline(roster, reads)

    pages = {
        "📊 赛事总览": lambda: page_overview(roster, reads, merged, segments, suspicions),
        "🏆 榜单排名": lambda: page_rankings(roster, merged, segments, suspicions),
        "📈 分段流量 & 拥堵": lambda: page_flow(roster, merged, segments),
        "⚠️ 可疑名单": lambda: page_suspicion(roster, merged, suspicions),
        "🚦 分区波次编排": lambda: page_waves(roster),
    }
    sel = st.sidebar.radio("导航", list(pages.keys()))
    pages[sel]()


if __name__ == "__main__":
    main()
