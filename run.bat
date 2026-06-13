@echo off
REM 启动马拉松赛事计时分析平台（端口 7956）
echo ==============================================
echo   城市马拉松赛事计时分析平台
echo   端口: 7956
echo ==============================================
cd /d %~dp0
python -m streamlit run app.py --server.port 7956 --server.address 0.0.0.0
pause
