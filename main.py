import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# ------------------------------
# 페이지 기본 설정
# ------------------------------
st.set_page_config(page_title="기온 예측기", layout="centered")
st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 이용해 연도별 평균기온의 추세를 회귀 분석으로 예측합니다.")

# ------------------------------
# 데이터 불러오기
# ------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df

df = load_data()

# ------------------------------
# 연도별 평균기온 및 관측일수 계산
# ------------------------------
yearly = df.groupby("연도").agg(
    평균기온=("평균기온", "mean"),
    관측일수=("평균기온", "count")
).reset_index()

# ------------------------------
# 조건에 맞는 데이터만 필터링
# - 2025년까지만 사용
# - 관측일수가 300일 미만인 해는 제외
# ------------------------------
yearly_filtered = yearly[
    (yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)
].sort_values("연도").reset_index(drop=True)

if len(yearly_filtered) < 2:
    st.error("회귀 분석을 하기에 데이터가 충분하지 않습니다.")
    st.stop()

# ------------------------------
# 전체 기간 회귀 분석 (연도 -> 평균기온)
# ------------------------------
x = yearly_filtered["연도"].values
y = yearly_filtered["평균기온"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

start_year = int(yearly_filtered["연도"].min())
end_year = int(yearly_filtered["연도"].max())
n_years = len(yearly_filtered)

# 100년당 상승 온도로 환산
slope_per_100yr = slope * 100

# ------------------------------
# 최근 20년 회귀 분석
# ------------------------------
recent_start_year = end_year - 19  # 최근 20개 연도(끝 연도 포함)
yearly_recent = yearly_filtered[yearly_filtered["연도"] >= recent_start_year]

has_recent_enough_data = len(yearly_recent) >= 2

if has_recent_enough_data:
    x_recent = yearly_recent["연도"].values
    y_recent = yearly_recent["평균기온"].values
    slope_recent, intercept_recent, r_value_recent, p_value_recent, std_err_recent = stats.linregress(
        x_recent, y_recent
    )
    slope_recent_per_100yr = slope_recent * 100
    n_recent = len(yearly_recent)
    recent_actual_start = int(yearly_recent["연도"].min())
    recent_actual_end = int(yearly_recent["연도"].max())

# ------------------------------
# 산점도 + 회귀직선 그래프 (Plotly)
# ------------------------------
fig = go.Figure()

# 전체 데이터 산점도
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode="markers",
    name="연평균기온 (관측)",
    marker=dict(color="royalblue", size=7)
))

# 전체 기간 회귀직선
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept

fig.add_trace(go.Scatter(
    x=line_x, y=line_y,
    mode="lines",
    name="전체 기간 회귀 직선",
    line=dict(color="firebrick", width=2)
))

# 최근 20년 회귀직선 (해당 구간만 표시)
if has_recent_enough_data:
    line_x_recent = np.linspace(x_recent.min(), x_recent.max(), 50)
    line_y_recent = slope_recent * line_x_recent + intercept_recent

    fig.add_trace(go.Scatter(
        x=line_x_recent, y=line_y_recent,
        mode="lines",
        name="최근 20년 회귀 직선",
        line=dict(color="darkorange", width=3, dash="dash")
    ))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    title="연도별 서울 평균기온과 회귀 직선 (전체 vs 최근 20년)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 상관계수 및 회귀 정보 표시
# ------------------------------
st.subheader("📊 회귀 분석 정보 (전체 기간)")
col1, col2, col3 = st.columns(3)
col1.metric("사용된 연도 수", f"{n_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")

st.write(f"**상관계수 (r):** {r_value:.4f}")
st.write(f"**결정계수 (R²):** {r_squared:.4f}")

# ------------------------------
# 100년당 상승 온도 - 전체 vs 최근 20년 비교
# ------------------------------
st.subheader("🔥 100년당 기온 상승률 비교")

if has_recent_enough_data:
    compare_col1, compare_col2 = st.columns(2)

    with compare_col1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background-color:#eef3fb; border-radius:10px;">
                <h4>전체 기간</h4>
                <p style="color:#555;">{start_year}년 ~ {end_year}년 ({n_years}개 연도)</p>
                <h1 style="color:#1f77b4; font-size:50px;">{slope_per_100yr:+.2f} °C</h1>
                <p style="color:#777;">/ 100년</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with compare_col2:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background-color:#fdf2e9; border-radius:10px;">
                <h4>최근 20년</h4>
                <p style="color:#555;">{recent_actual_start}년 ~ {recent_actual_end}년 ({n_recent}개 연도)</p>
                <h1 style="color:#d62728; font-size:50px;">{slope_recent_per_100yr:+.2f} °C</h1>
                <p style="color:#777;">/ 100년</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    diff = slope_recent_per_100yr - slope_per_100yr
    if diff > 0:
        st.info(f"📈 최근 20년의 기온 상승 속도가 전체 기간보다 **{diff:.2f}°C/100년** 더 빠릅니다.")
    else:
        st.info(f"📉 최근 20년의 기온 상승 속도가 전체 기간보다 **{abs(diff):.2f}°C/100년** 더 느립니다.")
else:
    st.warning("최근 20년 구간에 회귀 분석을 하기에 충분한 데이터가 없습니다.")

# ------------------------------
# 슬라이더로 연도 선택 -> 예상 기온 계산 (전체 기간 회귀식 사용)
# ------------------------------
st.subheader("🔮 연도별 예상 기온 확인하기")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;">
        <h3>{selected_year}년 예상 평균기온 (전체 기간 회귀식 기준)</h3>
        <h1 style="color:#d62728; font-size:60px;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("⚠️ 이 예측값은 과거 데이터를 바탕으로 한 단순 선형 회귀 결과이며, 실제 기온과 다를 수 있습니다.")
