"""주식 추세 분석 앱 — Streamlit 대시보드.

실행:  streamlit run app.py

티커를 입력하면 캔들 차트, 자동 추세 판정, 보조지표(RSI/MACD),
규칙 기반 매매 신호, 지지/저항선을 한 화면에서 보여준다.

⚠️ 투자 판단 및 그 책임은 전적으로 사용자 본인에게 있습니다. 본 도구는 참고용입니다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import alerts, indicators, levels, signals
from src.data import VALID_INTERVALS, VALID_PERIODS, fetch_ohlcv, sample_data
from src.trend import classify_trend, detect_crosses

st.set_page_config(page_title="주식 추세 분석기", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# 사이드바 — 입력
# ---------------------------------------------------------------------------
def sidebar_inputs() -> dict:
    st.sidebar.header("⚙️ 설정")
    ticker = st.sidebar.text_input(
        "티커 / 종목코드",
        value="AAPL",
        help="예) AAPL, TSLA, NVDA · 한국: 005930 또는 005930.KS, 코스닥 035720.KQ",
    ).strip()

    period = st.sidebar.selectbox("기간", VALID_PERIODS, index=VALID_PERIODS.index("1y"))
    interval = st.sidebar.selectbox("주기", VALID_INTERVALS, index=0)

    st.sidebar.subheader("이동평균선 (일)")
    ma_short = st.sidebar.number_input("단기", min_value=2, max_value=120, value=20, step=1)
    ma_mid = st.sidebar.number_input("중기", min_value=5, max_value=200, value=60, step=1)
    ma_long = st.sidebar.number_input("장기", min_value=10, max_value=400, value=120, step=1)

    st.sidebar.subheader("표시 옵션")
    show_bb = st.sidebar.checkbox("볼린저밴드", value=True)
    show_levels = st.sidebar.checkbox("지지/저항선", value=True)
    show_crosses = st.sidebar.checkbox("골든/데드크로스 마커", value=True)

    st.sidebar.divider()
    demo = st.sidebar.checkbox(
        "🧪 데모 데이터 사용 (오프라인)",
        value=False,
        help="네트워크가 없거나 시세 조회가 막힌 환경에서 UI를 확인할 때. 실제 시세가 아닌 합성 데이터입니다.",
    )

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "windows": (int(ma_short), int(ma_mid), int(ma_long)),
        "show_bb": show_bb,
        "show_levels": show_levels,
        "show_crosses": show_crosses,
        "demo": demo,
    }


# ---------------------------------------------------------------------------
# 상단 — 추세 판정 카드 + 종합 신호
# ---------------------------------------------------------------------------
def render_summary(df: pd.DataFrame, windows: tuple[int, ...]) -> None:
    trend = classify_trend(df, windows=windows)
    report = signals.evaluate(df)

    price = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
    change = price - prev
    change_pct = (change / prev * 100) if prev else 0.0

    c1, c2, c3 = st.columns([1.1, 1.1, 1.4])
    with c1:
        st.metric("현재가", f"{price:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
    with c2:
        st.metric(
            f"추세 판정 {trend.emoji}",
            trend.label,
            f"신뢰도 {trend.confidence}%",
            delta_color="off",
        )
    with c3:
        st.metric(
            f"종합 신호 {report.emoji}",
            report.stance,
            f"점수 {report.score:+d}",
            delta_color="off",
        )

    with st.expander("판정 근거 보기", expanded=True):
        st.markdown("**추세 근거**")
        for reason in trend.reasons:
            st.markdown(f"- {reason}")


# ---------------------------------------------------------------------------
# 메인 차트
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, cfg: dict) -> go.Figure:
    """OHLCV + 설정으로 plotly Figure를 만든다 (Streamlit 런타임 불필요, 테스트 가능)."""
    close = df["Close"].astype("float64")
    windows = cfg["windows"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("가격 · 이동평균 · 볼린저밴드", "RSI(14)", "MACD"),
    )

    # --- (1) 캔들 + 이동평균 ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
            increasing_line_color="#e0453e",  # 상승=빨강(한국식)
            decreasing_line_color="#1f6fe0",  # 하락=파랑
        ),
        row=1,
        col=1,
    )
    ma_colors = ["#f5a623", "#7ed321", "#9013fe"]
    for w, color in zip(windows, ma_colors):
        if w <= len(close):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators.sma(close, w),
                    name=f"SMA{w}",
                    line=dict(width=1.2, color=color),
                ),
                row=1,
                col=1,
            )

    # --- 볼린저밴드 ---
    if cfg["show_bb"]:
        bb = indicators.bollinger(close, 20, 2.0)
        fig.add_trace(
            go.Scatter(x=df.index, y=bb["upper"], name="BB 상단",
                       line=dict(width=0.8, color="rgba(150,150,150,0.6)")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=bb["lower"], name="BB 하단",
                       line=dict(width=0.8, color="rgba(150,150,150,0.6)"),
                       fill="tonexty", fillcolor="rgba(150,150,150,0.08)"),
            row=1, col=1,
        )

    # --- 지지/저항선 ---
    if cfg["show_levels"]:
        lv = levels.support_resistance(df)
        for s in lv["supports"]:
            fig.add_hline(y=s, line=dict(color="#1f6fe0", width=1, dash="dot"),
                          annotation_text=f"지지 {s:,.2f}", annotation_position="right",
                          row=1, col=1)
        for r in lv["resistances"]:
            fig.add_hline(y=r, line=dict(color="#e0453e", width=1, dash="dot"),
                          annotation_text=f"저항 {r:,.2f}", annotation_position="right",
                          row=1, col=1)

    # --- 골든/데드크로스 마커 ---
    if cfg["show_crosses"]:
        crosses = detect_crosses(df, short=windows[0], long=windows[1])
        for idx, row in crosses.iterrows():
            is_golden = row["type"] == "golden"
            fig.add_trace(
                go.Scatter(
                    x=[idx], y=[row["price"]], mode="markers",
                    marker=dict(
                        symbol="triangle-up" if is_golden else "triangle-down",
                        size=12,
                        color="#e0453e" if is_golden else "#1f6fe0",
                        line=dict(width=1, color="white"),
                    ),
                    name="골든크로스" if is_golden else "데드크로스",
                    showlegend=False,
                    hovertext="골든크로스" if is_golden else "데드크로스",
                ),
                row=1, col=1,
            )

    # --- (2) RSI ---
    rsi_series = indicators.rsi(close, 14)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name="RSI",
                             line=dict(width=1.2, color="#f5a623")), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="rgba(224,69,62,0.5)", width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="rgba(31,111,224,0.5)", width=1, dash="dash"), row=2, col=1)

    # --- (3) MACD ---
    m = indicators.macd(close)
    fig.add_trace(go.Bar(x=df.index, y=m["hist"], name="히스토그램",
                         marker_color="rgba(120,120,120,0.5)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=m["macd"], name="MACD",
                             line=dict(width=1.1, color="#1f6fe0")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=m["signal"], name="Signal",
                             line=dict(width=1.1, color="#e0453e")), row=3, col=1)

    fig.update_layout(
        height=760,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


def render_chart(df: pd.DataFrame, cfg: dict) -> None:
    st.plotly_chart(build_chart(df, cfg), use_container_width=True)


# ---------------------------------------------------------------------------
# 신호 테이블
# ---------------------------------------------------------------------------
def render_signal_table(df: pd.DataFrame) -> None:
    report = signals.evaluate(df)
    st.subheader("📋 규칙별 신호")
    st.dataframe(
        signals.signals_dataframe(report),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# 오늘의 알림 (매매 신호 알림)
# ---------------------------------------------------------------------------
def render_alerts(df: pd.DataFrame, ticker: str, windows: tuple[int, int]) -> None:
    """최신 봉 기준으로 오늘 트리거된 기술적 조건을 알림 섹션으로 표시한다."""
    st.subheader("🔔 오늘의 알림")
    today_alerts = alerts.check_alerts(df, ticker, windows=windows)

    if not today_alerts:
        st.info("오늘 트리거된 알림이 없습니다.")
    else:
        # 중요도 high는 강조 톤(warning), 그 외는 info 톤으로 개별 표시
        for a in today_alerts:
            if a.severity == "high":
                st.warning(f"{a.emoji} {a.message}")
            else:
                st.info(f"{a.emoji} {a.message}")
        # 요약 표(종류/방향/중요도/내용)
        st.dataframe(
            alerts.alerts_dataframe(today_alerts),
            use_container_width=True,
            hide_index=True,
        )

    st.caption("※ 오늘의 알림은 기술적 조건 탐지 결과로 참고용이며, 매매 지시가 아닙니다.")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("📈 주식 추세 분석기")
    st.caption("추세 · 이동평균 · RSI · MACD · 매매 신호 · 지지/저항을 한눈에")

    cfg = sidebar_inputs()

    if cfg["demo"]:
        st.warning("🧪 **데모 모드** — 아래는 합성 샘플 데이터이며 실제 시세가 아닙니다.")
        df = sample_data()
    else:
        if not cfg["ticker"]:
            st.info("왼쪽 사이드바에 티커를 입력하세요. (예: AAPL, 005930)")
            return
        try:
            with st.spinner(f"{cfg['ticker']} 데이터를 불러오는 중…"):
                df = fetch_ohlcv(cfg["ticker"], cfg["period"], cfg["interval"])
        except Exception as exc:  # noqa: BLE001 - 사용자에게 원인 표시
            st.error(f"데이터를 불러오지 못했습니다: {exc}")
            st.info(
                "네트워크가 막힌 환경이라면 사이드바의 **'데모 데이터 사용'** 을 켜서 "
                "UI를 확인할 수 있습니다."
            )
            return

    if len(df) < 2:
        st.warning("표시할 데이터가 충분하지 않습니다. 기간을 늘려보세요.")
        return

    tab_analysis, tab_screener, tab_backtest = st.tabs(["분석", "스크리너", "백테스트"])

    # --- 분석 탭: 요약 + 차트 + 신호 + 오늘의 알림 ---
    with tab_analysis:
        render_summary(df, cfg["windows"])
        # 알림은 단기/장기 이동평균 기준으로 판정 (사이드바 설정 재사용)
        render_alerts(df, cfg["ticker"] or "DEMO", (cfg["windows"][0], cfg["windows"][1]))
        render_chart(df, cfg)
        render_signal_table(df)

    # --- 스크리너 탭: 향후 check_watchlist 기반 기능으로 채울 예정 ---
    with tab_screener:
        st.info("🛠️ 스크리너는 준비 중입니다. (워치리스트 일괄 알림 점검 예정)")

    # --- 백테스트 탭: 향후 기능 ---
    with tab_backtest:
        st.info("🛠️ 백테스트는 준비 중입니다.")

    st.divider()
    st.caption(
        "⚠️ 본 도구는 기술적 지표 기반 참고용 분석이며, 투자 자문이나 매매 지시가 아닙니다. "
        "데이터는 지연될 수 있고 오류가 있을 수 있으며, 모든 투자 판단과 책임은 사용자 본인에게 있습니다."
    )


if __name__ == "__main__":
    main()
