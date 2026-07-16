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

from src import alerts, backtest, indicators, levels, screener, signals, universe
from src.data import (
    VALID_INTERVALS,
    VALID_PERIODS,
    fetch_ohlcv,
    market_label,
    resolve_ticker,
    sample_data,
)
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
        help=(
            "예) AAPL, TSLA, NVDA · 한국: 005930 또는 005930.KS, 코스닥 035720.KQ · "
            "한글 종목명(예: 삼성전자, 카카오)도 인식"
        ),
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
def render_summary(
    df: pd.DataFrame,
    windows: tuple[int, ...],
    currency: str = "USD",
    market: str = "US",
) -> None:
    trend = classify_trend(df, windows=windows)
    report = signals.evaluate(df)

    price = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
    change = price - prev
    change_pct = (change / prev * 100) if prev else 0.0

    # 통화별 가격 포맷: KRW는 원화 기호·정수, 그 외(USD 등)는 소수 2자리
    if currency == "KRW":
        price_str = f"₩{price:,.0f}"
        change_str = f"{change:+,.0f} ({change_pct:+.2f}%)"
    else:
        price_str = f"${price:,.2f}"
        change_str = f"{change:+,.2f} ({change_pct:+.2f}%)"

    c1, c2, c3 = st.columns([1.1, 1.1, 1.4])
    with c1:
        st.metric("현재가", price_str, change_str)
        # 시장/통화 배지 (예: 'KOSPI · KRW', 'US · USD', 데모는 'DEMO · -')
        st.caption(f"{market} · {currency}")
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
# 매매 점수 (0~100 게이지)
# ---------------------------------------------------------------------------
def build_score_gauge(ts: signals.TradeScore) -> go.Figure:
    """0~100 매매 점수를 게이지로 그린다 (Streamlit 런타임 불필요, 테스트 가능)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=ts.score,
            number={"suffix": " 점", "font": {"size": 40}},
            title={"text": f"매매 점수 {ts.emoji} <b>{ts.label}</b>"},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100]},
                "bar": {"color": "rgba(40,40,40,0.85)"},
                "steps": [
                    {"range": [0, 25], "color": "#c0392b"},   # 강한 매도
                    {"range": [25, 40], "color": "#e88f88"},  # 매도
                    {"range": [40, 60], "color": "#f2e2a8"},  # 중립
                    {"range": [60, 75], "color": "#8fc0e8"},  # 매수
                    {"range": [75, 100], "color": "#1f6fe0"}, # 강한 매수
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": ts.score,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def render_trade_score(df: pd.DataFrame) -> None:
    """분석 탭 상단의 한눈 매매 점수 섹션."""
    ts = signals.trade_score(df)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(build_score_gauge(ts), use_container_width=True)
    with c2:
        st.metric(f"종합 판단 {ts.emoji}", ts.label, f"{ts.score}/100")
        st.caption(
            "0=강한 매도 · 50=중립 · 100=강한 매수. "
            "추세(55%)와 규칙 신호(45%)를 합친 참고 점수입니다."
        )
        st.caption("※ 개인 참고용이며 매매 지시·수익 보장이 아닙니다.")


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
# 백테스트
# ---------------------------------------------------------------------------
def build_backtest_chart(
    result: backtest.BacktestResult,
    buy_hold: pd.Series | None = None,
) -> go.Figure:
    """백테스트 자산곡선 Figure를 만든다 (Streamlit 런타임 불필요, 테스트 가능).

    build_chart의 go.Scatter 색상/레이아웃 패턴을 재사용하며, 바이앤홀드 곡선을
    회색 점선 비교선으로 함께 표시한다.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve,
            name="전략 자산",
            line=dict(width=1.6, color="#e0453e"),  # build_chart 상승색(빨강) 재사용
        )
    )
    if buy_hold is not None:
        fig.add_trace(
            go.Scatter(
                x=buy_hold.index,
                y=buy_hold,
                name="바이앤홀드",
                line=dict(width=1.2, color="rgba(150,150,150,0.8)", dash="dot"),
            )
        )
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="자산(초기=1.0)",
    )
    return fig


def render_backtest(df: pd.DataFrame, cfg: dict) -> None:
    """백테스트 탭 UI — 전략 선택, 자산곡선, 지표 표, 거래 상세."""
    st.subheader("🔁 신호 기반 백테스트 (롱-온리)")

    label_map = {"골든크로스": "golden_cross", "RSI 과매도/과매수": "rsi"}
    strat_label = st.selectbox("전략", list(label_map.keys()))
    strategy = label_map[strat_label]

    # 골든크로스는 사이드바 이동평균 설정(단기/장기)을 그대로 재사용
    short, long = cfg["windows"][0], cfg["windows"][1]
    if strategy == "golden_cross":
        st.caption(f"단기 {short}일 · 장기 {long}일 이동평균 교차 기준")

    result = backtest.run_backtest(df, strategy, short=short, long=long)

    # 바이앤홀드 비교 곡선(첫 유효종가 기준 정규화)
    close = df["Close"].astype("float64")
    valid = close.dropna()
    buy_hold = close / float(valid.iloc[0]) if len(valid) else None

    st.plotly_chart(
        build_backtest_chart(result, buy_hold),
        use_container_width=True,
    )

    st.dataframe(
        backtest.summary_dataframe(result),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(f"거래 상세 ({result.num_trades}건)"):
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            trades_df = trades_df.rename(
                columns={
                    "entry_date": "진입일",
                    "entry_price": "진입가",
                    "exit_date": "청산일",
                    "exit_price": "청산가",
                    "return_pct": "수익률",
                }
            )
            trades_df["수익률"] = trades_df["수익률"].map(lambda v: f"{v * 100:+.2f}%")
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("체결된 거래가 없습니다. (신호 미발생 또는 데이터 부족)")

    st.caption("※ 과거 성과는 미래 수익을 보장하지 않으며 참고용이고 매매 지시가 아닙니다.")


# ---------------------------------------------------------------------------
# 종목 스크리너
# ---------------------------------------------------------------------------
def parse_tickers(text: str) -> list[str]:
    """텍스트 입력(콤마/줄바꿈 구분)을 티커 목록으로 파싱한다.

    Streamlit 런타임 없이 테스트 가능한 순수 함수. 빈 항목/중복은 제거한다.
    """
    raw = text.replace("\n", ",").split(",")
    out: list[str] = []
    for tok in raw:
        t = tok.strip()
        if t and t not in out:
            out.append(t)
    return out


def make_demo_fetch(days: int = 400):
    """데모(오프라인) 모드용 주입 fetch를 만든다 (Streamlit 런타임 불필요, 테스트 가능).

    티커 문자열을 시드로 변환해 종목마다 서로 다른 합성 시계열을 돌려준다.
    **실제 시세가 아닌 합성 데이터**이므로 화면에는 경고 문구를 함께 표시한다.
    """

    def _fetch(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        seed = abs(hash(ticker)) % 10_000  # 티커별 고정 시드
        return sample_data(days=days, seed=seed)

    return _fetch


def render_screener(cfg: dict) -> None:
    """스크리너 탭 UI — 티커 목록 입력 · 필터 · 실행 · 결과 표."""
    st.subheader("🔎 종목 스크리너")
    st.caption("여러 종목을 일괄 조회·평가해 추세/신호 기준으로 필터링·랭킹합니다.")

    text = st.text_area(
        "티커 목록 (콤마 또는 줄바꿈 구분)",
        value="",
        placeholder="AAPL, MSFT, 005930",
        help="예) AAPL, MSFT, NVDA · 한국: 005930 또는 005930.KS, 코스닥 035720.KQ",
    )

    # --- 필터 위젯 ---
    f1, f2 = st.columns(2)
    with f1:
        trend_up = st.checkbox("추세 상승만", value=False)
        recent_golden = st.checkbox("최근 골든크로스", value=False)
    with f2:
        rsi_range = st.slider("RSI 범위", 0, 100, (0, 100))
        use_above_sma = st.checkbox("종가가 SMA20 위", value=False)

    criteria: dict = {}
    if trend_up:
        criteria["trend_up"] = True
    if recent_golden:
        criteria["recent_golden_cross"] = True
    # RSI 범위가 기본(0~100)이 아닐 때만 조건으로 반영
    if rsi_range[0] > 0:
        criteria["rsi_min"] = float(rsi_range[0])
    if rsi_range[1] < 100:
        criteria["rsi_max"] = float(rsi_range[1])
    if use_above_sma:
        criteria["above_sma"] = 20

    if cfg["demo"]:
        st.warning(
            "🧪 **데모 모드** — 스크리너 결과는 티커별 합성 샘플 데이터 기반이며 실제 시세가 아닙니다."
        )

    if st.button("스크리닝 실행"):
        tickers = parse_tickers(text)
        if not tickers:
            st.info("티커를 한 개 이상 입력하세요. (예: AAPL, MSFT, 005930)")
            return

        # 데모 모드면 오프라인 주입 fetch, 라이브 모드면 기본 fetch_ohlcv(fetch=None)
        fetch = make_demo_fetch() if cfg["demo"] else None
        with st.spinner(f"{len(tickers)}개 종목을 평가하는 중…"):
            result = screener.screen(
                tickers,
                criteria=criteria or None,
                fetch=fetch,
                period=cfg["period"],
                interval=cfg["interval"],
                windows=cfg["windows"],
            )

        if result.empty:
            st.info("조건을 통과한 종목이 없습니다. 필터를 완화하거나 티커를 확인하세요.")
        else:
            st.dataframe(result, use_container_width=True, hide_index=True)

    st.caption("※ 스크리너 결과는 기술적 조건 기반 참고용이며 매매 지시가 아닙니다.")


# ---------------------------------------------------------------------------
# 추천 (대표 종목 자동 스캔 → 매매 점수 상위)
# ---------------------------------------------------------------------------
def render_recommend(cfg: dict) -> None:
    """추천 탭 — 한국·미국 대표주를 스캔해 매매 점수 상위 종목을 표로 보여준다."""
    st.subheader("🎯 오늘의 매수 후보 추천")
    st.caption(
        "대표 종목을 자동으로 훑어 **매매 점수 높은 순**으로 한국·미국 각 10개를 뽑습니다. "
        "종목 추천/보증이 아니라 기술적 점수 순 정렬입니다."
    )

    min_score = st.slider("최소 매매 점수 (이 점수 이상만)", 50, 90, 60, step=5)

    if cfg["demo"]:
        st.warning(
            "🧪 **데모 모드** — 추천 결과는 합성 샘플 데이터 기반이며 실제 시세가 아닙니다. "
            "진짜 추천은 실시간 데이터가 되는 환경(내 링크/PC)에서 실행하세요."
        )

    if st.button("추천 종목 스캔"):
        fetch = make_demo_fetch() if cfg["demo"] else None
        c_kr, c_us = st.columns(2)

        with c_kr:
            st.markdown("### 🇰🇷 한국 Top 10")
            with st.spinner("한국 대표주 스캔 중… (실시간은 다소 걸릴 수 있어요)"):
                kr = screener.recommend(
                    universe.KR_UNIVERSE, top_n=10, fetch=fetch,
                    period=cfg["period"], interval=cfg["interval"],
                    windows=cfg["windows"], min_score=min_score,
                )
            if kr.empty:
                st.info("조건(매수 점수 이상)을 만족하는 종목이 없습니다.")
            else:
                st.dataframe(kr, use_container_width=True, hide_index=True)

        with c_us:
            st.markdown("### 🇺🇸 미국 Top 10")
            with st.spinner("미국 대표주 스캔 중…"):
                us = screener.recommend(
                    universe.US_UNIVERSE, top_n=10, fetch=fetch,
                    period=cfg["period"], interval=cfg["interval"],
                    windows=cfg["windows"], min_score=min_score,
                )
            if us.empty:
                st.info("조건(매수 점수 이상)을 만족하는 종목이 없습니다.")
            else:
                st.dataframe(us, use_container_width=True, hide_index=True)

    st.caption("※ 참고용 개인 도구입니다. 매매 지시·수익 보장이 아니며 투자 책임은 본인에게 있습니다.")


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
        # 데모는 합성 데이터이므로 시장/통화 라벨을 표기하지 않는다
        market, currency = "DEMO", "-"
    else:
        if not cfg["ticker"]:
            st.info("왼쪽 사이드바에 티커를 입력하세요. (예: AAPL, 005930, 삼성전자)")
            return
        # 입력(한글명/코드/일반티커)을 야후 심볼로 정규화하고 시장·통화 라벨 획득
        try:
            resolved = resolve_ticker(cfg["ticker"])
        except ValueError as exc:
            st.info(str(exc))
            return
        market, currency = market_label(resolved)
        try:
            with st.spinner(f"{cfg['ticker']} ({resolved}) 데이터를 불러오는 중…"):
                df = fetch_ohlcv(resolved, cfg["period"], cfg["interval"])
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

    tab_analysis, tab_recommend, tab_screener, tab_backtest = st.tabs(
        ["분석", "추천", "스크리너", "백테스트"]
    )

    # --- 분석 탭: 요약 + 차트 + 신호 + 오늘의 알림 ---
    with tab_analysis:
        render_trade_score(df)
        render_summary(df, cfg["windows"], currency=currency, market=market)
        # 알림은 단기/장기 이동평균 기준으로 판정 (사이드바 설정 재사용)
        render_alerts(df, cfg["ticker"] or "DEMO", (cfg["windows"][0], cfg["windows"][1]))
        render_chart(df, cfg)
        render_signal_table(df)

    # --- 추천 탭: 대표주 자동 스캔 → 매매 점수 상위 ---
    with tab_recommend:
        render_recommend(cfg)

    # --- 스크리너 탭: 여러 종목 일괄 조회·필터·랭킹 ---
    with tab_screener:
        render_screener(cfg)

    # --- 백테스트 탭: 신호 기반 롱-온리 시뮬레이션 ---
    with tab_backtest:
        render_backtest(df, cfg)

    st.divider()
    st.caption(
        "⚠️ 본 도구는 기술적 지표 기반 참고용 분석이며, 투자 자문이나 매매 지시가 아닙니다. "
        "데이터는 지연될 수 있고 오류가 있을 수 있으며, 모든 투자 판단과 책임은 사용자 본인에게 있습니다."
    )


if __name__ == "__main__":
    main()
