"""매매 신호 알림 모듈.

최신 봉(오늘) 기준으로 방금 트리거된 기술적 조건을 탐지해 ``Alert`` 목록으로
반환한다. 모든 판정은 기존 ``src.indicators`` / ``src.trend`` / ``src.levels``
함수를 그대로 재사용하며, 규칙 기반 참고 신호일 뿐 매매 지시가 아니다.

탐지 조건:
  1. 골든/데드크로스 (이동평균 교차) — ``trend.detect_crosses``
  2. RSI 30/70 경계 돌파 — ``indicators.rsi``
  3. MACD 히스토그램 부호 전환(시그널 교차) — ``indicators.macd``
  4. 지지/저항 돌파 — ``levels.support_resistance``
  5. 볼린저밴드 상/하단 이탈 — ``indicators.bollinger``

주의: 본 도구는 참고용이며 투자 자문이나 매매 지시가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from . import indicators, levels
from .data import fetch_ohlcv
from .trend import detect_crosses

# 조건 종류별 기본 중요도(severity) 매핑
_SEVERITY = {
    "cross": "high",
    "level": "high",
    "macd": "medium",
    "bollinger": "medium",
    "rsi": "medium",
}


@dataclass
class Alert:
    """오늘 트리거된 알림 1건.

    trend.TrendResult / signals.SignalReport와 동일한 스타일의 dataclass다.
    """

    kind: str  # "cross" | "rsi" | "macd" | "level" | "bollinger"
    direction: str  # "bullish"(강세/매수 우호) | "bearish"(약세/매도 우호)
    message: str  # 사람이 읽을 한국어 근거 문장 (참고용, 매매 지시 아님)
    severity: str  # "high" | "medium" | "low"

    @property
    def emoji(self) -> str:
        """방향(direction)에 따른 표시용 이모지."""
        return {"bullish": "🟢", "bearish": "🔴"}.get(self.direction, "⚪")


def check_alerts(
    df: pd.DataFrame,
    ticker: str,
    windows: tuple[int, int] = (20, 60),
) -> list[Alert]:
    """최신 봉 기준으로 오늘 트리거된 조건들만 Alert 리스트로 반환한다.

    조건이 하나도 없으면 빈 리스트를 돌려준다. 각 조건은 기존 함수를 그대로
    재사용해 판정하며, 데이터 부족(NaN/봉 수 미달) 시 해당 조건은 조용히 건너뛴다.

    Parameters
    ----------
    df : OHLCV DataFrame (Close 컬럼 필수, DatetimeIndex 권장)
    ticker : 알림 메시지에 표기할 종목명/티커
    windows : (단기, 장기) 이동평균 기간
    """
    alerts: list[Alert] = []
    if df is None or len(df) < 2:
        return alerts

    close = df["Close"].astype("float64")
    short, long = int(windows[0]), int(windows[1])

    # --- (1) 골든/데드크로스: '오늘' 발생한 경우만 -----------------------
    try:
        crosses = detect_crosses(df, short=short, long=long)
        if not crosses.empty and crosses.index[-1] == df.index[-1]:
            ctype = crosses.iloc[-1]["type"]
            if ctype == "golden":
                alerts.append(
                    Alert(
                        kind="cross",
                        direction="bullish",
                        message=(
                            f"[{ticker}] 오늘 골든크로스({short}/{long}) 발생 — "
                            f"단기 이평이 장기 이평을 상향 돌파"
                        ),
                        severity=_SEVERITY["cross"],
                    )
                )
            elif ctype == "dead":
                alerts.append(
                    Alert(
                        kind="cross",
                        direction="bearish",
                        message=(
                            f"[{ticker}] 오늘 데드크로스({short}/{long}) 발생 — "
                            f"단기 이평이 장기 이평을 하향 돌파"
                        ),
                        severity=_SEVERITY["cross"],
                    )
                )
    except Exception:  # noqa: BLE001 - 개별 조건 실패는 조용히 건너뜀
        pass

    # --- (2) RSI 30/70 경계 돌파 ----------------------------------------
    try:
        r = indicators.rsi(close, 14).dropna()
        if len(r) >= 2:
            prev, cur = float(r.iloc[-2]), float(r.iloc[-1])
            if prev < 30 <= cur:
                alerts.append(
                    Alert(
                        kind="rsi",
                        direction="bullish",
                        message=(
                            f"[{ticker}] RSI가 과매도 30선을 상향 돌파 "
                            f"({prev:.1f}→{cur:.1f}) — 반등 관심"
                        ),
                        severity=_SEVERITY["rsi"],
                    )
                )
            elif prev > 70 >= cur:
                alerts.append(
                    Alert(
                        kind="rsi",
                        direction="bearish",
                        message=(
                            f"[{ticker}] RSI가 과매수 70선을 하향 이탈 "
                            f"({prev:.1f}→{cur:.1f}) — 과열 주의"
                        ),
                        severity=_SEVERITY["rsi"],
                    )
                )
    except Exception:  # noqa: BLE001
        pass

    # --- (3) MACD 히스토그램 부호 전환(시그널 교차) --------------------
    #     signals.evaluate와 동일 규칙: prev<=0<cur(bullish), prev>=0>cur(bearish)
    try:
        hist = indicators.macd(close)["hist"].dropna()
        if len(hist) >= 2:
            prev, cur = float(hist.iloc[-2]), float(hist.iloc[-1])
            if prev <= 0 < cur:
                alerts.append(
                    Alert(
                        kind="macd",
                        direction="bullish",
                        message=(
                            f"[{ticker}] MACD가 시그널선을 상향 돌파(골든) — "
                            f"상승 모멘텀 전환"
                        ),
                        severity=_SEVERITY["macd"],
                    )
                )
            elif prev >= 0 > cur:
                alerts.append(
                    Alert(
                        kind="macd",
                        direction="bearish",
                        message=(
                            f"[{ticker}] MACD가 시그널선을 하향 돌파(데드) — "
                            f"하락 모멘텀 전환"
                        ),
                        severity=_SEVERITY["macd"],
                    )
                )
    except Exception:  # noqa: BLE001
        pass

    # --- (4) 지지/저항 돌파 --------------------------------------------
    #     레벨은 반드시 직전 봉까지(df.iloc[:-1])로 계산한다.
    #     (support_resistance는 현재가 기준으로 분류하므로 df 전체로는 돌파를 못 잡음)
    try:
        prev_close = float(close.iloc[-2])
        cur_close = float(close.iloc[-1])
        lv = levels.support_resistance(df.iloc[:-1])
        # 저항 상향 돌파: 전일 종가 <= 저항 R < 오늘 종가
        for rr in lv["resistances"]:
            if prev_close <= rr < cur_close:
                alerts.append(
                    Alert(
                        kind="level",
                        direction="bullish",
                        message=(
                            f"[{ticker}] 저항 {rr:,.2f} 상향 돌파 "
                            f"(종가 {prev_close:,.2f}→{cur_close:,.2f})"
                        ),
                        severity=_SEVERITY["level"],
                    )
                )
                break
        # 지지 하향 이탈: 전일 종가 >= 지지 S > 오늘 종가
        for ss in lv["supports"]:
            if prev_close >= ss > cur_close:
                alerts.append(
                    Alert(
                        kind="level",
                        direction="bearish",
                        message=(
                            f"[{ticker}] 지지 {ss:,.2f} 하향 이탈 "
                            f"(종가 {prev_close:,.2f}→{cur_close:,.2f})"
                        ),
                        severity=_SEVERITY["level"],
                    )
                )
                break
    except Exception:  # noqa: BLE001
        pass

    # --- (5) 볼린저밴드 상/하단 이탈 -----------------------------------
    #     신선도: 전일은 밴드 내부였다가 오늘 이탈한 경우만 알린다.
    #     signals.evaluate 방향 규칙과 일치: 상단 초과=bearish, 하단 미만=bullish
    try:
        bb = indicators.bollinger(close, 20, 2.0).dropna()
        if len(bb) >= 2:
            prev_close = float(close.iloc[-2])
            cur_close = float(close.iloc[-1])
            upper_prev = float(bb["upper"].iloc[-2])
            lower_prev = float(bb["lower"].iloc[-2])
            upper_cur = float(bb["upper"].iloc[-1])
            lower_cur = float(bb["lower"].iloc[-1])
            # 상단: 전일 내부(<=upper) → 오늘 초과(>upper) = 과열/이탈(bearish)
            if prev_close <= upper_prev and cur_close > upper_cur:
                alerts.append(
                    Alert(
                        kind="bollinger",
                        direction="bearish",
                        message=(
                            f"[{ticker}] 볼린저 상단({upper_cur:,.2f}) 상향 이탈 — "
                            f"단기 과열 주의"
                        ),
                        severity=_SEVERITY["bollinger"],
                    )
                )
            # 하단: 전일 내부(>=lower) → 오늘 미만(<lower) = 과매도/이탈(bullish)
            elif prev_close >= lower_prev and cur_close < lower_cur:
                alerts.append(
                    Alert(
                        kind="bollinger",
                        direction="bullish",
                        message=(
                            f"[{ticker}] 볼린저 하단({lower_cur:,.2f}) 하향 이탈 — "
                            f"단기 과매도(반등 관심)"
                        ),
                        severity=_SEVERITY["bollinger"],
                    )
                )
    except Exception:  # noqa: BLE001
        pass

    return alerts


def check_watchlist(
    tickers: Sequence[str],
    fetch: Callable[[str], pd.DataFrame] = fetch_ohlcv,
    period: str = "6mo",
    interval: str = "1d",
    windows: tuple[int, int] = (20, 60),
) -> dict[str, list[Alert]]:
    """워치리스트의 여러 티커를 한 번에 점검한다.

    각 티커에 대해 ``fetch``로 df를 얻어 ``check_alerts``를 호출하고
    ``{ticker: list[Alert]}`` 딕셔너리를 반환한다. 특정 티커의 조회 실패/예외는
    해당 티커만 빈 리스트로 처리하고 전체는 계속 진행한다(전체 중단 금지).

    ``fetch``를 주입 가능하게 설계해, 오프라인 테스트에서는 sample_data 기반의
    가짜 fetch를 넣어 검증한다. 정규화(normalize_ticker)는 fetch_ohlcv 내부에
    위임한다.

    Notes
    -----
    기본 fetch(fetch_ohlcv)는 라이브(yfinance) 경로다.
    # 사용자 PC에서 검증 필요 — 이 샌드박스는 fc.yahoo.com(403) 차단으로 실행 불가.
    """
    results: dict[str, list[Alert]] = {}
    for ticker in tickers:
        try:
            # fetch 시그니처 호환: period/interval을 지원하면 넘기고, 아니면 티커만.
            try:
                df = fetch(ticker, period=period, interval=interval)  # type: ignore[call-arg]
            except TypeError:
                df = fetch(ticker)
            results[ticker] = check_alerts(df, ticker, windows)
        except Exception:  # noqa: BLE001 - 개별 티커 실패는 건너뜀
            results[ticker] = []
    return results


# 표시용 한국어 라벨 매핑
_KIND_LABEL = {
    "cross": "이평 교차",
    "rsi": "RSI",
    "macd": "MACD",
    "level": "지지/저항",
    "bollinger": "볼린저밴드",
}
_DIRECTION_LABEL = {
    "bullish": "강세🟢",
    "bearish": "약세🔴",
}
_SEVERITY_LABEL = {
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
}


def alerts_dataframe(alerts: list[Alert]) -> pd.DataFrame:
    """Alert 리스트를 표시용 DataFrame으로 변환한다.

    signals.signals_dataframe와 동일한 패턴의 헬퍼로, app.py의 st.dataframe에서
    바로 렌더링할 수 있게 컬럼 ['종류','방향','중요도','내용']으로 매핑한다.
    """
    rows = [
        {
            "종류": _KIND_LABEL.get(a.kind, a.kind),
            "방향": _DIRECTION_LABEL.get(a.direction, a.direction),
            "중요도": _SEVERITY_LABEL.get(a.severity, a.severity),
            "내용": a.message,
        }
        for a in alerts
    ]
    return pd.DataFrame(rows, columns=["종류", "방향", "중요도", "내용"])
