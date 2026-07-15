"""규칙 기반 매수/매도 신호 모듈.

여러 보조지표(이동평균 교차, RSI, MACD, 볼린저밴드)를 각각 +1(매수)·-1(매도)·0(중립)
으로 평가해 합산하고, 최신 시점의 종합 스탠스(매수 관심 / 중립 / 매도 관심)를 낸다.

주의: 규칙 기반 참고 신호일 뿐 매매 지시가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import indicators
from .trend import detect_crosses


@dataclass
class SignalReport:
    """최신 시점 종합 신호."""

    stance: str  # "매수 관심" | "중립" | "매도 관심"
    score: int  # 규칙 점수 합계 (양수=매수 우위)
    rows: list[dict] = field(default_factory=list)  # 규칙별 상세

    @property
    def emoji(self) -> str:
        return {"매수 관심": "🟢", "매도 관심": "🔴"}.get(self.stance, "🟡")


def _verdict_label(v: int) -> str:
    return {1: "매수", -1: "매도", 0: "중립"}[v]


def evaluate(df: pd.DataFrame) -> SignalReport:
    """최신 봉 기준으로 각 규칙을 평가해 종합 신호를 만든다."""
    close = df["Close"].astype("float64")
    rows: list[dict] = []

    def add(rule: str, verdict: int, detail: str) -> None:
        rows.append({"규칙": rule, "판정": _verdict_label(verdict), "근거": detail, "_v": verdict})

    # --- 1. 이동평균 교차 (최근 5봉 이내) -----------------------------
    crosses = detect_crosses(df, short=20, long=60)
    if not crosses.empty and len(df) >= 5:
        last_cross = crosses.iloc[-1]
        recent_idx = df.index[-5:]
        if crosses.index[-1] in recent_idx:
            if last_cross["type"] == "golden":
                add("골든크로스(20/60)", 1, "최근 단기 이평이 장기 이평을 상향 돌파")
            else:
                add("데드크로스(20/60)", -1, "최근 단기 이평이 장기 이평을 하향 돌파")
        else:
            add("이평 교차", 0, "최근 5봉 내 교차 없음")
    else:
        add("이평 교차", 0, "교차 데이터 부족")

    # --- 2. RSI ------------------------------------------------------
    r = indicators.rsi(close, 14)
    if r.notna().any():
        rv = float(r.dropna().iloc[-1])
        if rv < 30:
            add("RSI(14)", 1, f"RSI {rv:.1f} — 과매도(반등 관심)")
        elif rv > 70:
            add("RSI(14)", -1, f"RSI {rv:.1f} — 과매수(과열 주의)")
        else:
            add("RSI(14)", 0, f"RSI {rv:.1f} — 중립 구간")
    else:
        add("RSI(14)", 0, "데이터 부족")

    # --- 3. MACD 히스토그램 부호/전환 --------------------------------
    m = indicators.macd(close)
    hist = m["hist"].dropna()
    if len(hist) >= 2:
        cur, prev = float(hist.iloc[-1]), float(hist.iloc[-2])
        if prev <= 0 < cur:
            add("MACD", 1, "MACD가 시그널선을 상향 돌파(골든)")
        elif prev >= 0 > cur:
            add("MACD", -1, "MACD가 시그널선을 하향 돌파(데드)")
        elif cur > 0:
            add("MACD", 1, "MACD가 시그널선 위(상승 모멘텀)")
        else:
            add("MACD", -1, "MACD가 시그널선 아래(하락 모멘텀)")
    else:
        add("MACD", 0, "데이터 부족")

    # --- 4. 볼린저밴드 위치 ------------------------------------------
    bb = indicators.bollinger(close, 20, 2.0)
    if bb["upper"].notna().any():
        price = float(close.iloc[-1])
        upper = float(bb["upper"].dropna().iloc[-1])
        lower = float(bb["lower"].dropna().iloc[-1])
        if price < lower:
            add("볼린저밴드", 1, "하단 밴드 이탈 — 단기 과매도(반등 관심)")
        elif price > upper:
            add("볼린저밴드", -1, "상단 밴드 이탈 — 단기 과열(주의)")
        else:
            add("볼린저밴드", 0, "밴드 내부에서 움직임")
    else:
        add("볼린저밴드", 0, "데이터 부족")

    # --- 종합 --------------------------------------------------------
    score = sum(row["_v"] for row in rows)
    if score >= 2:
        stance = "매수 관심"
    elif score <= -2:
        stance = "매도 관심"
    else:
        stance = "중립"

    # 내부 판정 값(_v)은 표에서 숨김
    for row in rows:
        row.pop("_v", None)

    return SignalReport(stance=stance, score=score, rows=rows)


def signals_dataframe(report: SignalReport) -> pd.DataFrame:
    """SignalReport를 표시용 DataFrame으로 변환."""
    return pd.DataFrame(report.rows, columns=["규칙", "판정", "근거"])
