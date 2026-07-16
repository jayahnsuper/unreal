"""규칙 기반 매수/매도 신호 모듈.

여러 보조지표(이동평균 교차, RSI, MACD, 볼린저밴드)를 각각 +1(매수)·-1(매도)·0(중립)
으로 평가해 합산하고, 최신 시점의 종합 스탠스(매수 관심 / 중립 / 매도 관심)를 낸다.

주의: 규칙 기반 참고 신호일 뿐 매매 지시가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import indicators
from .trend import classify_trend, detect_crosses


@dataclass
class SignalReport:
    """최신 시점 종합 신호."""

    stance: str  # "매수 관심" | "중립" | "매도 관심"
    score: int  # 규칙 점수 합계 (양수=매수 우위)
    rows: list[dict] = field(default_factory=list)  # 규칙별 상세

    @property
    def emoji(self) -> str:
        return {"매수 관심": "🟢", "매도 관심": "🔴"}.get(self.stance, "🟡")


@dataclass
class TradeScore:
    """추세 + 규칙 신호를 합친 한눈 매매 점수(0~100)."""

    score: int  # 0(강한 매도) ~ 50(중립) ~ 100(강한 매수)
    label: str  # "강한 매수" | "매수" | "중립" | "매도" | "강한 매도"
    trend_score: float  # 추세 점수 -1..+1 (참고용)
    signal_score: int  # 규칙 점수 합계 (참고용)

    @property
    def emoji(self) -> str:
        return {
            "강한 매수": "🟢",
            "매수": "🟢",
            "매도": "🔴",
            "강한 매도": "🔴",
        }.get(self.label, "🟡")


@dataclass
class SellTiming:
    """보유 포지션의 매도 타이밍 판정(단타 보조)."""

    urgency: int  # 0(보유 지속) ~ 100(즉시 매도 고려)
    action: str  # "매도 고려" | "관망" | "보유 지속"
    reasons: list[str] = field(default_factory=list)
    stop_loss: float | None = None  # 매수단가 기준 ATR 손절가(입력 시)
    trailing_stop: float | None = None  # 최근 고점 - 2·ATR (추적 손절)
    target: float | None = None  # 가장 가까운 저항(목표가 참고)

    @property
    def emoji(self) -> str:
        return {"매도 고려": "🔴", "관망": "🟡"}.get(self.action, "🟢")


def sell_timing(
    df: pd.DataFrame,
    entry_price: float | None = None,
    windows: tuple[int, int] = (20, 60),
) -> SellTiming:
    """보유 중인 종목의 매도 타이밍을 판정한다(단타 보조 · 참고용).

    과매수(RSI)·데드크로스·MACD 하락전환·볼린저 상단 이탈·저항 근접을 점수화하고,
    ATR 기반 손절가/추적 손절가와 가장 가까운 저항(목표가)을 제안한다.

    ⚠️ 참고용이며 매매 지시가 아니다. 손절/목표가는 계산 예시일 뿐이다.
    """
    from .levels import support_resistance
    from .trend import detect_crosses

    close = df["Close"].astype("float64")
    price = float(close.iloc[-1])
    reasons: list[str] = []
    score = 0

    # --- RSI 과매수 ---
    r = indicators.rsi(close, 14).dropna()
    if len(r):
        rv = float(r.iloc[-1])
        if rv >= 70:
            score += 25
            reasons.append(f"RSI {rv:.0f} — 과매수권(차익실현 고려)")

    # --- 데드크로스(최근 5봉 이내) ---
    try:
        crosses = detect_crosses(df, short=windows[0], long=windows[1])
        if not crosses.empty and crosses["type"].iloc[-1] == "dead":
            pos = df.index.get_loc(crosses.index[-1])
            if (len(df) - 1) - int(pos) <= 5:
                score += 30
                reasons.append("최근 데드크로스 발생 — 단기 하락 전환 신호")
    except Exception:  # noqa: BLE001 - 개별 계산 실패는 무시
        pass

    # --- MACD 하락 전환 ---
    hist = indicators.macd(close)["hist"].dropna()
    if len(hist) >= 2:
        cur, prev = float(hist.iloc[-1]), float(hist.iloc[-2])
        if prev >= 0 > cur:
            score += 25
            reasons.append("MACD가 시그널선을 하향 돌파(하락 모멘텀 전환)")
        elif cur < 0:
            score += 10
            reasons.append("MACD가 시그널선 아래(하락 모멘텀)")

    # --- 볼린저 상단 이탈(과열) ---
    bb = indicators.bollinger(close, 20, 2.0)
    if bb["upper"].notna().any():
        upper = float(bb["upper"].dropna().iloc[-1])
        if price > upper:
            score += 15
            reasons.append("볼린저 상단 이탈 — 단기 과열")

    # --- 저항 근접 ---
    levels = support_resistance(df)
    resistances = [r for r in levels.get("resistances", []) if r >= price]
    target = min(resistances) if resistances else None
    if target is not None and (target - price) / price <= 0.01:
        score += 10
        reasons.append(f"저항선({target:,.2f}) 근접 — 돌파 실패 시 매도 압력")

    # --- ATR 기반 손절/추적 손절 ---
    atr_series = indicators.atr(df, 14).dropna()
    stop_loss = trailing_stop = None
    if len(atr_series):
        atr_val = float(atr_series.iloc[-1])
        recent_high = float(df["High"].astype("float64").tail(10).max())
        trailing_stop = recent_high - 2.0 * atr_val
        if entry_price is not None and entry_price > 0:
            stop_loss = entry_price - 1.5 * atr_val

    urgency = min(100, score)
    if urgency >= 55:
        action = "매도 고려"
    elif urgency >= 30:
        action = "관망"
    else:
        action = "보유 지속"
        if not reasons:
            reasons.append("뚜렷한 매도 신호가 없습니다 (추세·모멘텀 유지)")

    return SellTiming(
        urgency=urgency,
        action=action,
        reasons=reasons,
        stop_loss=stop_loss,
        trailing_stop=trailing_stop,
        target=target,
    )


def trade_score(df: pd.DataFrame) -> TradeScore:
    """0~100 매매 점수를 계산한다.

    추세 판정 점수(-1..+1, 가중 0.55)와 규칙 신호 점수(정규화 -1..+1, 가중 0.45)를
    합쳐 50(중립)을 기준으로 0~100 척도로 환산한다. 높을수록 매수 우위.

    ⚠️ 참고용 개인 도구다. 매매 지시나 수익 보장이 아니다.
    """
    trend = classify_trend(df)
    report = evaluate(df)

    # 규칙 점수를 규칙 개수로 나눠 -1..+1 로 정규화
    n_rules = len(report.rows) or 1
    signal_norm = report.score / n_rules

    combined = 0.55 * trend.score + 0.45 * signal_norm  # -1..+1
    score = int(round(50 + combined * 50))
    score = max(0, min(100, score))

    if score >= 75:
        label = "강한 매수"
    elif score >= 60:
        label = "매수"
    elif score > 40:
        label = "중립"
    elif score > 25:
        label = "매도"
    else:
        label = "강한 매도"

    return TradeScore(
        score=score,
        label=label,
        trend_score=trend.score,
        signal_score=report.score,
    )


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
