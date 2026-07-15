"""추세 판정 모듈 — 앱의 핵심 기능.

여러 근거(이동평균 배열, 가격 위치, 단기 이평 기울기)를 -1..+1 점수로 종합해
**상승 / 하락 / 횡보** 라벨과 신뢰도(0~100), 그리고 사람이 읽을 근거 문장을 만든다.
또한 골든크로스/데드크로스 발생 시점을 탐지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators

# 단기 / 중기 / 장기 이동평균 기본값 (일봉 기준)
DEFAULT_WINDOWS = (20, 60, 120)


@dataclass
class TrendResult:
    """추세 판정 결과."""

    label: str  # "상승" | "하락" | "횡보" | "데이터 부족"
    score: float  # -1.0(강한 하락) ~ +1.0(강한 상승)
    confidence: int  # 0~100, 라벨에 대한 신뢰도
    reasons: list[str] = field(default_factory=list)
    windows: tuple[int, ...] = DEFAULT_WINDOWS

    @property
    def emoji(self) -> str:
        return {"상승": "🟢", "하락": "🔴", "횡보": "🟡"}.get(self.label, "⚪")


def _fit_windows(n: int, windows: tuple[int, ...]) -> tuple[int, ...]:
    """데이터 길이에 맞춰 사용할 이동평균 기간을 추린다.

    가장 짧은 기간조차 채우지 못하면 빈 튜플을 반환한다.
    """
    usable = tuple(w for w in windows if w <= n)
    return usable


def _slope_score(series: pd.Series, lookback: int = 20) -> float:
    """단기 이평의 최근 기울기를 -1..+1로 정규화.

    최근 ``lookback`` 봉을 선형회귀해 봉당 변화율을 구하고, 부드럽게 클리핑한다.
    """
    seg = series.dropna().tail(lookback)
    if len(seg) < 3:
        return 0.0
    x = np.arange(len(seg), dtype="float64")
    y = seg.to_numpy(dtype="float64")
    slope = np.polyfit(x, y, 1)[0]
    mean = float(np.mean(y))
    if mean == 0:
        return 0.0
    # 봉당 변화율(%)을 tanh로 눌러 -1..+1 범위로
    pct_per_bar = slope / mean
    return float(np.tanh(pct_per_bar * 100))


def classify_trend(
    df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> TrendResult:
    """OHLCV DataFrame으로부터 현재 추세를 판정한다.

    세 가지 근거를 -1..+1 점수로 합산해 평균한다.
      1. 이동평균 배열 (정배열↑ / 역배열↓)
      2. 종가가 각 이동평균 위/아래에 있는 비율
      3. 단기 이동평균의 기울기
    """
    close = df["Close"].astype("float64")
    n = len(close)
    usable = _fit_windows(n, windows)

    if len(usable) < 2:
        return TrendResult(
            label="데이터 부족",
            score=0.0,
            confidence=0,
            reasons=[
                f"판정에 필요한 데이터가 부족합니다 (현재 {n}봉). "
                f"기간을 늘리거나 더 오래된 종목을 선택하세요."
            ],
            windows=usable,
        )

    mas = {w: indicators.sma(close, w).iloc[-1] for w in usable}
    price = float(close.iloc[-1])
    reasons: list[str] = []

    # 상대 데드밴드: 가격 대비 이 정도(2%) 차이는 나야 '의미 있는' 차이로 본다.
    # 이렇게 하면 이평이 거의 붙어 있는 횡보장을 정/역배열로 오판하지 않는다.
    band = price * 0.02 if price else 1.0

    # --- 1. 이동평균 배열 점수 (연속값) ------------------------------
    ma_values = [mas[w] for w in usable]  # 단기→장기 순
    pair_scores = [
        max(-1.0, min(1.0, (ma_values[i] - ma_values[i + 1]) / band))
        for i in range(len(ma_values) - 1)
    ]
    alignment = float(np.mean(pair_scores)) if pair_scores else 0.0
    if alignment > 0.5:
        reasons.append(
            f"이동평균이 정배열에 가깝습니다 ({' > '.join(f'{w}일' for w in usable)}) → 상승 우호적"
        )
    elif alignment < -0.5:
        reasons.append(
            f"이동평균이 역배열에 가깝습니다 ({' < '.join(f'{w}일' for w in usable)}) → 하락 우호적"
        )
    else:
        reasons.append("이동평균이 서로 붙어 있습니다 (혼조) → 방향성 불명확")

    # --- 2. 가격 위치 점수 (연속값) ----------------------------------
    pos_scores = [max(-1.0, min(1.0, (price - mas[w]) / band)) for w in usable]
    position = float(np.mean(pos_scores))
    above = sum(1 for w in usable if price > mas[w])
    if above == len(usable):
        reasons.append(f"종가가 모든 이동평균({', '.join(f'{w}일' for w in usable)}) 위에 있습니다")
    elif above == 0:
        reasons.append("종가가 모든 이동평균 아래에 있습니다")
    else:
        reasons.append(f"종가가 {len(usable)}개 이평 중 {above}개 위에 있습니다")

    # --- 3. 단기 이평 기울기 점수 ------------------------------------
    short_ma = indicators.sma(close, usable[0])
    slope = _slope_score(short_ma, lookback=min(20, n))
    if slope > 0.15:
        reasons.append(f"단기({usable[0]}일) 이동평균이 상승 기울기입니다")
    elif slope < -0.15:
        reasons.append(f"단기({usable[0]}일) 이동평균이 하락 기울기입니다")
    else:
        reasons.append(f"단기({usable[0]}일) 이동평균이 거의 횡보(완만)합니다")

    # --- 종합 --------------------------------------------------------
    score = float(np.mean([alignment, position, slope]))
    score = max(-1.0, min(1.0, score))

    if score > 0.25:
        label = "상승"
        confidence = int(round(min(1.0, score) * 100))
    elif score < -0.25:
        label = "하락"
        confidence = int(round(min(1.0, abs(score)) * 100))
    else:
        label = "횡보"
        # 0에 가까울수록 횡보 신뢰도가 높다
        confidence = int(round((1 - abs(score) / 0.25) * 100))

    return TrendResult(
        label=label,
        score=round(score, 3),
        confidence=confidence,
        reasons=reasons,
        windows=usable,
    )


def detect_crosses(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 60,
) -> pd.DataFrame:
    """골든크로스/데드크로스 발생 시점을 찾는다.

    Returns
    -------
    컬럼 ``type``('golden'|'dead'), ``price``, ``short_ma``, ``long_ma`` 을 가진
    DataFrame. 인덱스는 교차가 발생한 날짜. 교차가 없으면 빈 DataFrame.
    """
    close = df["Close"].astype("float64")
    s = indicators.sma(close, short)
    l = indicators.sma(close, long)

    diff = s - l
    prev = diff.shift(1)

    golden = (prev <= 0) & (diff > 0)
    dead = (prev >= 0) & (diff < 0)

    events = []
    for idx in df.index[golden.fillna(False)]:
        events.append((idx, "golden"))
    for idx in df.index[dead.fillna(False)]:
        events.append((idx, "dead"))

    if not events:
        return pd.DataFrame(columns=["type", "price", "short_ma", "long_ma"])

    events.sort(key=lambda e: e[0])
    out = pd.DataFrame(
        {
            "type": [t for _, t in events],
            "price": [float(close.loc[d]) for d, _ in events],
            "short_ma": [float(s.loc[d]) for d, _ in events],
            "long_ma": [float(l.loc[d]) for d, _ in events],
        },
        index=[d for d, _ in events],
    )
    out.index.name = "Date"
    return out
