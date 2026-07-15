"""지지/저항 가격대 산출 모듈.

스윙 고점/저점(로컬 극값, pivot)을 찾아 가까운 값끼리 묶은 뒤,
현재가 아래는 지지선, 위는 저항선으로 분류한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pivots(series: pd.Series, window: int, kind: str) -> list[float]:
    """로컬 극값(스윙 고/저점)을 찾는다.

    ``kind='high'`` 이면 좌우 ``window`` 봉 중 최대인 지점,
    ``kind='low'`` 이면 최소인 지점의 값을 모은다.
    """
    values = series.to_numpy(dtype="float64")
    n = len(values)
    out: list[float] = []
    for i in range(window, n - window):
        segment = values[i - window : i + window + 1]
        center = values[i]
        if kind == "high" and center == segment.max():
            out.append(center)
        elif kind == "low" and center == segment.min():
            out.append(center)
    return out


def _cluster(levels: list[float], tolerance: float) -> list[float]:
    """서로 ``tolerance`` 비율 이내인 가격대를 하나로 묶어 평균값을 낸다."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters: list[list[float]] = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - clusters[-1][-1]) / clusters[-1][-1] <= tolerance:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    return [float(np.mean(c)) for c in clusters]


def support_resistance(
    df: pd.DataFrame,
    window: int = 10,
    tolerance: float = 0.02,
    max_levels: int = 3,
) -> dict[str, list[float]]:
    """현재가 기준 주요 지지/저항 가격대를 반환한다.

    Parameters
    ----------
    window : 스윙 고/저점 판별에 쓸 좌우 봉 수
    tolerance : 가까운 가격대를 묶는 허용 비율(2% 기본)
    max_levels : 각각 최대 몇 개까지 반환할지

    Returns
    -------
    ``{"supports": [...내림차순...], "resistances": [...오름차순...]}``
    (지지선은 현재가에 가까운 것부터, 저항선도 현재가에 가까운 것부터)
    """
    if len(df) < 2 * window + 1:
        return {"supports": [], "resistances": []}

    price = float(df["Close"].iloc[-1])

    highs = _pivots(df["High"].astype("float64"), window, "high")
    lows = _pivots(df["Low"].astype("float64"), window, "low")

    all_levels = _cluster(highs + lows, tolerance)

    supports = sorted([lv for lv in all_levels if lv < price], reverse=True)
    resistances = sorted([lv for lv in all_levels if lv > price])

    return {
        "supports": supports[:max_levels],
        "resistances": resistances[:max_levels],
    }
