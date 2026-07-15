"""기술적 지표 계산 모듈.

외부 TA 라이브러리에 의존하지 않고 pandas/numpy로 직접 구현한다.
계산식이 코드에 그대로 드러나므로 단위 테스트로 정확성을 검증하기 쉽다.

모든 함수는 입력 Series/DataFrame과 같은 인덱스를 갖는 결과를 반환하며,
계산에 필요한 봉 수가 부족한 앞부분은 NaN으로 채워진다.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """단순 이동평균 (Simple Moving Average)."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """지수 이동평균 (Exponential Moving Average)."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """상대강도지수 (RSI) — Wilder 평활 방식.

    RSI = 100 - 100 / (1 + RS), RS = 평균상승폭 / 평균하락폭.
    0(약세)~100(강세). 통상 70 이상 과매수, 30 이하 과매도로 본다.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder 평활 = alpha(1/window)의 지수이동평균
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    # 하락이 전혀 없으면(avg_loss=0) RS=inf -> RSI=100 으로 명시 처리
    result = result.where(avg_loss != 0, 100.0)
    # 상승이 전혀 없으면 RSI=0
    result = result.where(avg_gain != 0, 0.0)
    # 계산 구간 이전은 NaN 유지
    result[avg_gain.isna() | avg_loss.isna()] = pd.NA
    return result.astype("float64")


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD 지표.

    반환: 컬럼 ``macd``, ``signal``, ``hist`` 를 가진 DataFrame.
    - macd  = EMA(fast) - EMA(slow)
    - signal = EMA(macd, signal)
    - hist  = macd - signal  (히스토그램)
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}
    )


def bollinger(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """볼린저 밴드.

    반환: 컬럼 ``mid``(중심선=SMA), ``upper``, ``lower`` 를 가진 DataFrame.
    """
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """평균 실질 변동폭 (Average True Range) — Wilder 방식.

    ``df`` 는 High/Low/Close 컬럼(대문자)을 포함해야 한다.
    지지/저항 밴드 폭이나 변동성 기반 손절폭 계산에 쓴다.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
