"""주가 데이터 조회 모듈 (yfinance 기반).

- 미국 티커(``AAPL``)와 한국 티커(``005930.KS`` / 코스닥은 ``.KQ``) 모두 지원.
- 결과 DataFrame은 컬럼을 ``Open/High/Low/Close/Volume`` 단일 레벨로 정규화한다.
- 캐싱: Streamlit 환경이면 ``st.cache_data`` 로, 아니면(테스트/CLI) 무캐시로 동작해
  streamlit 미설치 상태에서도 import 가능하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 기간(period) / 주기(interval) 선택지 — app.py 사이드바에서 재사용
VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
VALID_INTERVALS = ["1d", "1wk", "1mo"]

_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache(func):
    """Streamlit이 있으면 st.cache_data, 없으면 원본 함수를 그대로 반환."""
    try:  # pragma: no cover - 환경에 따라 분기
        import streamlit as st

        return st.cache_data(ttl=900, show_spinner=False)(func)
    except Exception:
        return func


def normalize_ticker(ticker: str) -> str:
    """티커 문자열 정리: 공백 제거 및 대문자화.

    숫자로만 이루어진 한국 종목코드에 시장 접미사가 없으면 KOSPI(.KS)를 기본 가정한다.
    (코스닥이면 사용자가 ``035720.KQ`` 처럼 직접 입력)
    """
    t = ticker.strip().upper()
    if t.isdigit() and len(t) == 6:
        return f"{t}.KS"
    return t


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance가 MultiIndex 컬럼을 줄 경우 단일 레벨로 평탄화."""
    if isinstance(df.columns, pd.MultiIndex):
        # (필드, 티커) 또는 (티커, 필드) 두 형태 모두 대비
        lvl0 = set(df.columns.get_level_values(0))
        if set(_REQUIRED_COLS).issubset(lvl0):
            df = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
        else:
            df = df.xs(df.columns.get_level_values(0)[0], axis=1, level=0)
    return df


@_cache
def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """티커의 OHLCV 시계열을 반환한다.

    Parameters
    ----------
    ticker : 예) ``"AAPL"``, ``"TSLA"``, ``"005930.KS"``, ``"005930"``
    period : ``VALID_PERIODS`` 중 하나
    interval : ``VALID_INTERVALS`` 중 하나

    Returns
    -------
    DatetimeIndex 를 가진 DataFrame (컬럼: Open/High/Low/Close/Volume).

    Raises
    ------
    ValueError : 티커가 비었거나, 데이터가 조회되지 않는 경우.
    """
    import yfinance as yf

    symbol = normalize_ticker(ticker)
    if not symbol:
        raise ValueError("티커를 입력하세요.")

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"'{ticker}' 데이터를 찾을 수 없습니다. 티커를 확인하세요. "
            f"(한국 종목은 6자리 코드 또는 코드.KS/.KQ 형식)"
        )

    df = _flatten_columns(df, symbol)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df[_REQUIRED_COLS].dropna(how="all")
    df.index.name = "Date"
    return df


def sample_data(days: int = 400, seed: int = 42) -> pd.DataFrame:
    """오프라인/데모용 합성 OHLCV 데이터.

    네트워크가 없는 환경에서도 UI를 확인할 수 있도록, 추세 + 노이즈로
    그럴듯한 가격 시계열을 만든다. **실제 시세가 아니다.**
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2024-12-31", periods=days, freq="B")
    # 완만한 상승 추세 + 사인 파동 + 랜덤워크
    trend = np.linspace(0, 0.6, days)
    wave = 0.12 * np.sin(np.linspace(0, 8 * np.pi, days))
    noise = np.cumsum(rng.normal(0, 0.015, days))
    close = 100 * np.exp(trend + wave + noise)

    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
    volume = rng.integers(1_000_000, 5_000_000, days)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df
