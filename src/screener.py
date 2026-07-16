"""종목 스크리너 모듈.

여러 종목의 OHLCV 시계열을 일괄로 조회·평가해 추세/신호 기준으로
**필터링·랭킹**하는 표(DataFrame)를 만든다.

핵심 설계
---------
- 계산은 기존 로직을 그대로 재사용한다: ``trend.classify_trend`` / ``trend.detect_crosses`` /
  ``signals.evaluate`` / ``indicators.rsi`` / ``indicators.sma``.
- 데이터 조회(``fetch``)는 주입 가능하게 하여 네트워크 없이도 테스트/데모가 된다.
  ``fetch`` 를 넘기지 않으면 라이브 경로(``data.fetch_ohlcv``)를 쓴다.
- ``analyze_ticker`` 가 만드는 dict 키 집합과 ``screen`` 결과 DataFrame 컬럼 집합은
  항상 일치한다(단일 종목 평가와 표 생성의 정합성 보장).

주의: 스크리너 결과는 기술적 조건 기반 참고용이며 매매 지시가 아니다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import indicators, signals
from .data import fetch_ohlcv, normalize_ticker
from .trend import DEFAULT_WINDOWS, classify_trend, detect_crosses

# 결과 DataFrame의 고정 컬럼 스키마(= analyze_ticker 반환 dict의 키 순서).
# 이 순서/집합이 곧 표의 컬럼이 되므로 두 곳이 항상 일치해야 한다.
COLUMNS: tuple[str, ...] = (
    "티커",
    "추세",
    "신뢰도",
    "RSI",
    "신호",
    "신호점수",
    "추세점수",
    "최근골든",
    "above_sma20",
)

# '최근골든' 판정에 쓰는 "최근" 봉 수. 최근 이 봉 수 이내에 발생하고
# 그 뒤로 데드크로스가 없는(=현재 골든 국면) 골든크로스만 True로 본다.
RECENT_GOLDEN_BARS: int = 30

# above_sma20 판정에 쓰는 SMA 기간(종가가 이 기간 SMA 위인지).
ABOVE_SMA_PERIOD: int = 20


def _latest_valid(series: pd.Series) -> float:
    """Series의 마지막 유효(비-NaN) 값을 float로 반환. 없으면 NaN."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


def _recent_golden_cross(
    df: pd.DataFrame,
    short: int,
    long: int,
    recent_bars: int = RECENT_GOLDEN_BARS,
) -> bool:
    """최근 ``recent_bars`` 봉 이내에 발생한 골든크로스가 마지막 교차인지 판정.

    가장 최근 교차가 골든이고 그 시점이 최근 구간 안에 있으면 True.
    (그 뒤 데드크로스가 있으면 마지막 교차가 데드이므로 자동으로 False가 된다.)
    """
    try:
        crosses = detect_crosses(df, short=short, long=long)
    except Exception:  # noqa: BLE001 - 개별 종목 계산 실패는 안전하게 무시
        return False
    if crosses.empty:
        return False
    last_type = crosses["type"].iloc[-1]
    if last_type != "golden":
        return False
    last_idx = crosses.index[-1]
    try:
        pos = df.index.get_loc(last_idx)
    except KeyError:
        return False
    bars_from_end = (len(df) - 1) - int(pos)
    return bars_from_end <= recent_bars


def analyze_ticker(
    df: pd.DataFrame,
    ticker: str,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict:
    """단일 종목 OHLCV를 받아 지표/추세/신호를 종합한 지표 dict를 반환한다.

    네트워크 불필요·예외 없이 안전하게 동작한다. 데이터 부족/NaN도 방어적으로 처리한다.

    Returns
    -------
    dict
        키는 ``COLUMNS`` 와 동일: '티커','추세','신뢰도','RSI','신호','신호점수',
        '추세점수','최근골든','above_sma20'. RSI가 전부 NaN이면 ``float('nan')``.
    """
    close = df["Close"].astype("float64")

    # --- 추세: 라벨 / 신뢰도 / score(-1..1) ---
    trend = classify_trend(df, windows=windows)

    # --- 신호: 스탠스 / 점수 ---
    report = signals.evaluate(df)

    # --- RSI(14) 최신값 (전부 NaN이면 NaN) ---
    rsi_val = _latest_valid(indicators.rsi(close, 14))

    # --- 최근 골든크로스 (windows의 단기/장기 사용) ---
    short = windows[0] if len(windows) >= 1 else 20
    long = windows[1] if len(windows) >= 2 else 60
    recent_golden = _recent_golden_cross(df, short=short, long=long)

    # --- 종가가 SMA20 위인지 (데이터 부족/NaN이면 False) ---
    sma20 = _latest_valid(indicators.sma(close, ABOVE_SMA_PERIOD))
    price = _latest_valid(close)
    above_sma20 = bool(
        not np.isnan(sma20) and not np.isnan(price) and price > sma20
    )

    return {
        "티커": ticker,
        "추세": trend.label,
        "신뢰도": int(trend.confidence),
        "RSI": float(rsi_val),
        "신호": report.stance,
        "신호점수": int(report.score),
        "추세점수": float(trend.score),
        "최근골든": bool(recent_golden),
        "above_sma20": above_sma20,
    }


def passes_criteria(metrics: dict, criteria: dict | None) -> bool:
    """metrics dict가 필터 criteria를 **모두(AND)** 만족하는지 반환.

    지원 키
    -------
    - ``trend_up`` (bool)          : True면 추세=='상승'만 통과
    - ``recent_golden_cross`` (bool): True면 최근 골든크로스 발생 종목만 통과
    - ``rsi_min`` (float)          : RSI >= 값 (RSI가 NaN이면 탈락)
    - ``rsi_max`` (float)          : RSI <= 값 (RSI가 NaN이면 탈락)
    - ``above_sma`` (int 기간)     : 종가가 해당 기간 SMA 위. 현재 analyze_ticker는
      ``above_sma20`` 만 계산하므로 기간 20의 판정을 사용한다(다른 기간은
      대응 컬럼이 있으면 그 값을, 없으면 above_sma20을 근거로 한다).

    criteria가 None이거나 빈 dict면 항상 True(전부 통과).
    """
    if not criteria:
        return True

    # --- 추세 상승만 ---
    if criteria.get("trend_up"):
        if metrics.get("추세") != "상승":
            return False

    # --- 최근 골든크로스 ---
    if criteria.get("recent_golden_cross"):
        if not metrics.get("최근골든", False):
            return False

    # --- RSI 범위 (NaN은 무조건 탈락) ---
    rsi_min = criteria.get("rsi_min")
    rsi_max = criteria.get("rsi_max")
    if rsi_min is not None or rsi_max is not None:
        rsi_val = metrics.get("RSI", float("nan"))
        if rsi_val is None or (isinstance(rsi_val, float) and np.isnan(rsi_val)):
            return False
        if rsi_min is not None and rsi_val < float(rsi_min):
            return False
        if rsi_max is not None and rsi_val > float(rsi_max):
            return False

    # --- SMA 상향(종가가 SMA 위) ---
    above = criteria.get("above_sma")
    if above is not None:
        key = f"above_sma{int(above)}"
        val = metrics.get(key, metrics.get("above_sma20"))
        if not val:
            return False

    return True


def screen(
    tickers: list[str],
    criteria: dict | None = None,
    fetch: Callable[..., pd.DataFrame] | None = None,
    period: str = "1y",
    interval: str = "1d",
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    sort_by: str = "추세점수",
) -> pd.DataFrame:
    """여러 종목을 조회·평가해 랭킹 표(DataFrame)를 반환하는 진입점.

    Parameters
    ----------
    tickers : 조회할 티커 목록.
    criteria : 필터 조건 dict(``passes_criteria`` 참조). None이면 전부 통과.
    fetch : ``fetch(ticker, period, interval) -> DataFrame`` 시그니처의 조회 함수.
        None이면 ``data.fetch_ohlcv``(라이브 경로)를 쓴다.  # 사용자 PC에서 검증 필요
        테스트/데모에서는 sample_data 기반 주입 fetch로 네트워크 없이 검증한다.
    period, interval : fetch에 전달할 기간/주기.
    windows : 추세/교차 계산에 쓸 이동평균 기간(기본 DEFAULT_WINDOWS).
    sort_by : 정렬 기준 컬럼(기본 '추세점수'). 내림차순 정렬.

    Returns
    -------
    pd.DataFrame
        criteria를 통과한 종목만 담아 ``sort_by`` 내림차순 정렬 후 index를 리셋.
        통과 종목이 없으면 ``COLUMNS`` 스키마의 빈 DataFrame.

    각 티커의 조회/계산 예외는 개별 격리되어 해당 종목만 건너뛰고 계속한다.
    """
    fetcher = fetch if fetch is not None else fetch_ohlcv  # 사용자 PC에서 검증 필요

    rows: list[dict] = []
    for raw in tickers:
        try:
            ticker = normalize_ticker(str(raw))
            if not ticker:
                continue
            df = fetcher(ticker, period, interval)
            # 조회 실패/데이터 부족은 조용히 건너뛴다(결과에서 제외).
            if df is None or len(df) < 2:
                continue
            metrics = analyze_ticker(df, ticker, windows=windows)
            if passes_criteria(metrics, criteria):
                rows.append(metrics)
        except Exception:  # noqa: BLE001 - 개별 종목 실패는 격리하고 계속
            continue

    result = pd.DataFrame(rows, columns=list(COLUMNS))
    if not result.empty and sort_by in result.columns:
        # mergesort: 동점일 때 입력 순서를 보존(안정 정렬)
        result = result.sort_values(sort_by, ascending=False, kind="mergesort")
    return result.reset_index(drop=True)
