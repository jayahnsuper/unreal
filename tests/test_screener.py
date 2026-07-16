"""종목 스크리너 로직 단위 테스트.

네트워크 없이 합성 데이터와 주입 fetch로 필터링·랭킹·예외 격리를 검증한다.
(이 샌드박스는 라이브 시세 조회가 막혀 있으므로 모든 검증은 합성 데이터로 한다.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import screener
from src.screener import COLUMNS, analyze_ticker, passes_criteria, screen


def _ohlcv_from_close(close: list[float]) -> pd.DataFrame:
    """종가 리스트로부터 간단한 OHLCV DataFrame을 만든다 (테스트용)."""
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    s = pd.Series(close, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "Open": s.shift(1).fillna(s),
            "High": s * 1.01,
            "Low": s * 0.99,
            "Close": s,
            "Volume": 1000,
        }
    )


# 시나리오별 합성 종가 -------------------------------------------------------
def _up_df() -> pd.DataFrame:
    return _ohlcv_from_close(list(np.linspace(100, 220, 180)))  # 꾸준한 상승


def _down_df() -> pd.DataFrame:
    return _ohlcv_from_close(list(np.linspace(220, 100, 180)))  # 꾸준한 하락


def _flat_df() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return _ohlcv_from_close(list(150 + rng.normal(scale=0.5, size=180)))  # 횡보


def _golden_df() -> pd.DataFrame:
    # 하락 후 상승 전환 → 최근 구간에 골든크로스 발생 (test_golden_cross_detected 패턴 재사용)
    close = list(np.linspace(200, 100, 100)) + list(np.linspace(100, 260, 40))
    return _ohlcv_from_close(close)


def _make_fetch(mapping: dict[str, pd.DataFrame]):
    """ticker→DataFrame 매핑으로 주입용 fetch(ticker, period, interval)를 만든다."""

    def _fetch(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        return mapping[ticker]

    return _fetch


# --------------------------- analyze_ticker --------------------------------
def test_analyze_ticker_keys_match_columns():
    metrics = analyze_ticker(_up_df(), "UP")
    assert set(metrics.keys()) == set(COLUMNS)
    assert metrics["티커"] == "UP"
    assert metrics["추세"] in {"상승", "하락", "횡보", "데이터 부족"}
    assert isinstance(metrics["신뢰도"], int)
    assert isinstance(metrics["신호점수"], int)
    assert isinstance(metrics["추세점수"], float)
    assert isinstance(metrics["최근골든"], bool)
    assert isinstance(metrics["above_sma20"], bool)


def test_analyze_ticker_rsi_nan_safe():
    # 봉이 너무 적어 RSI(14)가 전부 NaN인 경우에도 예외 없이 NaN 반환.
    metrics = analyze_ticker(_ohlcv_from_close([100.0, 101.0, 102.0]), "TINY")
    assert isinstance(metrics["RSI"], float)
    assert np.isnan(metrics["RSI"])


# --------------------------- passes_criteria -------------------------------
def test_passes_criteria_none_and_empty_always_true():
    metrics = analyze_ticker(_up_df(), "UP")
    assert passes_criteria(metrics, None) is True
    assert passes_criteria(metrics, {}) is True


def test_passes_criteria_rsi_nan_fails_when_ranged():
    metrics = analyze_ticker(_ohlcv_from_close([100.0, 101.0, 102.0]), "TINY")
    assert np.isnan(metrics["RSI"])
    assert passes_criteria(metrics, {"rsi_min": 0, "rsi_max": 100}) is False


# --------------------------- screen: 스키마 & 정렬 -------------------------
def test_screen_schema_and_sorted_by_trend_score():
    fetch = _make_fetch({"UP": _up_df(), "DOWN": _down_df(), "FLAT": _flat_df()})
    result = screen(["UP", "DOWN", "FLAT"], criteria=None, fetch=fetch)

    # 고정 컬럼 스키마
    assert list(result.columns) == list(COLUMNS)
    # 통과 3종목 모두 포함
    assert len(result) == 3
    assert set(result["티커"]) == {"UP", "DOWN", "FLAT"}
    # '추세점수' 내림차순 정렬
    scores = result["추세점수"].tolist()
    assert scores == sorted(scores, reverse=True)
    # 상승 종목이 하락 종목보다 위에 있어야 함
    order = result["티커"].tolist()
    assert order.index("UP") < order.index("DOWN")


# --------------------------- screen: trend_up 필터 ------------------------
def test_screen_trend_up_filter():
    fetch = _make_fetch({"UP": _up_df(), "DOWN": _down_df(), "FLAT": _flat_df()})
    result = screen(["UP", "DOWN", "FLAT"], criteria={"trend_up": True}, fetch=fetch)

    assert not result.empty  # 상승 종목이 최소 1개는 통과
    assert (result["추세"] == "상승").all()
    assert "DOWN" not in set(result["티커"])


# --------------------------- screen: RSI 범위 필터 ------------------------
def test_screen_rsi_range_filters():
    fetch = _make_fetch({"UP": _up_df(), "DOWN": _down_df(), "FLAT": _flat_df()})

    wide = screen(["UP", "DOWN", "FLAT"], criteria={"rsi_min": 0, "rsi_max": 100}, fetch=fetch)
    # 유효 RSI 종목만 통과하고, 반환 RSI가 모두 범위 내
    assert not wide.empty
    assert wide["RSI"].notna().all()
    assert (wide["RSI"] >= 0).all() and (wide["RSI"] <= 100).all()

    narrow = screen(["UP", "DOWN", "FLAT"], criteria={"rsi_min": 99}, fetch=fetch)
    # 좁은 범위로 좁히면 통과 종목 수가 줄어든다
    assert len(narrow) <= len(wide)
    assert (narrow["RSI"] >= 99).all()


# --------------------------- screen: 예외 격리 ----------------------------
def test_screen_isolates_failing_ticker():
    def fetch(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        if ticker == "BOOM":
            raise RuntimeError("조회 실패 시뮬레이션")
        if ticker == "EMPTY":
            return _ohlcv_from_close([100.0])  # 봉 부족 → 제외
        return _up_df()

    result = screen(["UP", "BOOM", "EMPTY", "DOWN"], criteria=None, fetch=fetch)
    # 실패/부족 티커만 빠지고 정상 종목으로 결과 완성
    assert set(result["티커"]) == {"UP", "DOWN"}
    assert len(result) == 2


# --------------------------- screen: 최근 골든크로스 필터 -----------------
def test_screen_recent_golden_cross_filter():
    fetch = _make_fetch({"GOLD": _golden_df(), "UP": _up_df(), "FLAT": _flat_df()})
    result = screen(
        ["GOLD", "UP", "FLAT"],
        criteria={"recent_golden_cross": True},
        fetch=fetch,
    )
    tickers = set(result["티커"])
    # 골든크로스 발생 종목은 포함, 순수 상승/횡보 종목은 제외
    assert "GOLD" in tickers
    assert "UP" not in tickers
    assert "FLAT" not in tickers
    assert (result["최근골든"]).all()


# --------------------------- screen: 통과 0건 -----------------------------
def test_screen_empty_result_keeps_schema():
    fetch = _make_fetch({"DOWN": _down_df()})
    # 하락 종목만 있는데 상승만 필터 → 통과 0건
    result = screen(["DOWN"], criteria={"trend_up": True}, fetch=fetch)
    assert result.empty
    assert list(result.columns) == list(COLUMNS)


# --------------------------- screen: 컬럼-키 정합성 -----------------------
def test_screen_columns_equal_analyze_keys():
    fetch = _make_fetch({"UP": _up_df()})
    result = screen(["UP"], criteria=None, fetch=fetch)
    metrics = analyze_ticker(_up_df(), "UP")
    assert set(result.columns) == set(metrics.keys())
