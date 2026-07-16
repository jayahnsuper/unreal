"""지표·추세·신호·지지저항 로직 단위 테스트.

네트워크 없이 합성 데이터로 계산식의 정확성과 로직의 방향성을 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import indicators, levels, signals
from src.data import normalize_ticker, sample_data
from src.trend import classify_trend, detect_crosses


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


# --------------------------- SMA / EMA ---------------------------------
def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype="float64")
    result = indicators.sma(s, 3)
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert result.iloc[4] == pytest.approx(4.0)  # (3+4+5)/3


def test_ema_first_valid_equals_sma_seed():
    s = pd.Series([1, 2, 3, 4, 5], dtype="float64")
    ema = indicators.ema(s, 3)
    # min_periods=3 이므로 첫 두 값은 NaN, 3번째부터 값 존재
    assert pd.isna(ema.iloc[0])
    assert not pd.isna(ema.iloc[2])


def test_ema_constant_series_is_constant():
    s = pd.Series([7.0] * 10)
    ema = indicators.ema(s, 4).dropna()
    assert np.allclose(ema.to_numpy(), 7.0)


# --------------------------- RSI ---------------------------------------
def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 30, dtype="float64"))  # 계속 상승
    r = indicators.rsi(s, 14).dropna()
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    s = pd.Series(np.arange(30, 1, -1, dtype="float64"))  # 계속 하락
    r = indicators.rsi(s, 14).dropna()
    assert r.iloc[-1] == pytest.approx(0.0)


def test_rsi_within_bounds():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(size=200)))
    r = indicators.rsi(s, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


# --------------------------- MACD --------------------------------------
def test_macd_columns_and_hist_relation():
    s = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(size=100)))
    m = indicators.macd(s)
    assert list(m.columns) == ["macd", "signal", "hist"]
    valid = m.dropna()
    assert np.allclose((valid["macd"] - valid["signal"]).to_numpy(),
                       valid["hist"].to_numpy())


# --------------------------- Bollinger ---------------------------------
def test_bollinger_ordering_and_width():
    s = pd.Series(100 + np.cumsum(np.random.default_rng(2).normal(size=100)))
    bb = indicators.bollinger(s, 20, 2.0).dropna()
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()


def test_bollinger_constant_series_zero_width():
    s = pd.Series([50.0] * 30)
    bb = indicators.bollinger(s, 20, 2.0).dropna()
    assert np.allclose(bb["upper"].to_numpy(), bb["lower"].to_numpy())


# --------------------------- ATR ---------------------------------------
def test_atr_positive():
    df = _ohlcv_from_close(list(100 + np.cumsum(np.random.default_rng(3).normal(size=100))))
    a = indicators.atr(df, 14).dropna()
    assert (a > 0).all()


# --------------------------- Trend -------------------------------------
def test_trend_uptrend_detected():
    close = list(np.linspace(100, 200, 150))  # 꾸준한 상승
    df = _ohlcv_from_close(close)
    result = classify_trend(df)
    assert result.label == "상승"
    assert result.score > 0
    assert result.confidence > 0
    assert result.reasons


def test_trend_downtrend_detected():
    close = list(np.linspace(200, 100, 150))  # 꾸준한 하락
    df = _ohlcv_from_close(close)
    result = classify_trend(df)
    assert result.label == "하락"
    assert result.score < 0


def test_trend_sideways_detected():
    rng = np.random.default_rng(4)
    close = list(150 + rng.normal(scale=0.5, size=150))  # 좁은 박스권
    df = _ohlcv_from_close(close)
    result = classify_trend(df)
    assert result.label == "횡보"


def test_trend_insufficient_data():
    # 8봉 미만은 정말로 데이터 부족
    df = _ohlcv_from_close([100, 101, 102])
    result = classify_trend(df)
    assert result.label == "데이터 부족"


def test_trend_short_data_uses_shortened_windows():
    # 표준 이평(20/60/120)은 못 들어가지만 8봉 이상이면 축소 이평으로 판정한다
    close = list(np.linspace(100, 140, 25))  # 25봉 꾸준한 상승
    df = _ohlcv_from_close(close)
    result = classify_trend(df)
    assert result.label != "데이터 부족"
    assert len(result.windows) == 2  # 축소된 (short, long)
    # 축소 판정 안내 문구가 포함된다
    assert any("축소 이평" in r for r in result.reasons)


# --------------------------- Crosses -----------------------------------
def test_golden_cross_detected():
    # 하락 후 상승으로 전환 → 골든크로스가 존재해야 함
    close = list(np.linspace(200, 100, 80)) + list(np.linspace(100, 260, 80))
    df = _ohlcv_from_close(close)
    crosses = detect_crosses(df, short=20, long=60)
    assert not crosses.empty
    assert "golden" in set(crosses["type"])


# --------------------------- Signals -----------------------------------
def test_signals_report_structure():
    df = _ohlcv_from_close(list(np.linspace(100, 200, 150)))
    report = signals.evaluate(df)
    assert report.stance in {"매수 관심", "중립", "매도 관심"}
    assert isinstance(report.score, int)
    table = signals.signals_dataframe(report)
    assert list(table.columns) == ["규칙", "판정", "근거"]
    assert len(table) >= 4


# --------------------------- Trade score -------------------------------
def test_trade_score_bounds_and_labels():
    up = _ohlcv_from_close(list(np.linspace(100, 220, 160)))
    down = _ohlcv_from_close(list(np.linspace(220, 100, 160)))
    ts_up = signals.trade_score(up)
    ts_down = signals.trade_score(down)

    # 0~100 범위
    assert 0 <= ts_up.score <= 100
    assert 0 <= ts_down.score <= 100
    # 방향성: 상승장 점수 > 하락장 점수, 그리고 각각 매수/매도 쪽
    assert ts_up.score > ts_down.score
    assert ts_up.label in {"매수", "강한 매수"}
    assert ts_down.label in {"매도", "강한 매도"}


def test_trade_score_sideways_is_neutral():
    rng = np.random.default_rng(7)
    flat = _ohlcv_from_close(list(150 + rng.normal(scale=0.5, size=160)))
    ts = signals.trade_score(flat)
    assert 40 <= ts.score <= 60
    assert ts.label == "중립"


# --------------------------- Sell timing -------------------------------
def test_sell_timing_structure_and_bounds():
    df = _ohlcv_from_close(list(np.linspace(100, 200, 160)))
    stiming = signals.sell_timing(df)
    assert 0 <= stiming.urgency <= 100
    assert stiming.action in {"매도 고려", "관망", "보유 지속"}
    # 최근 고점 기반 추적 손절가는 현재가보다 낮아야 한다
    assert stiming.trailing_stop is None or stiming.trailing_stop < float(df["Close"].iloc[-1])


def test_sell_timing_stop_loss_from_entry():
    df = _ohlcv_from_close(list(np.linspace(100, 200, 160)))
    stiming = signals.sell_timing(df, entry_price=190.0)
    # 매수단가 입력 시 손절가 = 매수단가 - 1.5·ATR < 매수단가
    assert stiming.stop_loss is not None
    assert stiming.stop_loss < 190.0


def test_sell_timing_overbought_flags_sell():
    # 강한 상승 지속 → RSI 과매수 → 차익실현(매도) 신호가 잡혀야 한다
    up = _ohlcv_from_close(list(np.linspace(100, 220, 160)))
    stiming = signals.sell_timing(up)
    assert stiming.urgency >= 25
    assert any(("과매수" in r) or ("RSI" in r) for r in stiming.reasons)


def test_sell_timing_calm_hold():
    # 잔잔한 횡보 → 뚜렷한 매도 신호 없음 → 보유 지속 쪽
    rng = np.random.default_rng(11)
    flat = _ohlcv_from_close(list(150 + rng.normal(scale=0.4, size=160)))
    stiming = signals.sell_timing(flat)
    assert stiming.action in {"보유 지속", "관망"}


# --------------------------- Levels ------------------------------------
def test_support_resistance_split_around_price():
    rng = np.random.default_rng(5)
    close = list(150 + 20 * np.sin(np.linspace(0, 12, 200)) + rng.normal(size=200))
    df = _ohlcv_from_close(close)
    lv = levels.support_resistance(df)
    price = float(df["Close"].iloc[-1])
    assert all(s < price for s in lv["supports"])
    assert all(r > price for r in lv["resistances"])


# --------------------------- Data helpers ------------------------------
def test_normalize_ticker_korean_code_gets_ks_suffix():
    assert normalize_ticker("005930") == "005930.KS"
    assert normalize_ticker(" aapl ") == "AAPL"
    assert normalize_ticker("035720.KQ") == "035720.KQ"


def test_sample_data_shape_and_columns():
    df = sample_data(days=300)
    assert len(df) == 300
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # High/Low가 Open/Close를 감싸는지 (일봉 정합성)
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9).all()
    # 분석 파이프라인이 데모 데이터에서 정상 동작
    assert classify_trend(df).label in {"상승", "하락", "횡보"}
    assert signals.evaluate(df).stance in {"매수 관심", "중립", "매도 관심"}
