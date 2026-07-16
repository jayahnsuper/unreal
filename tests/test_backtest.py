"""백테스트 모듈 단위 테스트.

네트워크 없이 손으로 구성한 종가 시계열로, 롱-온리 신호 매매 시뮬레이션의
정확성(수익률·승률·거래수·자산곡선·MDD·바이앤홀드)을 검증한다.
tests/test_indicators.py 의 _ohlcv_from_close 패턴을 그대로 따른다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest, indicators


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


# 하락 → 상승(고점) → 하락 으로 구성해 short=2,long=4 에서
# 골든(진입) 1회, 이어서 데드(청산) 1회가 명확히 발생하는 종가 시계열.
# 진입가 8.0, 청산가 18.0 (수익), 보유 구간 고점 26 → 청산 18 낙폭 포함.
_GOLDEN_CLOSE = [
    10, 10, 10, 10,          # 초기 평탄
    9, 8, 7, 6, 5,           # 하락
    6, 8, 11, 15, 20, 26,    # 반등(골든크로스 진입 구간)
    25, 22, 18, 13, 9, 6, 4  # 하락(데드크로스 청산 구간)
]


# --------------------------- 골든크로스 정확검증 --------------------------
def test_golden_cross_single_trade():
    from src.trend import detect_crosses

    df = _ohlcv_from_close(_GOLDEN_CLOSE)
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4)

    # 골든→데드 1쌍 = 완료 거래 1건
    assert result.num_trades == 1

    crosses = detect_crosses(df, short=2, long=4)
    golden_ts = crosses[crosses["type"] == "golden"].index[0]
    dead_ts = crosses[crosses["type"] == "dead"].index[-1]
    entry_close = float(df.loc[golden_ts, "Close"])
    exit_close = float(df.loc[dead_ts, "Close"])

    # 총수익률 == 청산종가/진입종가 - 1
    assert result.total_return == pytest.approx(exit_close / entry_close - 1.0)
    # 수익 거래이므로 승률 1.0
    assert entry_close < exit_close
    assert result.win_rate == pytest.approx(1.0)

    # 거래 시점이 detect_crosses의 golden/dead 발생 시점과 일치
    trade = result.trades[0]
    assert trade["entry_date"] == golden_ts
    assert trade["exit_date"] == dead_ts
    assert trade["entry_price"] == pytest.approx(entry_close)
    assert trade["exit_price"] == pytest.approx(exit_close)


# --------------------------- 자산곡선 정합성 ------------------------------
def test_equity_curve_consistency():
    df = _ohlcv_from_close(_GOLDEN_CLOSE)
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4, initial_capital=1.0)
    eq = result.equity_curve

    # 길이 == 데이터 길이, 인덱스 정렬
    assert len(eq) == len(df)
    assert list(eq.index) == list(df.index)

    # 첫 유효값 == 초기자본, 마지막 값 == 초기자본*(1+총수익)
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] == pytest.approx(1.0 * (1 + result.total_return))

    # 진입 이전(플랫) 구간은 값이 불변(=초기자본)
    trade = result.trades[0]
    pre = eq.loc[: trade["entry_date"]].iloc[:-1]  # 진입봉 직전까지
    assert np.allclose(pre.to_numpy(), 1.0)

    # 청산 이후(플랫) 구간은 값이 불변(=마지막 실현 자본)
    post = eq.loc[trade["exit_date"] :]
    assert np.allclose(post.to_numpy(), float(eq.iloc[-1]))


# --------------------------- RSI 전략 정확검증 ----------------------------
def test_rsi_single_trade():
    # 하락(RSI<30 진입) 후 상승(RSI>70 청산)으로 정확히 1회 왕복
    close = list(np.linspace(100, 70, 20)) + list(np.linspace(70, 140, 25))
    df = _ohlcv_from_close(close)

    result = backtest.run_backtest(df, "rsi")
    assert result.num_trades == 1

    # 손계산: 첫 RSI<30 진입봉, 그 이후 첫 RSI>70 청산봉을 직접 찾아 대조
    r = indicators.rsi(df["Close"].astype("float64"), 14)
    entry_ts = None
    exit_ts = None
    for ts, val in r.items():
        if pd.isna(val):
            continue
        if entry_ts is None and float(val) < 30.0:
            entry_ts = ts
        elif entry_ts is not None and exit_ts is None and float(val) > 70.0:
            exit_ts = ts
            break
    entry_close = float(df.loc[entry_ts, "Close"])
    exit_close = float(df.loc[exit_ts, "Close"])

    assert result.trades[0]["entry_date"] == entry_ts
    assert result.trades[0]["exit_date"] == exit_ts
    assert result.total_return == pytest.approx(exit_close / entry_close - 1.0)


# --------------------------- 바이앤홀드 / 초과수익 ------------------------
def test_buy_and_hold_and_excess():
    df = _ohlcv_from_close(_GOLDEN_CLOSE)
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4)

    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    assert result.buy_and_hold_return == pytest.approx(last_close / first_close - 1.0)
    assert result.excess_return == pytest.approx(
        result.total_return - result.buy_and_hold_return
    )


# --------------------------- MDD -----------------------------------------
def test_mdd_drawdown_matches_peak_to_trough():
    df = _ohlcv_from_close(_GOLDEN_CLOSE)
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4)
    trade = result.trades[0]

    # 진입~청산 구간의 자산곡선을 종가로 독립 재구성해 peak-to-trough 손검증.
    # (전량매매·초기자본 1.0 이므로 보유구간 자산 = 종가/진입가)
    seg = df["Close"].astype("float64").loc[trade["entry_date"] : trade["exit_date"]]
    eq_seg = seg / trade["entry_price"]
    expected_mdd = float((eq_seg / eq_seg.cummax() - 1.0).min())

    assert result.mdd == pytest.approx(expected_mdd)
    assert result.mdd < 0.0  # 보유구간에 낙폭이 존재


def test_mdd_zero_for_monotonic_equity():
    # 하락(골든 진입) 후 마지막까지 단조 상승 → 강제청산, 자산곡선 단조증가
    close = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
             11, 13, 16, 20, 25, 31, 38, 46]
    df = _ohlcv_from_close(close)
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4)

    assert result.num_trades == 1
    # 마지막 봉까지 보유 → 마지막 종가로 강제 청산
    assert result.trades[-1]["exit_date"] == df.index[-1]
    assert result.mdd == pytest.approx(0.0)


# --------------------------- 신호 없음 / 미지원 --------------------------
def test_no_signal_produces_no_trades():
    df = _ohlcv_from_close([100.0] * 30)  # 완전 평탄 → 교차 없음
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4)

    assert result.num_trades == 0
    assert result.total_return == pytest.approx(0.0)
    assert result.win_rate == 0.0
    assert np.allclose(result.equity_curve.to_numpy(), 1.0)


def test_rsi_no_trigger_produces_no_trades():
    # 좁은 진동 → RSI가 30~70 사이에 머물러 진입 트리거 없음
    close = list(100 + 2 * np.sin(np.linspace(0, 6 * np.pi, 60)))
    df = _ohlcv_from_close(close)
    result = backtest.run_backtest(df, "rsi")

    assert result.num_trades == 0
    assert result.total_return == pytest.approx(0.0)
    assert np.allclose(result.equity_curve.to_numpy(), 1.0)


def test_unsupported_strategy_raises():
    df = _ohlcv_from_close([100.0] * 30)
    with pytest.raises(ValueError):
        backtest.run_backtest(df, "unknown_strategy")


# --------------------------- 요약 표 -------------------------------------
def test_summary_dataframe_structure():
    df = _ohlcv_from_close(_GOLDEN_CLOSE)
    result = backtest.run_backtest(df, "golden_cross", short=2, long=4)
    table = backtest.summary_dataframe(result)

    assert list(table.columns) == ["지표", "값"]
    metrics = set(table["지표"])
    assert {"총수익률", "바이앤홀드", "초과수익", "승률", "거래수", "MDD(최대낙폭)"} <= metrics
    # 거래수는 '건' 단위 포맷
    trade_row = table[table["지표"] == "거래수"]["값"].iloc[0]
    assert trade_row.endswith("건")


# --------------------------- 데모 데이터 스모크 ---------------------------
def test_runs_on_sample_data():
    # 합성 데모 데이터에서 두 전략 모두 예외 없이 동작(자산곡선 길이 정합)
    from src.data import sample_data

    df = sample_data(days=300)
    for strat in ("golden_cross", "rsi"):
        result = backtest.run_backtest(df, strat, short=20, long=60)
        assert len(result.equity_curve) == len(df)
        assert result.num_trades >= 0
