"""매매 신호 알림(src.alerts) 단위 테스트.

네트워크 없이 합성 데이터로 '오늘 트리거된 조건' 탐지 로직을 검증한다.
라이브(yfinance) 경로는 이 샌드박스에서 실행 불가하므로 fetch 주입으로 대체한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import alerts, indicators
from src.alerts import Alert, alerts_dataframe, check_alerts, check_watchlist
from src.data import sample_data
from src.trend import detect_crosses


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


def _kinds(al: list[Alert], kind: str) -> list[Alert]:
    return [a for a in al if a.kind == kind]


# --------------------------- 골든/데드크로스 --------------------------------
def test_cross_golden_on_last_bar():
    # 하락 후 상승 전환 → 골든크로스. 그 교차 봉을 마지막 봉으로 잘라서 '오늘'로 만든다.
    close = list(np.linspace(200, 100, 80)) + list(np.linspace(100, 260, 80))
    df = _ohlcv_from_close(close)
    crosses = detect_crosses(df, short=20, long=60)
    gidx = crosses[crosses["type"] == "golden"].index[0]
    pos = df.index.get_loc(gidx)
    df_today = df.iloc[: pos + 1]

    al = check_alerts(df_today, "TEST", windows=(20, 60))
    cross_alerts = _kinds(al, "cross")
    assert len(cross_alerts) == 1
    assert cross_alerts[0].direction == "bullish"
    assert cross_alerts[0].severity == "high"


def test_cross_not_today_no_alert():
    # 골든크로스가 과거(마지막 봉보다 앞)에만 있으면 cross 알림이 나오면 안 된다.
    close = list(np.linspace(200, 100, 80)) + list(np.linspace(100, 260, 80))
    df = _ohlcv_from_close(close)  # 교차는 중간에 있고 마지막 봉은 상승 지속
    al = check_alerts(df, "TEST", windows=(20, 60))
    assert _kinds(al, "cross") == []


def test_cross_dead_on_last_bar():
    close = list(np.linspace(100, 260, 80)) + list(np.linspace(260, 100, 80))
    df = _ohlcv_from_close(close)
    crosses = detect_crosses(df, short=20, long=60)
    didx = crosses[crosses["type"] == "dead"].index[0]
    pos = df.index.get_loc(didx)
    df_today = df.iloc[: pos + 1]

    al = check_alerts(df_today, "TEST", windows=(20, 60))
    cross_alerts = _kinds(al, "cross")
    assert len(cross_alerts) == 1
    assert cross_alerts[0].direction == "bearish"


# --------------------------- RSI 30/70 돌파 --------------------------------
def _first_cross_index(series: pd.Series, predicate) -> int:
    """predicate(prev, cur)가 처음 참이 되는 위치 인덱스를 찾는다."""
    for i in range(1, len(series)):
        a, b = series.iloc[i - 1], series.iloc[i]
        if pd.notna(a) and pd.notna(b) and predicate(float(a), float(b)):
            return i
    raise AssertionError("조건을 만족하는 봉을 찾지 못했습니다")


def test_rsi_bullish_cross_up_30():
    close = list(np.linspace(100, 70, 50)) + list(np.linspace(70, 100, 50))
    df = _ohlcv_from_close(close)
    r = indicators.rsi(df["Close"], 14)
    i = _first_cross_index(r, lambda a, b: a < 30 <= b)
    al = check_alerts(df.iloc[: i + 1], "TEST", windows=(20, 60))
    rsi_alerts = _kinds(al, "rsi")
    assert len(rsi_alerts) == 1
    assert rsi_alerts[0].direction == "bullish"


def test_rsi_bearish_cross_down_70():
    close = list(np.linspace(70, 100, 50)) + list(np.linspace(100, 70, 50))
    df = _ohlcv_from_close(close)
    r = indicators.rsi(df["Close"], 14)
    i = _first_cross_index(r, lambda a, b: a > 70 >= b)
    al = check_alerts(df.iloc[: i + 1], "TEST", windows=(20, 60))
    rsi_alerts = _kinds(al, "rsi")
    assert len(rsi_alerts) == 1
    assert rsi_alerts[0].direction == "bearish"


def test_rsi_no_alert_in_midrange():
    rng = np.random.default_rng(4)
    close = list(150 + rng.normal(scale=0.5, size=150))  # 좁은 박스권 → RSI 30~70
    df = _ohlcv_from_close(close)
    al = check_alerts(df, "TEST", windows=(20, 60))
    assert _kinds(al, "rsi") == []


# --------------------------- MACD 시그널 교차 -------------------------------
def test_macd_bullish_sign_flip():
    close = list(np.linspace(120, 90, 50)) + list(np.linspace(90, 130, 50))
    df = _ohlcv_from_close(close)
    h = indicators.macd(df["Close"])["hist"]
    i = _first_cross_index(h, lambda a, b: a <= 0 < b)
    al = check_alerts(df.iloc[: i + 1], "TEST", windows=(20, 60))
    macd_alerts = _kinds(al, "macd")
    assert len(macd_alerts) == 1
    assert macd_alerts[0].direction == "bullish"


def test_macd_bearish_sign_flip():
    close = list(np.linspace(90, 130, 50)) + list(np.linspace(130, 90, 50))
    df = _ohlcv_from_close(close)
    h = indicators.macd(df["Close"])["hist"]
    i = _first_cross_index(h, lambda a, b: a >= 0 > b)
    al = check_alerts(df.iloc[: i + 1], "TEST", windows=(20, 60))
    macd_alerts = _kinds(al, "macd")
    assert len(macd_alerts) == 1
    assert macd_alerts[0].direction == "bearish"


# --------------------------- 볼린저 상/하단 이탈 ----------------------------
def test_bollinger_bearish_spike_above_upper():
    rng = np.random.default_rng(0)
    close = list(150 + rng.normal(0, 0.5, 40))
    close[-1] = 170.0  # 마지막 봉만 상단 급등 스파이크
    df = _ohlcv_from_close(close)
    al = check_alerts(df, "TEST", windows=(20, 60))
    bb_alerts = _kinds(al, "bollinger")
    assert len(bb_alerts) == 1
    assert bb_alerts[0].direction == "bearish"


def test_bollinger_bullish_crash_below_lower():
    rng = np.random.default_rng(0)
    close = list(150 + rng.normal(0, 0.5, 40))
    close[-1] = 130.0  # 마지막 봉만 하단 급락
    df = _ohlcv_from_close(close)
    al = check_alerts(df, "TEST", windows=(20, 60))
    bb_alerts = _kinds(al, "bollinger")
    assert len(bb_alerts) == 1
    assert bb_alerts[0].direction == "bullish"


# --------------------------- 지지/저항 돌파 ---------------------------------
def test_level_bullish_breakout_resistance():
    t = np.linspace(0, 12 * np.pi, 120)
    close = list(110 + 10 * np.sin(t))  # 100~120 진동 → 저항 ~120대 형성
    close[-2] = 118.0  # 전일: 저항 아래
    close[-1] = 125.0  # 오늘: 저항 상향 돌파
    df = _ohlcv_from_close(close)
    al = check_alerts(df, "TEST", windows=(20, 60))
    level_alerts = _kinds(al, "level")
    assert any(a.direction == "bullish" for a in level_alerts)
    assert all(a.severity == "high" for a in level_alerts)


def test_level_bearish_breakdown_support():
    t = np.linspace(0, 12 * np.pi, 120)
    close = list(110 + 10 * np.sin(t))
    close[-2] = 102.0  # 전일: 지지 위
    close[-1] = 95.0  # 오늘: 지지 하향 이탈
    df = _ohlcv_from_close(close)
    al = check_alerts(df, "TEST", windows=(20, 60))
    level_alerts = _kinds(al, "level")
    assert any(a.direction == "bearish" for a in level_alerts)


# --------------------------- 무트리거 & 워치리스트 & 회귀 -------------------
def test_no_alerts_on_quiet_sample_data():
    # 특별한 이벤트가 없는 잔잔한 데모 데이터 → 알림 없음(오탐 없음)
    al = check_alerts(sample_data(), "DEMO", windows=(20, 60))
    assert al == []


def test_check_watchlist_with_injected_fetch():
    # 오프라인: sample_data 변형을 반환하는 가짜 fetch 주입 (네트워크 불필요)
    def fake_fetch(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        seed = abs(hash(ticker)) % 1000
        return sample_data(seed=seed)

    result = check_watchlist(["AAA", "BBB", "CCC"], fetch=fake_fetch)
    assert set(result.keys()) == {"AAA", "BBB", "CCC"}
    for ticker, al in result.items():
        assert isinstance(al, list)
        assert all(isinstance(a, Alert) for a in al)


def test_check_watchlist_skips_failing_ticker():
    def flaky_fetch(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        if ticker == "BAD":
            raise ValueError("조회 실패")
        return sample_data()

    result = check_watchlist(["GOOD", "BAD"], fetch=flaky_fetch)
    assert result["BAD"] == []  # 실패 티커는 빈 리스트, 전체 중단 없음
    assert "GOOD" in result


def test_alert_is_dataclass_with_fields():
    a = Alert(kind="cross", direction="bullish", message="예시", severity="high")
    assert a.kind == "cross"
    assert a.direction == "bullish"
    assert a.message == "예시"
    assert a.severity == "high"
    assert a.emoji == "🟢"
    assert Alert("cross", "bearish", "예시", "high").emoji == "🔴"


def test_alerts_dataframe_columns():
    al = [
        Alert(kind="cross", direction="bullish", message="골든", severity="high"),
        Alert(kind="rsi", direction="bearish", message="과열", severity="medium"),
    ]
    dfa = alerts_dataframe(al)
    assert list(dfa.columns) == ["종류", "방향", "중요도", "내용"]
    assert len(dfa) == 2
    assert dfa.iloc[0]["종류"] == "이평 교차"
    assert dfa.iloc[0]["방향"] == "강세🟢"


def test_alerts_dataframe_empty():
    dfa = alerts_dataframe([])
    assert list(dfa.columns) == ["종류", "방향", "중요도", "내용"]
    assert len(dfa) == 0
