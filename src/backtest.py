"""신호 기반 롱-온리 백테스트 모듈.

과거 OHLCV 시계열에 대해 규칙 신호(골든크로스 / RSI 과매도·과매수)로
**전량(all-in) 매수 → 청산**을 순차 복리로 시뮬레이션한다.

- 진입/청산 체결가는 모두 해당 시점 종가(Close).
- 보유(롱) 구간에서만 자산이 종가 비율로 변하고, 플랫(현금) 구간에서는 값이 불변.
- 마지막 봉까지 보유 중이면 마지막 종가로 강제 청산해 결과에 반영한다.

주의: 과거 성과는 미래 수익을 보장하지 않으며, 참고용 시뮬레이션일 뿐 매매 지시가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import indicators
from .trend import detect_crosses

# 지원 전략 식별자
STRATEGIES = ("golden_cross", "rsi")


@dataclass
class BacktestResult:
    """백테스트 산출물.

    Attributes
    ----------
    strategy : 사용한 전략명("golden_cross" | "rsi").
    total_return : 총수익률(소수). 예) 0.25 == +25%.
    win_rate : 승률(0~1). 종료된 거래 중 수익률>0 인 비율. 거래가 없으면 0.0.
    num_trades : 완료된 왕복 거래 수. 종료 시점 미청산 포지션은 마지막 종가로
        강제 청산해 1건으로 집계한다.
    mdd : 최대낙폭(자산곡선 peak-to-trough, 음수 소수). 낙폭이 없으면 0.0.
    buy_and_hold_return : 바이앤홀드 수익률(마지막종가 / 첫유효종가 - 1).
    excess_return : total_return - buy_and_hold_return.
    equity_curve : 자산곡선 Series(df.index 정렬, 첫 유효값=initial_capital).
    trades : 거래 상세 dict 리스트
        (entry_date / entry_price / exit_date / exit_price / return_pct).
    """

    strategy: str
    total_return: float
    win_rate: float
    num_trades: int
    mdd: float
    buy_and_hold_return: float
    excess_return: float
    equity_curve: pd.Series
    trades: list[dict] = field(default_factory=list)

    @property
    def emoji(self) -> str:
        # 총수익률 부호로 간단한 시각 표식을 제공한다.
        if self.total_return > 0:
            return "🟢"
        if self.total_return < 0:
            return "🔴"
        return "⚪"


def _signal_masks(
    df: pd.DataFrame,
    strategy: str,
    *,
    short: int,
    long: int,
    rsi_period: int,
    rsi_buy: float,
    rsi_sell: float,
) -> tuple[dict, dict]:
    """전략별로 봉마다의 (진입 후보, 청산 후보) 여부를 계산한다.

    Returns
    -------
    (want_enter, want_exit) : 각각 ``{timestamp: bool}`` 형태의 dict.
        - golden_cross : detect_crosses 의 'golden' 이벤트에서 진입 후보,
          'dead' 이벤트에서 청산 후보. (직접 SMA 교차를 재구현하지 않고 재사용)
        - rsi : RSI < rsi_buy 이면 진입 후보, RSI > rsi_sell 이면 청산 후보.
    """
    want_enter: dict = {}
    want_exit: dict = {}

    if strategy == "golden_cross":
        crosses = detect_crosses(df, short=short, long=long)
        for ts, row in crosses.iterrows():
            if row["type"] == "golden":
                want_enter[ts] = True
            elif row["type"] == "dead":
                want_exit[ts] = True
    elif strategy == "rsi":
        close = df["Close"].astype("float64")
        r = indicators.rsi(close, rsi_period)
        for ts, val in r.items():
            if pd.isna(val):
                continue
            fv = float(val)
            if fv < rsi_buy:
                want_enter[ts] = True
            elif fv > rsi_sell:
                want_exit[ts] = True
    else:  # 방어적 처리 — 호출부에서 이미 검증하지만 안전하게 유지
        raise ValueError(f"지원하지 않는 전략입니다: {strategy!r}")

    return want_enter, want_exit


def run_backtest(
    df: pd.DataFrame,
    strategy: str = "golden_cross",
    *,
    short: int = 20,
    long: int = 60,
    rsi_period: int = 14,
    rsi_buy: float = 30.0,
    rsi_sell: float = 70.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """OHLCV DataFrame에 대해 롱-온리 신호 매매를 시뮬레이션한다.

    Parameters
    ----------
    df : Open/High/Low/Close/Volume 컬럼을 가진 OHLCV DataFrame.
    strategy : "golden_cross" 또는 "rsi".
        - "golden_cross" : ``detect_crosses(df, short, long)`` 결과를 시간순으로
          훑어, 플랫 상태에서 'golden' 이벤트를 만나면 진입, 보유 중일 때
          'dead' 이벤트를 만나면 청산한다.
        - "rsi" : ``indicators.rsi(close, rsi_period)`` 를 봉 단위로 순회하며,
          플랫이고 RSI < rsi_buy 이면 해당 봉 종가로 진입, 보유 중이고
          RSI > rsi_sell 이면 해당 봉 종가로 청산한다.
    short, long : 골든크로스 전략의 단기/장기 이동평균 기간.
    rsi_period, rsi_buy, rsi_sell : RSI 전략 파라미터.
    initial_capital : 초기 자본(자산곡선 기준값). 기본 1.0.

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError : 지원하지 않는 ``strategy`` 문자열인 경우.

    Notes
    -----
    전량 매매(all-in)로 거래가 순차 복리 반영되며, 자산곡선은 보유 구간에서만
    종가 비율로 마킹되고 플랫 구간에서는 값이 불변이다. 마지막 봉까지 보유
    중이면 마지막 종가로 강제 청산해 total_return / num_trades / equity_curve
    에 반영한다. 신호가 전혀 없으면 num_trades=0, total_return=0.0,
    win_rate=0.0, 자산곡선은 전부 initial_capital 이다.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"지원하지 않는 전략입니다: {strategy!r} "
            f"(가능: {', '.join(STRATEGIES)})"
        )

    close = df["Close"].astype("float64")
    n = len(close)

    # 빈 입력 방어 — 길이 0이면 빈 자산곡선을 돌려준다.
    if n == 0:
        empty = pd.Series(dtype="float64")
        return BacktestResult(
            strategy=strategy,
            total_return=0.0,
            win_rate=0.0,
            num_trades=0,
            mdd=0.0,
            buy_and_hold_return=0.0,
            excess_return=0.0,
            equity_curve=empty,
            trades=[],
        )

    want_enter, want_exit = _signal_masks(
        df,
        strategy,
        short=short,
        long=long,
        rsi_period=rsi_period,
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell,
    )

    # --- 순차 시뮬레이션 --------------------------------------------------
    capital = float(initial_capital)  # 청산 시점마다 갱신되는 실현 자본
    holding = False
    entry_price = 0.0
    entry_date = None
    equity_values: list[float] = []
    trades: list[dict] = []

    for ts, raw in close.items():
        price = float(raw)

        # 이벤트 처리: 플랫이면 진입, 보유 중이면 청산 (한 봉에 하나만 발생)
        if holding and want_exit.get(ts, False):
            # 종가로 청산 → 자본을 복리 반영하고 거래 기록
            capital = capital * (price / entry_price)
            trades.append(
                {
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": ts,
                    "exit_price": price,
                    "return_pct": price / entry_price - 1.0,
                }
            )
            holding = False
            entry_price = 0.0
            entry_date = None
        elif not holding and want_enter.get(ts, False):
            # 종가로 진입 (이 봉의 자산은 capital 그대로: price/price == 1)
            holding = True
            entry_price = price
            entry_date = ts

        # 자산 마킹: 보유 중이면 종가 비율, 플랫이면 직전 실현 자본(불변)
        if holding:
            equity_values.append(capital * (price / entry_price))
        else:
            equity_values.append(capital)

    # --- 마지막 봉까지 보유 중이면 강제 청산 ------------------------------
    if holding:
        last_ts = close.index[-1]
        last_price = float(close.iloc[-1])
        capital = capital * (last_price / entry_price)
        trades.append(
            {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": last_ts,
                "exit_price": last_price,
                "return_pct": last_price / entry_price - 1.0,
            }
        )
        # 강제 청산가 == 마지막 종가이므로 마지막 자산 마킹값과 동일(재조정 불필요)
        holding = False

    equity_curve = pd.Series(equity_values, index=close.index, dtype="float64")

    # --- 성과 지표 --------------------------------------------------------
    total_return = capital / float(initial_capital) - 1.0
    num_trades = len(trades)
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    win_rate = (wins / num_trades) if num_trades else 0.0

    # 최대낙폭(MDD): 자산곡선의 고점 대비 최대 하락률(음수). 낙폭 없으면 0.0.
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    mdd = float(drawdown.min()) if len(drawdown) else 0.0
    if mdd > 0.0:  # 부동소수 오차 방어
        mdd = 0.0

    # 바이앤홀드: 첫 유효종가 대비 마지막 유효종가 수익률
    valid_close = close.dropna()
    if len(valid_close) >= 1:
        first_close = float(valid_close.iloc[0])
        last_close = float(valid_close.iloc[-1])
        buy_and_hold_return = (last_close / first_close - 1.0) if first_close else 0.0
    else:
        buy_and_hold_return = 0.0

    excess_return = total_return - buy_and_hold_return

    return BacktestResult(
        strategy=strategy,
        total_return=total_return,
        win_rate=win_rate,
        num_trades=num_trades,
        mdd=mdd,
        buy_and_hold_return=buy_and_hold_return,
        excess_return=excess_return,
        equity_curve=equity_curve,
        trades=trades,
    )


def summary_dataframe(result: BacktestResult) -> pd.DataFrame:
    """BacktestResult를 UI 지표 표(컬럼: '지표', '값')로 변환한다.

    signals.signals_dataframe 패턴을 따라, 사람이 읽기 좋은 포맷(%, 건수)으로
    총수익률 / 바이앤홀드 / 초과수익 / 승률 / 거래수 / MDD 를 담는다.
    백테스트 탭의 st.dataframe 에서 그대로 재사용한다.
    """
    rows = [
        {"지표": "총수익률", "값": f"{result.total_return * 100:+.2f}%"},
        {"지표": "바이앤홀드", "값": f"{result.buy_and_hold_return * 100:+.2f}%"},
        {"지표": "초과수익", "값": f"{result.excess_return * 100:+.2f}%"},
        {"지표": "승률", "값": f"{result.win_rate * 100:.1f}%"},
        {"지표": "거래수", "값": f"{result.num_trades}건"},
        {"지표": "MDD(최대낙폭)", "값": f"{result.mdd * 100:.2f}%"},
    ]
    return pd.DataFrame(rows, columns=["지표", "값"])
