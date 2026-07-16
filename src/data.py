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

# ---------------------------------------------------------------------------
# 한국 인기 종목 한글명 → 야후 심볼 매핑
# ---------------------------------------------------------------------------
# 코스피 종목은 ``.KS``, 코스닥 종목은 ``.KQ`` 접미사를 사용한다.
# 키는 사용자가 입력하기 쉬운 대표 명칭(원문 유지)이며, 조회 시에는
# ``_normalize_name_key`` 로 공백 제거·소문자화하여 정확 매칭한다(부분일치 금지).
KR_NAME_TO_CODE: dict[str, str] = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "카카오": "035720.KS",
    "카카오뱅크": "323410.KS",
    "카카오게임즈": "293490.KQ",
    "NAVER": "035420.KS",
    "네이버": "035420.KS",
    "셀트리온": "068270.KS",
    "에코프로": "086520.KQ",
    "에코프로비엠": "247540.KQ",
    "LG에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "POSCO홀딩스": "005490.KS",
    "포스코홀딩스": "005490.KS",
    "삼성바이오로직스": "207940.KS",
    "LG화학": "051910.KS",
    "삼성SDI": "006400.KS",
    "KB금융": "105560.KS",
}


def _normalize_name_key(text: str) -> str:
    """종목명 매칭용 정규화 키: 모든 공백 제거 + 소문자화.

    'NAVER' / 'naver' / ' 네이버 ' 처럼 대소문자·공백만 다른 입력을
    동일하게 취급하기 위한 내부 헬퍼다.
    """
    return "".join(text.split()).lower()


# 정규화 키 기반 역방향 조회 테이블 (모듈 로드 시 1회 구성)
_KR_NAME_LOOKUP: dict[str, str] = {
    _normalize_name_key(name): symbol for name, symbol in KR_NAME_TO_CODE.items()
}


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


def resolve_ticker(text: str) -> str:
    """사용자 입력 텍스트를 야후 심볼로 정규화한다.

    처리 순서:

    1. 앞뒤 공백 제거 후 빈 문자열이면 ``ValueError('티커를 입력하세요.')``.
    2. 한글 종목명(공백·대소문자 무시)이 :data:`KR_NAME_TO_CODE` 에 정확히
       일치하면 해당 야후 심볼을 반환(부분일치는 하지 않아 오탐을 방지).
    3. 그 외에는 :func:`normalize_ticker` 에 위임 — 6자리 숫자코드는 ``.KS``,
       일반 티커는 대문자화. ``005930.KS`` / ``035720.KQ`` 처럼 이미 접미사가
       붙은 입력은 그대로 통과한다.

    Parameters
    ----------
    text : 예) ``"삼성전자"``, ``" 네이버 "``, ``"AAPL"``, ``"005930"``

    Returns
    -------
    str : 정규화된 야후 심볼.

    Raises
    ------
    ValueError : 입력이 비어 있는 경우.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("티커를 입력하세요.")

    # (2) 한글 종목명 정확 매칭
    hit = _KR_NAME_LOOKUP.get(_normalize_name_key(stripped))
    if hit is not None:
        return hit

    # (3) 기존 정규화 로직 재사용 (6자리→.KS, 일반 티커 대문자화)
    return normalize_ticker(stripped)


def market_label(symbol: str) -> tuple[str, str]:
    """정규화된 심볼로부터 ``(시장, 통화)`` 라벨 튜플을 반환한다.

    규칙(대소문자 무시):

    - ``.KS`` 접미사 → ``("KOSPI", "KRW")``
    - ``.KQ`` 접미사 → ``("KOSDAQ", "KRW")``
    - 접미사 없는 순수 6자리 숫자코드 → ``("KOSPI", "KRW")`` 로 취급
      (호출부에서 :func:`resolve_ticker` 를 거치지 않은 6자리 입력도 안전하게 동작)
    - 그 외(일반 티커) → ``("US", "USD")``

    앱에서 현재가 옆에 시장·통화를 표시하는 데 사용한다.
    """
    s = symbol.strip().upper()
    if s.endswith(".KS"):
        return ("KOSPI", "KRW")
    if s.endswith(".KQ"):
        return ("KOSDAQ", "KRW")
    if s.isdigit() and len(s) == 6:
        # 접미사 없는 6자리는 국내 코스피로 기본 판정
        return ("KOSPI", "KRW")
    return ("US", "USD")


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
    use_fallback: bool = True,
) -> pd.DataFrame:
    """티커의 OHLCV 시계열을 반환한다.

    Parameters
    ----------
    ticker : 예) ``"AAPL"``, ``"삼성전자"``, ``"005930.KS"``, ``"005930"``
    period : ``VALID_PERIODS`` 중 하나
    interval : ``VALID_INTERVALS`` 중 하나
    use_fallback : yfinance 조회가 비거나 실패하면 선택 의존성
        FinanceDataReader 대체 소스를 시도할지 여부(기본 True). 기본
        위치인자 호출(``fetch_ohlcv('AAPL', '1y', '1d')``)과 100% 호환되도록
        기본값 있는 키워드 인자로만 추가한다.

    Returns
    -------
    DatetimeIndex 를 가진 DataFrame (컬럼: Open/High/Low/Close/Volume).

    Raises
    ------
    ValueError : 티커가 비었거나, 데이터가 조회되지 않는 경우.
    """
    import yfinance as yf

    # 기존 normalize_ticker 대신 resolve_ticker 사용(한글명 인식 추가).
    # 6자리/일반 티커 정규화 동작은 그대로 보존된다.
    symbol = resolve_ticker(ticker)

    df = None
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:  # noqa: BLE001 - 네트워크/조회 실패는 fallback으로 처리
        df = None

    if df is None or df.empty:
        # yfinance가 비었거나 예외 → 선택 의존성 대체 소스 시도
        if use_fallback:
            # 사용자 PC에서 검증 필요 (이 샌드박스는 네트워크 차단)
            alt = _fetch_via_fdr(symbol, period=period, interval=interval)
            if alt is not None and not alt.empty:
                df = alt

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


def _fetch_via_fdr(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame | None:
    """선택 의존성 FinanceDataReader 를 이용한 대체 데이터 조회(내부 헬퍼).

    yfinance 조회가 막힌 환경(예: Yahoo 방화벽 차단)에서 국내 종목을 보조로
    조회하기 위한 경로다. ``.KS`` / ``.KQ`` 심볼은 6자리 코드로 변환해
    ``fdr.DataReader(code)`` 로 조회하고, 결과를 :data:`_REQUIRED_COLS`
    (Open/High/Low/Close/Volume) 단일 레벨로 정규화해 반환한다.

    import 실패(미설치)나 네트워크 예외 시에는 조용히 ``None`` 을 반환하여
    호출부(:func:`fetch_ohlcv`)가 기존과 동일한 ``ValueError`` 로 처리하도록 한다.

    .. note::
        네트워크에 의존하는 경로다. **사용자 PC에서 검증 필요** — 이 샌드박스는
        외부 데이터 소스가 차단되어 있어 여기서는 실제로 실행/검증되지 않았다.
        (테스트에서는 호출하지 않는다.)
    """
    try:  # pragma: no cover - 선택 의존성/네트워크 경로 (사용자 PC에서 검증 필요)
        import FinanceDataReader as fdr  # 지연 import (미설치 시 ImportError)

        # .KS/.KQ 접미사는 FinanceDataReader용 6자리 코드로 변환
        code = symbol
        if symbol.upper().endswith((".KS", ".KQ")):
            code = symbol.rsplit(".", 1)[0]

        raw = fdr.DataReader(code)  # 사용자 PC에서 검증 필요
        if raw is None or raw.empty:
            return None

        # 컬럼을 단일 레벨 OHLCV로 정규화 (기존 _flatten_columns 재사용)
        raw = _flatten_columns(raw, symbol)
        missing = [c for c in _REQUIRED_COLS if c not in raw.columns]
        if missing:
            return None

        out = raw[_REQUIRED_COLS].dropna(how="all")
        out.index.name = "Date"
        return out
    except Exception:  # noqa: BLE001 - ImportError/네트워크 등 모두 조용히 무시
        return None


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
