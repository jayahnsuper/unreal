"""한국 주식 데이터 강화 기능 단위 테스트.

네트워크(yfinance / FinanceDataReader)에 의존하지 않고, 순수 함수만으로
한글 종목명 해석·심볼 정규화·시장/통화 라벨을 검증한다.

.. note::
    이 샌드박스는 Yahoo Finance가 방화벽 차단되어 라이브 조회가 불가하다.
    따라서 라이브 ``fetch_ohlcv`` 경로와 대체 소스 ``_fetch_via_fdr`` 는
    호출하지 않는다(합성/상수 검증만 수행).
"""

from __future__ import annotations

import re

import pytest

from src.data import (
    KR_NAME_TO_CODE,
    market_label,
    normalize_ticker,
    resolve_ticker,
)

# 야후 국내 심볼 형식: 6자리 숫자 + .KS(코스피)/.KQ(코스닥)
_KR_SYMBOL_RE = re.compile(r"^\d{6}\.(KS|KQ)$")


# --------------------------- 이름 → 코드 해석 --------------------------------
def test_resolve_ticker_korean_names():
    assert resolve_ticker("삼성전자") == "005930.KS"
    assert resolve_ticker("카카오") == "035720.KS"
    assert resolve_ticker("SK하이닉스") == "000660.KS"
    # 코스닥 종목은 .KQ 접미사
    assert resolve_ticker("에코프로") == "086520.KQ"


def test_resolve_ticker_name_ignores_case_and_spaces():
    # 공백/대소문자 무시 정확 매칭
    assert resolve_ticker(" 네이버 ") == "035420.KS"
    assert resolve_ticker("naver") == "035420.KS"
    assert resolve_ticker("NAVER") == "035420.KS"
    # 별칭도 동일 코드로 해석
    assert resolve_ticker("포스코홀딩스") == "005490.KS"
    assert resolve_ticker("POSCO홀딩스") == "005490.KS"


# --------------------------- 기존 정규화 회귀 없음 ---------------------------
def test_resolve_ticker_preserves_normalize_behavior():
    # 6자리 숫자코드 → .KS
    assert resolve_ticker("005930") == "005930.KS"
    # 일반 티커 대문자화
    assert resolve_ticker("aapl") == "AAPL"
    # 이미 접미사가 붙은 입력은 그대로 통과
    assert resolve_ticker("035720.KQ") == "035720.KQ"
    # resolve_ticker는 normalize_ticker를 위임 재사용한다
    assert resolve_ticker("005930") == normalize_ticker("005930")


def test_resolve_ticker_empty_raises():
    with pytest.raises(ValueError):
        resolve_ticker("")
    with pytest.raises(ValueError):
        resolve_ticker("   ")


def test_resolve_ticker_no_partial_match():
    # 부분일치 금지: 매핑에 없는 유사 입력은 일반 티커로 처리(오탐 방지)
    assert resolve_ticker("삼성") == "삼성"  # '삼성전자'와 부분일치하지 않음


# --------------------------- market_label 판정 ------------------------------
def test_market_label_rules():
    assert market_label("005930.KS") == ("KOSPI", "KRW")
    assert market_label("035720.KQ") == ("KOSDAQ", "KRW")
    assert market_label("AAPL") == ("US", "USD")


def test_market_label_bare_six_digits_is_kospi():
    # 접미사 없는 순수 6자리는 KOSPI/KRW로 취급
    assert market_label("005930") == ("KOSPI", "KRW")


def test_market_label_case_insensitive():
    assert market_label("005930.ks") == ("KOSPI", "KRW")
    assert market_label("aapl") == ("US", "USD")


# --------------------------- 매핑 무결성 ------------------------------------
def test_mapping_integrity():
    # 최소 15개 항목
    assert len(KR_NAME_TO_CODE) >= 15
    for name, symbol in KR_NAME_TO_CODE.items():
        # 모든 값이 6자리 숫자 + .KS/.KQ 형식
        assert _KR_SYMBOL_RE.match(symbol), f"{name} -> {symbol} 형식 오류"
        # 각 값을 market_label에 넣으면 통화는 KRW
        _, currency = market_label(symbol)
        assert currency == "KRW"


def test_mapping_resolves_through_resolve_ticker():
    # 매핑의 모든 한글명이 resolve_ticker로 정확히 해석된다
    for name, symbol in KR_NAME_TO_CODE.items():
        assert resolve_ticker(name) == symbol
