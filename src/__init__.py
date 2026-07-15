"""주식 추세 분석 앱의 핵심 로직 패키지.

모듈 구성:
- ``data``       : yfinance 기반 OHLCV 데이터 조회
- ``indicators`` : 기술적 지표 계산 (SMA/EMA/RSI/MACD/볼린저/ATR)
- ``trend``      : 추세 판정 및 골든/데드크로스 탐지
- ``signals``    : 규칙 기반 매수/매도 신호
- ``levels``     : 지지/저항 가격대 산출
"""
