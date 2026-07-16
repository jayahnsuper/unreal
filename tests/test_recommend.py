"""추천 스캔(screener.recommend) 단위 테스트.

네트워크 없이 sample_data 주입 fetch로 검증한다.
"""

from __future__ import annotations

from src import screener
from src.data import sample_data


def _demo_fetch(ticker: str, period: str = "6mo", interval: str = "1d"):
    """티커별 고정 시드의 합성 데이터를 돌려주는 주입 fetch."""
    seed = abs(hash(ticker)) % 10_000
    return sample_data(days=250, seed=seed)


def test_recommend_top_n_and_sorted():
    uni = {f"T{i}": f"종목{i}" for i in range(15)}
    res = screener.recommend(uni, top_n=10, fetch=_demo_fetch, min_score=0)
    # 상위 N 제한
    assert len(res) <= 10
    # 스키마 일치
    assert list(res.columns) == list(screener.RECOMMEND_COLUMNS)
    # 매매점수 내림차순 정렬
    scores = res["매매점수"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_min_score_filter():
    uni = {f"T{i}": f"종목{i}" for i in range(10)}
    res = screener.recommend(uni, top_n=10, fetch=_demo_fetch, min_score=101)
    # 101점 이상은 존재할 수 없으므로 결과는 비어야 한다
    assert res.empty


def test_recommend_all_pass_have_min_score():
    uni = {f"X{i}": f"종목{i}" for i in range(12)}
    res = screener.recommend(uni, top_n=12, fetch=_demo_fetch, min_score=55)
    if not res.empty:
        assert (res["매매점수"] >= 55).all()
