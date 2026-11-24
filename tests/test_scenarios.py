"""
추가 테스트 시나리오 (기존 코드 수정 없음).

miser.py / experiments.py 만 import 하므로 plotly·streamlit 없이도 실행된다.
    python -m pytest -q tests/test_scenarios.py

기존 tests/test_algorithms.py 와 겹치지 않는 경계 조건·차원·통계 시나리오를 다룬다.
몬테카를로 특성상 단일 실행의 오차를 고정 임계값으로 검사하지 않고,
정확해가 보장되는 경우(상수 함수)나 여러 시드의 RMSE, 넓은 표준오차 범위로만 검사한다.
"""

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import INTEGRANDS, make_bounds  # noqa: E402
from miser import run_miser, run_plain_monte_carlo  # noqa: E402

UNIT2 = [[0.0, 1.0], [0.0, 1.0]]


def const(value):
    return lambda p: np.full(len(p), value, dtype=float)


def gauss_nd(center=0.5, sigma=0.1):
    return lambda p: np.exp(-np.sum((p - center) ** 2, axis=1) / (2 * sigma**2))


def gauss_nd_exact(dim, center=0.5, sigma=0.1):
    one = sigma * math.sqrt(math.pi / 2) * (
        math.erf((1 - center) / (math.sqrt(2) * sigma))
        - math.erf((0 - center) / (math.sqrt(2) * sigma))
    )
    return one**dim


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


# ------------------------------------------------------------- 1. 정확해 시나리오
@pytest.mark.parametrize("bounds, expected", [
    (UNIT2, 2.5),
    ([[2.0, 5.0], [-1.0, 1.0]], 2.5 * 3 * 2),   # 평행이동·비단위 구간
    ([[-3.0, -1.0]], 2.5 * 2),                   # 1D 음수 구간
])
def test_constant_function_is_exact(bounds, expected):
    """상수 함수는 분산 0 → 두 알고리즘 모두 부피 × 상수를 정확히 반환해야 한다."""
    assert math.isclose(run_plain_monte_carlo(10_000, 1, const(2.5), bounds)["estimate"],
                        expected, rel_tol=1e-12)
    assert math.isclose(run_miser(10_000, 2, 300, const(2.5), bounds)["estimate"],
                        expected, rel_tol=1e-12)


def test_zero_variance_splits_budget_evenly():
    """파일럿 분산이 모두 0이면(weight_sum ≈ 0) 남은 예산을 정확히 반으로 나눈다."""
    ms = run_miser(10_000, 1, 300, const(1.0), UNIT2)
    assert ms["splits"], "분할이 한 번도 일어나지 않음"
    for ev in ms["splits"]:
        assert ev["var_left"] == 0.0 and ev["var_right"] == 0.0
        assert abs(ev["n_left"] - ev["n_right"]) <= 1


# ------------------------------------------------------------- 2. 경계 조건
@pytest.mark.parametrize("n", [50, 500, 600])
def test_small_budget_becomes_single_leaf(n):
    """n ≤ 2·leaf_size + pilot_cost 이면 분할 없이 리프 하나 = 일반 MC와 같은 구조."""
    f = INTEGRANDS["circle"]
    ms = run_miser(n, 1, 300, f.func, make_bounds(f, 0, 1))
    assert ms["splits"] == []
    assert len(ms["nodes"]) == 1 and ms["nodes"][0]["leaf"]
    assert ms["evaluations"] == n
    assert np.all(ms["kinds"] == 1)  # 파일럿 없음


def test_leaf_size_larger_than_budget():
    ms = run_miser(1_000, 1, 5_000, const(1.0), UNIT2)
    assert ms["splits"] == [] and ms["evaluations"] == 1_000


def test_depth_is_capped_at_12():
    """leaf_size 를 아주 작게 해도 재귀 깊이는 12를 넘지 않는다."""
    f = INTEGRANDS["circle"]
    ms = run_miser(200_000, 1, 1, f.func, make_bounds(f, 0, 1))
    assert max(nd["depth"] for nd in ms["nodes"]) <= 12
    assert ms["evaluations"] == 200_000


def test_every_leaf_respects_leaf_size():
    f = INTEGRANDS["village"]
    for leaf in (100, 300, 1_000):
        ms = run_miser(50_000, 3, leaf, f.func, make_bounds(f, 0, 1))
        assert min(nd["count"] for nd in ms["nodes"] if nd["leaf"]) >= leaf


@pytest.mark.parametrize("bad", [
    [[1.0, 1.0]],            # 폭 0
    [[1.0, 0.0]],            # 역순
    [[0.0, math.inf]],       # 무한
    [[0.0, 1.0], [math.nan, 1.0]],
])
def test_invalid_bounds_raise(bad):
    with pytest.raises(ValueError):
        run_plain_monte_carlo(100, 0, const(1.0), bad)
    with pytest.raises(ValueError):
        run_miser(100, 0, 300, const(1.0), bad)


# ------------------------------------------------------------- 3. 고차원
def test_3d_gaussian_budget_and_accuracy():
    """dim=3 일반화: 파일럿 비용 2·dim·m, 예산 보존, 추정치 타당성."""
    f, exact = gauss_nd(), gauss_nd_exact(3)
    ms = run_miser(30_000, 1, 300, f, [[0.0, 1.0]] * 3)
    mc = run_plain_monte_carlo(30_000, 0, f, [[0.0, 1.0]] * 3)
    assert ms["evaluations"] == mc["evaluations"] == 30_000
    assert ms["dim"] == 3
    for ev in ms["splits"]:
        assert ev["pilot_cost"] % (2 * 3) == 0
    # 피적분함수 표준편차 ≤ 1 이므로 10·1/√n 은 매우 넓은 허용 범위.
    tol = 10 / math.sqrt(30_000)
    assert abs(ms["estimate"] - exact) < tol
    assert abs(mc["estimate"] - exact) < tol


# ------------------------------------------------------------- 4. 통계 시나리오
def test_miser_reduces_rmse_on_smooth_peak():
    """매끈한 가우시안 피크(village density)에서는 MISER 의 RMSE 가 더 작다.
    (확인 당시 n=20,000, 40 시드에서 약 3.3배 차이)"""
    f = INTEGRANDS["village"]
    b = make_bounds(f, 0, 1)
    ref = f.reference(b)
    mc = [run_plain_monte_carlo(20_000, s, f.func, b)["estimate"] - ref for s in range(40)]
    ms = [run_miser(20_000, s + 1, 300, f.func, b)["estimate"] - ref for s in range(40)]
    assert rmse(ms) < rmse(mc)


def test_circle_indicator_no_guarantee_but_unbiased():
    """원 지시함수에서는 MISER 가 항상 더 낫지 않다(README 명시).
    대신 두 알고리즘 모두 평균 오차(편향)가 표준오차 범위 안에 있는지 확인."""
    f = INTEGRANDS["circle"]
    b = make_bounds(f, 0, 1)
    ref = f.reference(b)
    n, seeds = 20_000, 40
    for run in (lambda s: run_plain_monte_carlo(n, s, f.func, b),
                lambda s: run_miser(n, s + 1, 300, f.func, b)):
        errs = np.array([run(s)["estimate"] - ref for s in range(seeds)])
        # 평균 오차가 (실측 표준편차 / √시드 수) 의 6배 안이면 편향 없음으로 판정.
        assert abs(errs.mean()) < 6 * errs.std(ddof=1) / math.sqrt(seeds)


def test_different_seeds_give_different_samples():
    f = INTEGRANDS["circle"]
    b = make_bounds(f, 0, 1)
    a = run_miser(5_000, 1, 300, f.func, b)
    c = run_miser(5_000, 2, 300, f.func, b)
    assert not np.array_equal(a["points"], c["points"])


# ------------------------------------------------------------- 5. 시연 시나리오 4개
from scenarios import SCENARIOS  # noqa: E402

DEMO_KEYS = tuple(SCENARIOS)


def test_demo_order():
    assert DEMO_KEYS == ("circle", "joint_default", "interference", "peak")


@pytest.mark.parametrize("key", DEMO_KEYS)
def test_demo_reference_matches_large_mc(key):
    sc = SCENARIOS[key]
    v = sc["func"](np.random.default_rng(7).random((2_000_000, 2)))
    assert abs(v.mean() - sc["exact"]()) < 5 * v.std() / math.sqrt(len(v))


@pytest.mark.parametrize("key", DEMO_KEYS)
def test_demo_budget_and_instrumentation(key):
    ms = run_miser(20_000, 1, 300, SCENARIOS[key]["func"], UNIT2)
    assert ms["evaluations"] == 20_000
    area = sum((nd["box"][0][1] - nd["box"][0][0]) * (nd["box"][1][1] - nd["box"][1][0])
               for nd in ms["nodes"] if nd["leaf"])
    assert math.isclose(area, 1.0)


def test_circle_is_original_demo():
    """시연 목록의 원은 원래 데모 함수(in_village) 그대로다."""
    p = np.random.default_rng(0).random((100_000, 2))
    assert np.array_equal(SCENARIOS["circle"]["func"](p), INTEGRANDS["circle"].func(p))


def _demo_rmse(key, seeds=30):
    f, ref = SCENARIOS[key]["func"], SCENARIOS[key]["exact"]()
    mc = [run_plain_monte_carlo(20_000, s, f, UNIT2)["estimate"] - ref for s in range(seeds)]
    ms = [run_miser(20_000, s + 1, 300, f, UNIT2)["estimate"] - ref for s in range(seeds)]
    return rmse(mc), rmse(ms)


def test_interference_miser_loses():
    """간섭 무늬(반례1): 나눌 이득이 없어 MC보다 나쁨 (확인 당시 0.66배)."""
    mc, ms = _demo_rmse("interference")
    assert ms > mc


def test_peak_this_implementation_loses():
    """별빛(반례2): 파일럿이 적어 봉우리를 놓친다 → MC보다 나쁨 (확인 당시 0.31배)."""
    mc, ms = _demo_rmse("peak")
    assert ms > mc
