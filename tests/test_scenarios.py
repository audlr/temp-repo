# 시나리오 4개의 정답값과 알고리즘 동작을 확인하기 위해 추가
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


@pytest.mark.parametrize("bounds, expected", [
    (UNIT2, 2.5),
    ([[2.0, 5.0], [-1.0, 1.0]], 2.5 * 3 * 2),
    ([[-3.0, -1.0]], 2.5 * 2),
])
def test_constant_function_is_exact(bounds, expected):
    assert math.isclose(run_plain_monte_carlo(10_000, 1, const(2.5), bounds)["estimate"],
                        expected, rel_tol=1e-12)
    assert math.isclose(run_miser(10_000, 2, 300, const(2.5), bounds)["estimate"],
                        expected, rel_tol=1e-12)


def test_zero_variance_splits_budget_evenly():
    ms = run_miser(10_000, 1, 300, const(1.0), UNIT2)
    assert ms["splits"], "분할이 한 번도 일어나지 않음"
    for ev in ms["splits"]:
        assert ev["var_left"] == 0.0 and ev["var_right"] == 0.0
        assert abs(ev["n_left"] - ev["n_right"]) <= 1


@pytest.mark.parametrize("n", [50, 500, 600])
def test_small_budget_becomes_single_leaf(n):
    f = INTEGRANDS["circle"]
    ms = run_miser(n, 1, 300, f.func, make_bounds(f, 0, 1))
    assert ms["splits"] == []
    assert len(ms["nodes"]) == 1 and ms["nodes"][0]["leaf"]
    assert ms["evaluations"] == n
    assert np.all(ms["kinds"] == 1)


def test_leaf_size_larger_than_budget():
    ms = run_miser(1_000, 1, 5_000, const(1.0), UNIT2)
    assert ms["splits"] == [] and ms["evaluations"] == 1_000


def test_depth_is_capped_at_12():
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
    [[1.0, 1.0]],
    [[1.0, 0.0]],
    [[0.0, math.inf]],
    [[0.0, 1.0], [math.nan, 1.0]],
])
def test_invalid_bounds_raise(bad):
    with pytest.raises(ValueError):
        run_plain_monte_carlo(100, 0, const(1.0), bad)
    with pytest.raises(ValueError):
        run_miser(100, 0, 300, const(1.0), bad)


def test_3d_gaussian_budget_and_accuracy():
    f, exact = gauss_nd(), gauss_nd_exact(3)
    ms = run_miser(30_000, 1, 300, f, [[0.0, 1.0]] * 3)
    mc = run_plain_monte_carlo(30_000, 0, f, [[0.0, 1.0]] * 3)
    assert ms["evaluations"] == mc["evaluations"] == 30_000
    assert ms["dim"] == 3
    for ev in ms["splits"]:
        assert ev["pilot_cost"] % (2 * 3) == 0

    tol = 10 / math.sqrt(30_000)
    assert abs(ms["estimate"] - exact) < tol
    assert abs(mc["estimate"] - exact) < tol


def test_miser_reduces_rmse_on_smooth_peak():
    f = INTEGRANDS["village"]
    b = make_bounds(f, 0, 1)
    ref = f.reference(b)
    mc = [run_plain_monte_carlo(20_000, s, f.func, b)["estimate"] - ref for s in range(40)]
    ms = [run_miser(20_000, s + 1, 300, f.func, b)["estimate"] - ref for s in range(40)]
    assert rmse(ms) < rmse(mc)


def test_circle_indicator_no_guarantee_but_unbiased():
    f = INTEGRANDS["circle"]
    b = make_bounds(f, 0, 1)
    ref = f.reference(b)
    n, seeds = 20_000, 40
    for run in (lambda s: run_plain_monte_carlo(n, s, f.func, b),
                lambda s: run_miser(n, s + 1, 300, f.func, b)):
        errs = np.array([run(s)["estimate"] - ref for s in range(seeds)])

        assert abs(errs.mean()) < 6 * errs.std(ddof=1) / math.sqrt(seeds)


def test_different_seeds_give_different_samples():
    f = INTEGRANDS["circle"]
    b = make_bounds(f, 0, 1)
    a = run_miser(5_000, 1, 300, f.func, b)
    c = run_miser(5_000, 2, 300, f.func, b)
    assert not np.array_equal(a["points"], c["points"])


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
    p = np.random.default_rng(0).random((100_000, 2))
    assert np.array_equal(SCENARIOS["circle"]["func"](p), INTEGRANDS["circle"].func(p))


def _demo_rmse(key, seeds=30):
    f, ref = SCENARIOS[key]["func"], SCENARIOS[key]["exact"]()
    mc = [run_plain_monte_carlo(20_000, s, f, UNIT2)["estimate"] - ref for s in range(seeds)]
    ms = [run_miser(20_000, s + 1, 300, f, UNIT2)["estimate"] - ref for s in range(seeds)]
    return rmse(mc), rmse(ms)


def test_interference_miser_loses():
    mc, ms = _demo_rmse("interference")
    assert ms > mc


def test_peak_this_implementation_loses():
    mc, ms = _demo_rmse("peak")
    assert ms > mc
