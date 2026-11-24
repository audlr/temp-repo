"""
테스트 시나리오.

app.py 는 import 하는 순간 Streamlit UI 코드가 실행되므로, app.py 를 ast 로 파싱해
알고리즘 함수(standard_monte_carlo, miser_monte_carlo)만 꺼내 실행한다.
시연 시나리오는 scenarios.py 에서 그대로 import 한다.
numpy + pytest 만으로 실행된다:  python -m pytest -q tests
"""

import ast
import math
import pathlib
import sys
import time

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
sys.path.insert(0, str(ROOT))

from scenarios import SCENARIOS, PEAK_CENTER, PEAK_SIGMA  # noqa: E402

FUNCTIONS = {"standard_monte_carlo", "miser_monte_carlo"}


def load_algorithms():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    missing = FUNCTIONS - {n.name for n in body}
    assert not missing, f"app.py 에서 함수를 찾지 못함: {missing}"
    ns = {"math": math, "np": np, "time": time}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), ns)
    return ns


A = load_algorithms()
standard = A["standard_monte_carlo"]
miser = A["miser_monte_carlo"]
UNIT2 = [(0.0, 1.0), (0.0, 1.0)]
KEYS = list(SCENARIOS)


def rng(seed):
    return np.random.default_rng(seed)


def const(value):
    return lambda p: np.full(len(p), value, dtype=float)


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def errs(key, n=20_000, seeds=20):
    f, ref = SCENARIOS[key]["func"], SCENARIOS[key]["exact"]()
    mc = np.array([standard(f, UNIT2, n, rng(s))["estimate"] - ref for s in range(seeds)])
    ms = np.array([miser(f, UNIT2, n, rng(s + 1))["estimate"] - ref for s in range(seeds)])
    return mc, ms


# ------------------------------------------------------------- 1. 시나리오 정의·정답
def test_scenario_registry_shape():
    assert KEYS == ["circle", "joint_default", "interference", "peak"]
    for sc in SCENARIOS.values():
        assert callable(sc["func"]) and callable(sc["exact"])
        assert isinstance(sc["binary"], bool) and sc["label"] and sc["description"]


@pytest.mark.parametrize("key", KEYS)
def test_exact_matches_large_mc(key):
    """정답이 대규모 일반 MC(200만 점)와 5 표준오차 안에서 일치."""
    v = SCENARIOS[key]["func"](rng(7).random((2_000_000, 2)))
    assert abs(v.mean() - SCENARIOS[key]["exact"]()) < 5 * v.std() / math.sqrt(len(v))


def test_exact_values():
    assert math.isclose(SCENARIOS["circle"]["exact"](), math.pi * 0.15**2)
    assert abs(SCENARIOS["interference"]["exact"]() - 1.0) < 1e-3
    assert math.isclose(SCENARIOS["peak"]["exact"](), 2 * math.pi * PEAK_SIGMA**2, rel_tol=1e-6)
    assert 0.0015 < SCENARIOS["joint_default"]["exact"]() < 0.0018


def test_joint_default_is_binary_and_in_corner():
    f = SCENARIOS["joint_default"]["func"]
    p = rng(0).random((200_000, 2))
    v = f(p)
    assert set(np.unique(v)) <= {0.0, 1.0}
    assert np.all(p[v == 1, 0] < 0.02)  # 회사1 부도 <=> u1 < PD


def test_peak_maximum_at_center():
    f = SCENARIOS["peak"]["func"]
    assert f(np.array([PEAK_CENTER]))[0] == 1.0
    assert f(np.array([[0.9, 0.9]]))[0] < 1e-50


# ------------------------------------------------------------- 2. 예산·구조
@pytest.mark.parametrize("n", [500, 1_000, 10_000, 100_000])
@pytest.mark.parametrize("key", KEYS)
def test_equal_evaluation_budget(n, key):
    f = SCENARIOS[key]["func"]
    assert standard(f, UNIT2, n, rng(0))["evaluations"] == n
    ms = miser(f, UNIT2, n, rng(1))
    assert ms["evaluations"] == n
    assert len(ms["points"]) == len(ms["values"]) <= n


def test_small_budget_is_plain_leaf():
    ms = miser(SCENARIOS["peak"]["func"], UNIT2, 500, rng(0))  # 500 < 2·256
    assert ms["splits"] == [] and ms["depth"] == 0 and len(ms["points"]) == 500


@pytest.mark.parametrize("max_depth", [1, 3, 12])
def test_max_depth_respected(max_depth):
    ms = miser(SCENARIOS["peak"]["func"], UNIT2, 100_000, rng(0), max_depth=max_depth)
    assert ms["depth"] <= max_depth
    assert max(s["depth"] for s in ms["splits"]) < max_depth


def test_samples_inside_bounds_and_values_consistent():
    f = SCENARIOS["interference"]["func"]
    for res in (standard(f, UNIT2, 5_000, rng(0)), miser(f, UNIT2, 5_000, rng(1))):
        p = res["points"]
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert np.allclose(res["values"], f(p))


def test_reproducible_with_same_seed():
    f = SCENARIOS["joint_default"]["func"]
    a, b = miser(f, UNIT2, 20_000, rng(5)), miser(f, UNIT2, 20_000, rng(5))
    assert a["estimate"] == b["estimate"] and np.array_equal(a["points"], b["points"])


@pytest.mark.parametrize("bounds, expected", [
    (UNIT2, 2.5), ([(2.0, 5.0), (-1.0, 1.0)], 2.5 * 6), ([(0.0, 1.0)] * 3, 2.5)])
def test_constant_function_is_exact(bounds, expected):
    assert math.isclose(standard(const(2.5), bounds, 10_000, rng(0))["estimate"],
                        expected, rel_tol=1e-12)
    assert math.isclose(miser(const(2.5), bounds, 10_000, rng(1))["estimate"],
                        expected, rel_tol=1e-12)


# ------------------------------------------------------------- 3. 시연 시나리오의 기대 결과
@pytest.mark.parametrize("key", ["peak", "joint_default"])
def test_miser_clearly_wins_on_concentrated_scenarios(key):
    """별빛·동시 부도: MISER RMSE 가 최소 3배 작다 (확인 당시 약 9배, 17배)."""
    mc, ms = errs(key)
    assert rmse(mc) > 3 * rmse(ms)


def test_interference_miser_loses():
    """간섭 무늬(반례1): 나눌 이득이 없어 MC보다 나쁨 (확인 당시 약 0.54배)."""
    mc, ms = errs("interference", seeds=30)
    assert rmse(ms) > rmse(mc)


def test_circle_miser_wins():
    """원 넓이: 이 구현에서는 MISER 가 낫다 (확인 당시 n=10,000 에서 약 3.3배)."""
    mc, ms = errs("circle", n=10_000, seeds=30)
    assert rmse(ms) < rmse(mc)


@pytest.mark.parametrize("key", KEYS)
def test_estimators_unbiased(key):
    for e in errs(key):
        assert abs(e.mean()) < 6 * e.std(ddof=1) / math.sqrt(len(e))


# ------------------------------------------------------------- 5. 애니메이션 기록
@pytest.mark.parametrize("key", KEYS)
def test_event_recording_does_not_change_the_run(key):
    """events 기록은 난수를 쓰지 않으므로 결과가 완전히 같고, 평가 횟수도 맞는다."""
    f = SCENARIOS[key]["func"]
    a = miser(f, UNIT2, 10_000, rng(43))
    events = []
    b = miser(f, UNIT2, 10_000, rng(43), events=events)
    assert a["estimate"] == b["estimate"]
    assert np.array_equal(a["points"], b["points"])
    assert sum(len(e["points"]) for e in events if e["type"] != "split") == 10_000
    assert sum(e["type"] == "split" for e in events) == len(b["splits"])
