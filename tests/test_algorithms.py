"""
Tests for the algorithms, the instrumentation, and the Streamlit app.

Monte Carlo is random: tests never require a single run to beat a fixed error
threshold. Instead they check exact reproducibility, budgets, structural
invariants, and statistical bounds that are many standard errors wide.
"""

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import (  # noqa: E402
    INTEGRANDS, errors, make_bounds, regions_at_depth, repeated, sweep,
    validate_bounds,
)
from miser import run_miser, run_plain_monte_carlo  # noqa: E402
from visualization import plots  # noqa: E402

X2 = INTEGRANDS["x2"]
B01 = [[0.0, 1.0]]


# ------------------------------------------------- backward compatibility
@pytest.mark.parametrize("seed, mc_est, miser_est, n_splits", [
    (42, 0.020995272947157837, 0.020054804966237755, 21),
    (7, 0.020796691989287554, 0.020124908676613297, 22),
])
def test_default_call_matches_original_implementation(seed, mc_est, miser_est,
                                                      n_splits):
    """Values recorded from the original app.py (commit a4b6e31)."""
    assert run_plain_monte_carlo(10_000, seed)["estimate"] == mc_est
    ms = run_miser(10_000, seed + 1, 300)
    assert ms["estimate"] == miser_est
    assert len(ms["splits"]) == n_splits


# ------------------------------------------------- basic runs / budgets
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_budgets_are_equal(n):
    mc = run_plain_monte_carlo(n, 1, X2.func, B01)
    ms = run_miser(n, 2, 300, X2.func, B01)
    assert mc["evaluations"] == len(mc["points"]) == n
    assert ms["evaluations"] == len(ms["points"]) == len(ms["kinds"]) == n


def test_x2_estimates_are_statistically_plausible():
    # sd of x^2 on U(0,1) is sqrt(4/45) ≈ 0.298; 10 standard errors at n=10k.
    mc = run_plain_monte_carlo(10_000, 42, X2.func, B01)
    ms = run_miser(10_000, 43, 300, X2.func, B01)
    tol = 10 * 0.298 / math.sqrt(10_000)
    assert abs(mc["estimate"] - 1 / 3) < tol
    assert abs(ms["estimate"] - 1 / 3) < tol


def test_reproducible_with_same_seed():
    a = run_miser(20_000, 5, 300, X2.func, B01)
    b = run_miser(20_000, 5, 300, X2.func, B01)
    assert a["estimate"] == b["estimate"]
    assert np.array_equal(a["points"], b["points"])
    c = run_plain_monte_carlo(20_000, 5, X2.func, B01)
    d = run_plain_monte_carlo(20_000, 5, X2.func, B01)
    assert c["estimate"] == d["estimate"]


def test_samples_inside_bounds_and_values_consistent():
    bounds = [[-2.0, 3.0]]
    f = INTEGRANDS["gauss"]
    for res in (run_plain_monte_carlo(5_000, 0, f.func, bounds),
                run_miser(5_000, 0, 300, f.func, bounds)):
        x = res["points"][:, 0]
        assert x.min() >= -2.0 and x.max() <= 3.0
        # The plotted values are exactly f at the plotted points.
        assert np.allclose(res["values"], f.scalar(x))


# ------------------------------------------------- MISER instrumentation
def test_miser_tree_is_consistent():
    ms = run_miser(50_000, 3, 300, X2.func, B01)
    nodes = ms["nodes"]
    by_id = {nd["id"]: nd for nd in nodes}
    leaves = [nd for nd in nodes if nd["leaf"]]
    assert len(nodes) == len(ms["splits"]) + len(leaves)

    # Leaves tile the interval exactly.
    edges = sorted((nd["box"][0][0], nd["box"][0][1]) for nd in leaves)
    assert edges[0][0] == 0.0 and edges[-1][1] == 1.0
    assert all(edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1))

    # Budget conservation: parent = pilot cost + children.
    for ev in ms["splits"]:
        children = [nd for nd in nodes if nd["parent"] == ev["node_id"]]
        assert len(children) == 2
        assert by_id[ev["node_id"]]["count"] == (
            ev["pilot_cost"] + sum(c["count"] for c in children))

    # Production samples per leaf == leaf budget.
    prod = ms["points"][ms["kinds"] == 1, 0]
    for nd in leaves:
        lo, hi = nd["box"][0]
        inside = np.sum((prod >= lo) & (prod < hi)) if hi < 1.0 else \
            np.sum((prod >= lo) & (prod <= hi))
        assert inside == nd["count"]

    # Sum of leaf estimates is the returned estimate.
    assert math.isclose(sum(nd["estimate"] for nd in leaves), ms["estimate"],
                        rel_tol=1e-12)


def test_regions_at_depth_cover_domain():
    ms = run_miser(50_000, 3, 300, X2.func, B01)
    max_depth = max(nd["depth"] for nd in ms["nodes"])
    for d in range(max_depth + 1):
        regs = regions_at_depth(ms["nodes"], d)
        total = sum(r["box"][0][1] - r["box"][0][0] for r in regs)
        assert math.isclose(total, 1.0)
    assert len(regions_at_depth(ms["nodes"], 0)) == 1


def test_2d_village_instrumentation():
    f = INTEGRANDS["village"]
    ms = run_miser(20_000, 1, 300, f.func, make_bounds(f, 0, 1))
    area = sum((nd["box"][0][1] - nd["box"][0][0])
               * (nd["box"][1][1] - nd["box"][1][0])
               for nd in ms["nodes"] if nd["leaf"])
    assert math.isclose(area, 1.0)


# ------------------------------------------------- references / errors
def test_reference_values():
    assert math.isclose(X2.reference(B01), 1 / 3)
    assert math.isclose(INTEGRANDS["sin"].reference(B01), 1 - math.cos(1))
    assert math.isclose(INTEGRANDS["cos"].reference(B01), math.sin(1))
    assert math.isclose(INTEGRANDS["gauss"].reference(B01), 0.7468241328124271)
    # Village: analytic value close to the original grid approximation.
    v = INTEGRANDS["village"]
    from miser import reference_integral
    assert abs(v.reference(make_bounds(v, 0, 1)) - reference_integral()) < 1e-4


def test_relative_error_skipped_near_zero_reference():
    ref = INTEGRANDS["sin"].reference([[-1.0, 1.0]])
    assert abs(ref) < 1e-12
    assert errors(0.01, ref)[1] is None
    assert errors(0.30, 1 / 3)[1] == pytest.approx(0.1)


def test_validate_bounds():
    assert validate_bounds(0, 1) is None
    assert validate_bounds(1, 1) is not None
    assert validate_bounds(2, 1) is not None
    assert validate_bounds(float("nan"), 1) is not None
    with pytest.raises(ValueError):
        run_plain_monte_carlo(1000, 0, X2.func, [[1.0, 0.0]])


# ------------------------------------------------- sweeps
def test_sweep_and_repeated():
    rows = sweep(X2, B01, 42, 300, sizes=(1_000, 5_000))
    assert len(rows) == 4 and all(r["time"] > 0 for r in rows)
    rep = repeated(X2, B01, 42, 300, repeats=5, sizes=(1_000, 5_000))
    assert len(rep) == 4
    for r in rep:
        assert r["rmse"] >= r["mae"] >= 0 and r["std"] > 0
        # RMSE within a wide statistical bound (10 SE of a plain MC run).
        assert r["rmse"] < 10 * 0.298 / math.sqrt(r["n"])


# ------------------------------------------------- plots use real data
def test_plot_uses_actual_samples():
    mc = run_plain_monte_carlo(20_000, 0, X2.func, B01)
    fig = plots.sampling_figure_1d(X2, B01, mc, "Standard Monte Carlo",
                                   3_000, [0, 1], 1000)
    scat = [t for t in fig.data if t.name == "Samples"][0]
    xs = np.asarray(scat.x)
    assert len(xs) == 3_000
    assert np.isin(xs, mc["points"][:, 0]).all()
    hist = [t for t in fig.data if t.type == "bar"][0]
    assert int(np.sum(hist.y)) == 20_000  # histogram uses every sample


# ------------------------------------------------- Streamlit app smoke tests
def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)


def test_app_initial_render():
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "Monte Carlo vs MISER Integration"
    # Only the sidebar run button exists; the player has no controls of its own.
    assert [b.label for b in at.button] == ["Run Comparison"]


def test_app_run_hides_stats_until_animation_finishes():
    at = _app().run()
    at.number_input[0].set_value(20_000).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    run = at.session_state["run"]
    assert run["mc"]["evaluations"] == run["ms"]["evaluations"] == 20_000
    assert run["payload"]["n"] == 20_000
    # The browser component cannot run here, so statistics stay hidden.
    assert not any("Final estimate" in m.label for m in at.metric)
    # Simulate the component reporting "animation finished" for this run.
    at.session_state["anim_done"] = run["run_id"]
    at.run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert labels.count("Final estimate") == 2 and "Splits / depth" in labels
    rep = [b for b in at.button if b.label == "Run Repeated Experiments"][0]
    rep.click().run()
    assert not at.exception, at.exception
    assert at.success
    # A new run resets the "finished" state.
    at.sidebar.button[0].click().run()
    assert at.session_state["run"]["run_id"] == run["run_id"] + 1
    assert not any("Final estimate" in m.label for m in at.metric)


# ------------------------------------------------- live replay payload
def test_live_payload_replays_the_real_run():
    from visualization.live import build_payload
    f = INTEGRANDS["circle"]
    bounds = make_bounds(f, 0, 1)
    mc = run_plain_monte_carlo(20_000, 42, f.func, bounds)
    ms = run_miser(20_000, 43, 300, f.func, bounds)
    p = build_payload(1, f, mc, ms, f.reference(bounds))
    m = p["ms"]
    # Samples are sent in evaluation order, unchanged (up to display rounding).
    assert p["n"] == len(p["mc"]["x"]) == len(m["x"]) == len(m["kind"]) == 20_000
    assert np.allclose(m["x"], ms["points"][:, 0], atol=1e-6)
    assert np.allclose(m["y"], ms["points"][:, 1], atol=1e-6)
    assert np.allclose(p["mc"]["x"], mc["points"][:, 0], atol=1e-6)
    # Inside/outside colours come from the algorithm's own values = real judgement.
    for res, sent in ((mc, p["mc"]), (ms, m)):
        pts = res["points"]
        d2 = (pts[:, 0] - 0.70) ** 2 + (pts[:, 1] - 0.30) ** 2
        assert (np.asarray(sent["v"]) == (d2 <= 0.15 ** 2)).all()
    # Plain MC estimate is exactly the inside ratio (square area 1).
    assert mc["estimate"] == pytest.approx(sum(p["mc"]["v"]) / 20_000, rel=1e-12)
    kind = np.asarray(m["kind"])
    # Each leaf owns a contiguous block of production samples in run order.
    covered = np.zeros(20_000, dtype=bool)
    for start, count, *_ in m["leaves"]:
        assert (kind[start:start + count] == 1).all()
        covered[start:start + count] = True
    assert (covered == (kind == 1)).all()
    # MISER's estimate is the weighted sum over regions, not the inside ratio.
    assert math.isclose(sum(l[8] for l in m["leaves"]), ms["estimate"], rel_tol=1e-12)
    weighted = sum(l[2] * np.mean(np.asarray(m["v"])[l[0]:l[0] + l[1]]) for l in m["leaves"])
    assert math.isclose(weighted, ms["estimate"], rel_tol=1e-9)
    # Split / node events are in non-decreasing evaluation order, and a split is
    # shown only after its own pilot samples were evaluated.
    after = [s[0] for s in m["splits"]]
    assert after == sorted(after) and len(after) == len(ms["splits"])
    created = [nd[0] for nd in m["nodes"]]
    assert created == sorted(created) and created[0] == 0
    for ev in ms["splits"]:
        assert all(k == 0 for k in kind[ev["after_eval"] - ev["pilot_cost"]:ev["after_eval"]])


def test_circle_reference_and_estimates():
    f = INTEGRANDS["circle"]
    bounds = make_bounds(f, 0, 1)
    assert math.isclose(f.reference(bounds), math.pi * 0.15 ** 2)
    mc = run_plain_monte_carlo(100_000, 1, f.func, bounds)
    ms = run_miser(100_000, 2, 300, f.func, bounds)
    # sd of the indicator ≈ sqrt(p(1-p)) ≈ 0.257; 10 standard errors.
    tol = 10 * 0.257 / math.sqrt(100_000)
    assert abs(mc["estimate"] - f.reference(bounds)) < tol
    assert abs(ms["estimate"] - f.reference(bounds)) < tol


# 시나리오 4개의 정답값과 동작을 확인하기 위해 추가
UNIT2 = [[0.0, 1.0], [0.0, 1.0]]


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


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
