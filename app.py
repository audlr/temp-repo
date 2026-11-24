import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import player


# ---------------------------------------------------------
# Test functions and exact integrals (see scenarios.py)
# ---------------------------------------------------------
from scenarios import DEMO, SCENARIOS

SCENARIO_KEYS = list(SCENARIOS)


def exact_integral(test_name):
    return SCENARIOS[test_name]["exact"]()


def evaluate_function(points, test_name):
    return SCENARIOS[test_name]["func"](points)


def is_binary(test_name):
    """0/1 indicator scenarios show inside/outside counts and colours."""
    return SCENARIOS[test_name]["binary"]


# ---------------------------------------------------------
# Standard Monte Carlo
# ---------------------------------------------------------
def standard_monte_carlo(f, bounds, n, rng):
    lows = np.array([a for a, _ in bounds], dtype=float)
    highs = np.array([b for _, b in bounds], dtype=float)
    volume = float(np.prod(highs - lows))

    points = rng.uniform(lows, highs, size=(n, len(bounds)))
    values = f(points)
    estimate = volume * float(np.mean(values))

    return {
        "estimate": estimate,
        "points": points,
        "values": values,
        "evaluations": n,
        "depth": 0,
        "splits": [],
    }


# ---------------------------------------------------------
# MISER Monte Carlo
# ---------------------------------------------------------
def miser_monte_carlo(
    f,
    bounds,
    n,
    rng,
    min_samples=256,
    max_depth=12,
    pilot_fraction=0.10,
    events=None,
):
    """
    Recursive, variance-guided stratified Monte Carlo.

    At each node:
      1. Draw pilot samples.
      2. Estimate the variance in both halves for every dimension.
      3. Choose the split with the smallest estimated variance score.
      4. Allocate the remaining sample budget in proportion to
         subregion_volume * estimated_standard_deviation.
      5. Recurse into both subregions.

    The pilot samples guide the allocation and are not included in
    the final estimate. The total number of function evaluations,
    including pilot samples, is kept at approximately n.

    events: optional list. When given, every pilot draw, split decision and
    leaf draw is appended to it in evaluation order (for the animation).
    Recording does not consume random numbers, so results are unchanged.
    """
    total_evaluations = 0
    all_points = []
    all_values = []
    split_records = []
    deepest_level = 0

    def plain_leaf(box, sample_count, depth):
        nonlocal total_evaluations, deepest_level

        lows = np.array([a for a, _ in box], dtype=float)
        highs = np.array([b for _, b in box], dtype=float)
        volume = float(np.prod(highs - lows))

        pts = rng.uniform(lows, highs, size=(sample_count, len(box)))
        vals = f(pts)

        total_evaluations += sample_count
        deepest_level = max(deepest_level, depth)
        all_points.append(pts)
        all_values.append(vals)
        if events is not None:
            events.append({"type": "leaf", "box": box, "depth": depth,
                           "points": pts, "values": vals})

        return volume * float(np.mean(vals))

    def recurse(box, sample_count, depth):
        nonlocal total_evaluations, deepest_level

        dimensions = len(box)
        if (
            sample_count < 2 * min_samples
            or depth >= max_depth
            or dimensions == 0
        ):
            return plain_leaf(box, sample_count, depth)

        # Reserve part of this node's budget for pilot sampling.
        pilot_n = max(32, int(sample_count * pilot_fraction))
        pilot_n = min(pilot_n, sample_count - 2 * min_samples)

        if pilot_n <= 0:
            return plain_leaf(box, sample_count, depth)

        lows = np.array([a for a, _ in box], dtype=float)
        highs = np.array([b for _, b in box], dtype=float)
        pilot_points = rng.uniform(lows, highs, size=(pilot_n, dimensions))
        pilot_values = f(pilot_points)
        total_evaluations += pilot_n
        if events is not None:
            events.append({"type": "pilot", "box": box, "depth": depth,
                           "points": pilot_points, "values": pilot_values})

        # Test a midpoint split in every dimension. Select the dimension
        # with the smallest estimated sum of volume-weighted standard deviations.
        best_dimension = None
        best_score = float("inf")
        best_std_pair = None

        for dim in range(dimensions):
            midpoint = (lows[dim] + highs[dim]) / 2.0
            left_mask = pilot_points[:, dim] <= midpoint
            right_mask = ~left_mask

            # Require enough pilot observations on both sides.
            if left_mask.sum() < 2 or right_mask.sum() < 2:
                continue

            std_left = float(np.std(pilot_values[left_mask], ddof=1))
            std_right = float(np.std(pilot_values[right_mask], ddof=1))

            left_lengths = highs - lows
            right_lengths = highs - lows
            left_lengths = left_lengths.copy()
            right_lengths = right_lengths.copy()
            left_lengths[dim] = midpoint - lows[dim]
            right_lengths[dim] = highs[dim] - midpoint

            left_volume = float(np.prod(left_lengths))
            right_volume = float(np.prod(right_lengths))

            score = left_volume * std_left + right_volume * std_right
            if score < best_score:
                best_score = score
                best_dimension = dim
                best_std_pair = (std_left, std_right)

        if best_dimension is None:
            return plain_leaf(box, sample_count - pilot_n, depth)

        dim = best_dimension
        midpoint = (lows[dim] + highs[dim]) / 2.0

        left_box = list(box)
        right_box = list(box)
        left_box[dim] = (box[dim][0], midpoint)
        right_box[dim] = (midpoint, box[dim][1])

        left_volume = float(np.prod([
            b - a for a, b in left_box
        ]))
        right_volume = float(np.prod([
            b - a for a, b in right_box
        ]))

        std_left, std_right = best_std_pair
        weight_left = left_volume * std_left
        weight_right = right_volume * std_right
        weight_sum = weight_left + weight_right

        remaining = sample_count - pilot_n
        if weight_sum <= 1e-15:
            left_budget = int(round(remaining * left_volume / (left_volume + right_volume)))
        else:
            left_budget = int(round(remaining * weight_left / weight_sum))

        left_budget = max(min_samples, min(remaining - min_samples, left_budget))
        right_budget = remaining - left_budget

        split_records.append({
            "depth": depth,
            "dimension": dim,
            "coordinate": midpoint,
            "bounds": box,
        })
        if events is not None:
            events.append({"type": "split", "box": box, "depth": depth,
                           "dimension": dim, "coordinate": midpoint,
                           "std": (std_left, std_right),
                           "budget": (left_budget, right_budget)})

        left_estimate = recurse(left_box, left_budget, depth + 1)
        right_estimate = recurse(right_box, right_budget, depth + 1)

        return left_estimate + right_estimate

    estimate = recurse(bounds, n, 0)

    points = np.vstack(all_points) if all_points else np.empty((0, len(bounds)))
    values = np.concatenate(all_values) if all_values else np.empty(0)

    return {
        "estimate": estimate,
        "points": points,
        "values": values,
        "evaluations": total_evaluations,
        "depth": deepest_level,
        "splits": split_records,
    }


# ---------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------
st.set_page_config(
    page_title="Monte Carlo vs MISER",
    page_icon="📊",
    layout="wide",
)

st.title("Monte Carlo Integration: Standard vs MISER")
st.caption(
    "Compare uniform random sampling with recursive, variance-guided "
    "sample allocation."
)

def apply_demo_preset():
    """Choosing a scenario sets the seed and sample size that show it best."""
    preset = DEMO[st.session_state["test_name"]]["main"]
    st.session_state["n_samples"] = preset["n"]
    st.session_state["seed"] = preset["seed"]


if "test_name" not in st.session_state:
    st.session_state["test_name"] = SCENARIO_KEYS[0]
    apply_demo_preset()

with st.sidebar:
    st.header("Experiment Settings")
    test_name = st.selectbox(
        "Test Function",
        SCENARIO_KEYS,
        key="test_name",
        on_change=apply_demo_preset,
        format_func=lambda k: SCENARIOS[k]["label"],
        help="Four demo scenarios: basics (circle), strength (joint default), "
             "limit (interference), risk (starlight peak). Choosing one sets the "
             "demo seed and sample size below.",
    )
    n_samples = st.number_input(
        "Number of Samples",
        min_value=1000,
        max_value=500000,
        step=1000,
        key="n_samples",
    )
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=2**32 - 1,
        step=1,
        key="seed",
    )

    with st.expander("Advanced MISER Settings"):
        min_samples = st.number_input(
            "Minimum samples per leaf",
            min_value=32,
            max_value=5000,
            value=256,
            step=32,
        )
        max_depth = st.slider("Maximum recursion depth", 1, 20, 12)
        pilot_fraction = st.slider(
            "Pilot sample fraction",
            min_value=0.05,
            max_value=0.30,
            value=0.10,
            step=0.05,
        )

    run_button = st.button("Run Comparison", type="primary", use_container_width=True)

    st.divider()
    st.markdown(f"**Target:** {SCENARIOS[test_name]['description']}")
    st.latex(rf"I = {exact_integral(test_name):.6f}")

if run_button:
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    f = lambda pts: evaluate_function(pts, test_name)
    actual = exact_integral(test_name)

    # Independent generators make each run reproducible.
    plain_rng = np.random.default_rng(int(seed))
    miser_rng = np.random.default_rng(int(seed) + 1)

    with st.spinner("Running both integration algorithms..."):
        start = time.perf_counter()
        plain_result = standard_monte_carlo(
            f, bounds, int(n_samples), plain_rng
        )
        plain_time = time.perf_counter() - start

        start = time.perf_counter()
        miser_result = miser_monte_carlo(
            f,
            bounds,
            int(n_samples),
            miser_rng,
            min_samples=int(min_samples),
            max_depth=int(max_depth),
            pilot_fraction=float(pilot_fraction),
        )
        miser_time = time.perf_counter() - start

    st.session_state["comparison"] = {
        "test_name": test_name,
        "actual": actual,
        "plain": plain_result,
        "miser": miser_result,
        "plain_time": plain_time,
        "miser_time": miser_time,
        "requested_samples": int(n_samples),
        "settings": {"seed": int(seed), "min_samples": int(min_samples),
                     "max_depth": int(max_depth),
                     "pilot_fraction": float(pilot_fraction)},
    }

if "comparison" not in st.session_state:
    st.info("Set the experiment parameters in the sidebar and click **Run Comparison**.")
    st.stop()

result = st.session_state["comparison"]
if result["test_name"] != test_name:
    st.warning("The selected test function differs from the displayed run. Click Run Comparison to update the results.")
    st.stop()

actual = result["actual"]
plain = result["plain"]
miser = result["miser"]

plain_error = abs(plain["estimate"] - actual)
miser_error = abs(miser["estimate"] - actual)

st.subheader("Run Results")

def highlighted_metric(label, value):
    """A metric card with a highlighted background: the number to look at."""
    st.markdown(
        '<div style="background:#fff4c2;border:2px solid #f59e0b;border-radius:10px;'
        'padding:6px 12px;margin-bottom:12px">'
        f'<div style="font-size:0.875rem;color:#555">{label}</div>'
        f'<div style="font-size:2.25rem;line-height:1.3">{value}</div></div>',
        unsafe_allow_html=True,
    )


left, right = st.columns(2)

with left:
    st.markdown("### 🔵 Plain Monte Carlo")
    st.metric("Estimated Integral", f'{plain["estimate"]:.8f}')
    highlighted_metric("Absolute Error", f"{plain_error:.8f}")
    st.metric("Execution Time", f'{result["plain_time"]:.4f} s')
    st.caption(
        f'Evaluations: {plain["evaluations"]:,}'
        + (f' | Inside: {int(np.sum(plain["values"] > 0)):,} | '
           f'Outside: {int(np.sum(plain["values"] == 0)):,}'
           if is_binary(test_name) else "")
    )

with right:
    st.markdown("### 🟠 MISER Algorithm")
    st.metric("Estimated Integral", f'{miser["estimate"]:.8f}')
    highlighted_metric("Absolute Error", f"{miser_error:.8f}")
    st.metric("Execution Time", f'{result["miser_time"]:.4f} s')
    st.caption(
        f'Evaluations: {miser["evaluations"]:,} | '
        + (f'Inside: {int(np.sum(miser["values"] > 0)):,} | '
           f'Outside: {int(np.sum(miser["values"] == 0)):,} | '
           if is_binary(test_name) else "")
        + f'Depth: {miser["depth"]}'
    )

st.markdown(f"**True integral:** `{actual:.10f}`")

# 2-D replay of both runs with highlighted MISER decisions (player.py).
st.subheader("Live Sampling Animation")
st.caption(
    "Both runs are replayed in the order the samples were evaluated. "
    "MISER pauses briefly at each split decision: the split line appears, each half "
    "shows its share of the remaining samples, and the half that receives more "
    "samples flashes once. The grey background is f."
)
# Re-run MISER with the same seed only to record its events; the recorded run is
# identical to the timed one above (recording consumes no random numbers).
_events = []
miser_monte_carlo(
    lambda pts: evaluate_function(pts, test_name),
    [(0.0, 1.0), (0.0, 1.0)],
    int(result["requested_samples"]),
    np.random.default_rng(result["settings"]["seed"] + 1),
    min_samples=result["settings"]["min_samples"],
    max_depth=result["settings"]["max_depth"],
    pilot_fraction=result["settings"]["pilot_fraction"],
    events=_events,
)
player.render(player.build_payload(
    plain, _events, lambda pts: evaluate_function(pts, test_name),
    is_binary(test_name), actual, plain["estimate"], miser["estimate"],
))

# Comparison table and chart
st.subheader("Performance Comparison")
comparison_df = pd.DataFrame({
    "Algorithm": ["Plain Monte Carlo", "MISER"],
    "Estimate": [plain["estimate"], miser["estimate"]],
    "True Integral": [actual, actual],
    "Absolute Error": [plain_error, miser_error],
    "Execution Time (s)": [result["plain_time"], result["miser_time"]],
    "Function Evaluations": [plain["evaluations"], miser["evaluations"]],
    "Maximum Depth": [plain["depth"], miser["depth"]],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

error_fig = go.Figure()
error_fig.add_trace(go.Bar(
    x=["Plain Monte Carlo", "MISER"],
    y=[plain_error, miser_error],
    text=[f"{plain_error:.3g}", f"{miser_error:.3g}"],
    textposition="outside",
))
error_fig.update_layout(
    title="Absolute Error Comparison",
    yaxis_title="Absolute Error",
    height=350,
    showlegend=False,
)
st.plotly_chart(error_fig, use_container_width=True)

time_fig = go.Figure()
time_fig.add_trace(go.Bar(
    x=["Plain Monte Carlo", "MISER"],
    y=[result["plain_time"], result["miser_time"]],
    text=[f'{result["plain_time"]:.4f}s', f'{result["miser_time"]:.4f}s'],
    textposition="outside",
))
time_fig.update_layout(
    title="Execution Time Comparison",
    yaxis_title="Seconds",
    height=350,
    showlegend=False,
)
st.plotly_chart(time_fig, use_container_width=True)

