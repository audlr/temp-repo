import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scenarios import DEMO, SCENARIOS

# ---------------------------------------------------------
# Test functions and exact integrals
# ---------------------------------------------------------
CENTER = (0.70, 0.30)
RADIUS = 0.15
SIGMA = 0.06


def circle_indicator(points):
    """1 inside the circle, 0 outside."""
    dx = points[:, 0] - CENTER[0]
    dy = points[:, 1] - CENTER[1]
    return ((dx * dx + dy * dy) <= RADIUS**2).astype(float)


def gaussian_function(points):
    """Narrow 2-D Gaussian used in the original Python code."""
    dx = points[:, 0] - CENTER[0]
    dy = points[:, 1] - CENTER[1]
    return np.exp(-(dx * dx + dy * dy) / (2 * SIGMA**2))


def gaussian_1d_integral(a, b, mu, sigma=SIGMA):
    return sigma * math.sqrt(math.pi / 2) * (
        math.erf((b - mu) / (math.sqrt(2) * sigma))
        - math.erf((a - mu) / (math.sqrt(2) * sigma))
    )


# 원/가우시안 대신 시나리오 4개를 쓰기 위해 수정
def exact_integral(test_name):
    return SCENARIOS[test_name]["exact"]()


def evaluate_function(points, test_name):
    return SCENARIOS[test_name]["func"](points)


def is_binary(test_name):
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

# 시나리오별 시연용 Seed·샘플 수를 자동으로 넣기 위해 추가
def apply_demo_preset():
    preset = DEMO[st.session_state["test_name"]]["main"]
    st.session_state["n_samples"] = preset["n"]
    st.session_state["seed"] = preset["seed"]


if "test_name" not in st.session_state:
    st.session_state["test_name"] = list(SCENARIOS)[0]
    apply_demo_preset()

with st.sidebar:
    st.header("Experiment Settings")
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
    test_name = st.selectbox(
        "Test Function",
        list(SCENARIOS),
        format_func=lambda k: SCENARIOS[k]["label"],
        key="test_name",
        on_change=apply_demo_preset,
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
# 결과(Absolute Error) 하이라이트를 위해 추가
st.markdown(
    """<style>
    .st-key-hlplainabs, .st-key-hlmiserabs, .st-key-hlerrorchart {
        background: rgba(245, 158, 11, 0.18); border: 2px solid #f59e0b;
        border-radius: 10px; padding: 4px 12px;
    }
    </style>""",
    unsafe_allow_html=True,
)
left, right = st.columns(2)

with left:
    st.markdown("### 🔵 Plain Monte Carlo")
    st.metric("Estimated Integral", f'{plain["estimate"]:.8f}')
    with st.container(key="hlplainabs"):
        st.metric("Absolute Error", f"{plain_error:.8f}")
    st.metric("Execution Time", f'{result["plain_time"]:.4f} s')
    st.caption(
        f'Evaluations: {plain["evaluations"]:,} | '
        f'Inside: {int(np.sum(plain["values"] > 0)):,} | '
        f'Outside: {int(np.sum(plain["values"] == 0)):,}'
    )

with right:
    st.markdown("### 🟠 MISER Algorithm")
    st.metric("Estimated Integral", f'{miser["estimate"]:.8f}')
    with st.container(key="hlmiserabs"):
        st.metric("Absolute Error", f"{miser_error:.8f}")
    st.metric("Execution Time", f'{result["miser_time"]:.4f} s')
    st.caption(
        f'Evaluations: {miser["evaluations"]:,} | '
        f'Inside: {int(np.sum(miser["values"] > 0)):,} | '
        f'Outside: {int(np.sum(miser["values"] == 0)):,} | '
        f'Depth: {miser["depth"]}'
    )

st.markdown(f"**True integral:** `{actual:.10f}`")

# Client-side Plotly animation (avoids repeatedly rerendering the Streamlit page)
st.subheader("Live Sampling Animation")
st.caption(
    "Press Play to reveal samples progressively. The animation runs inside the "
    "3-D chart, so Streamlit does not redraw the entire page at every frame."
)

animation_seconds = st.select_slider(
    "Animation duration",
    options=[3, 5, 8, 12],
    value=8,
    format_func=lambda x: f"{x} seconds",
)

grid_axis = np.linspace(0, 1, 151)  # 좁은 봉우리·줄무늬가 보이게 수정
xx, yy = np.meshgrid(grid_axis, grid_axis)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
zz = evaluate_function(grid_points, test_name).reshape(xx.shape)

# Use a bounded number of visible points to keep the browser animation smooth.
animation_limit = min(1200, len(plain["points"]), len(miser["points"]))
plain_indices = np.linspace(0, len(plain["points"]) - 1, animation_limit, dtype=int)
miser_indices = np.linspace(0, len(miser["points"]) - 1, animation_limit, dtype=int)
plain_anim_points = plain["points"][plain_indices]
plain_anim_values = plain["values"][plain_indices]
miser_anim_points = miser["points"][miser_indices]
miser_anim_values = miser["values"][miser_indices]

fig_anim = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("🔵 Plain Monte Carlo", "🟠 MISER Algorithm"),
    horizontal_spacing=0.02,
)

# Static function surfaces are added only once.
for col in (1, 2):
    fig_anim.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale="Viridis",
            opacity=0.76,
            showscale=(col == 2),
            colorbar=dict(title="f(x, y)", len=0.72, x=1.02) if col == 2 else None,
            name="Function surface",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>f=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=col,
    )

def split_sample_groups(points, values):
    """Return outside and inside point arrays (for Circle); Gaussian points are all active."""
    if is_binary(test_name):
        inside_mask = values > 0
    else:
        inside_mask = np.ones(len(values), dtype=bool)
    return (
        (points[~inside_mask], values[~inside_mask]),
        (points[inside_mask], values[inside_mask]),
    )

# Four initially empty scatter traces are updated by Plotly frames.
for col, color, label in [
    (1, "#3977C3", "Plain MC"),
    (2, "#F28E2B", "MISER"),
]:
    outside_name = f"{label} outside"
    inside_name = f"{label} samples" if not is_binary(test_name) else f"{label} inside"
    fig_anim.add_trace(
        go.Scatter3d(
            x=[], y=[], z=[], mode="markers",
            name=outside_name,
            marker=dict(color="#A0A7B4", size=2.5, opacity=0.48),
            showlegend=(col == 1 and is_binary(test_name)),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>f=%{z:.4f}<extra></extra>",
        ),
        row=1, col=col,
    )
    fig_anim.add_trace(
        go.Scatter3d(
            x=[], y=[], z=[], mode="markers",
            name=inside_name,
            marker=dict(color=color, size=3.2, opacity=0.9),
            showlegend=(col == 1),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>f=%{z:.4f}<extra></extra>",
        ),
        row=1, col=col,
    )

# MISER split boundaries remain static throughout playback.
z_top = max(float(np.max(zz)), 1.0 if is_binary(test_name) else 0.05)
for split in miser["splits"]:
    box = split["bounds"]
    dim, coordinate = split["dimension"], split["coordinate"]
    if dim == 0:
        x_line, y_line = [coordinate, coordinate], [box[1][0], box[1][1]]
    else:
        x_line, y_line = [box[0][0], box[0][1]], [coordinate, coordinate]
    fig_anim.add_trace(
        go.Scatter3d(
            x=x_line,
            y=y_line,
            z=[z_top * 1.02, z_top * 1.02],
            mode="lines",
            line=dict(color="#D97706", width=3, dash="dash"),
            name="MISER split",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=2,
    )

# Build frames once; Plotly animates only the four point traces in the browser.
frame_count = 30
frames = []
for frame_no in range(frame_count + 1):
    fraction = frame_no / frame_count
    count = max(0, int(animation_limit * fraction))

    (p_out_pts, p_out_vals), (p_in_pts, p_in_vals) = split_sample_groups(
        plain_anim_points[:count], plain_anim_values[:count]
    )
    (m_out_pts, m_out_vals), (m_in_pts, m_in_vals) = split_sample_groups(
        miser_anim_points[:count], miser_anim_values[:count]
    )

    # Display playback elapsed time as a deterministic timeline label.
    elapsed_label = animation_seconds * fraction
    status_title = (
        f"Sampling progress: {fraction * 100:.0f}%"
        f"  |  Elapsed: {elapsed_label:.1f}s / {animation_seconds:.1f}s"
        f"  |  Plain: {count:,}/{animation_limit:,}"
        f"  |  MISER: {count:,}/{animation_limit:,}"
    )

    frames.append(
        go.Frame(
            name=str(frame_no),
            data=[
                go.Scatter3d(x=p_out_pts[:, 0], y=p_out_pts[:, 1], z=p_out_vals),
                go.Scatter3d(x=p_in_pts[:, 0], y=p_in_pts[:, 1], z=p_in_vals),
                go.Scatter3d(x=m_out_pts[:, 0], y=m_out_pts[:, 1], z=m_out_vals),
                go.Scatter3d(x=m_in_pts[:, 0], y=m_in_pts[:, 1], z=m_in_vals),
            ],
            traces=[2, 3, 4, 5],
            layout=go.Layout(title_text=status_title),
        )
    )

fig_anim.frames = frames

# Show a small initial batch immediately so the chart is never blank
# while the user is deciding whether to press Play.
initial_count = max(1, animation_limit // frame_count)
(p0_out, p0_out_v), (p0_in, p0_in_v) = split_sample_groups(
    plain_anim_points[:initial_count], plain_anim_values[:initial_count]
)
(m0_out, m0_out_v), (m0_in, m0_in_v) = split_sample_groups(
    miser_anim_points[:initial_count], miser_anim_values[:initial_count]
)
fig_anim.data[2].x, fig_anim.data[2].y, fig_anim.data[2].z = (
    p0_out[:, 0], p0_out[:, 1], p0_out_v
)
fig_anim.data[3].x, fig_anim.data[3].y, fig_anim.data[3].z = (
    p0_in[:, 0], p0_in[:, 1], p0_in_v
)
fig_anim.data[4].x, fig_anim.data[4].y, fig_anim.data[4].z = (
    m0_out[:, 0], m0_out[:, 1], m0_out_v
)
fig_anim.data[5].x, fig_anim.data[5].y, fig_anim.data[5].z = (
    m0_in[:, 0], m0_in[:, 1], m0_in_v
)

scene_style = dict(
    xaxis=dict(title="X", range=[0, 1]),
    yaxis=dict(title="Y", range=[0, 1]),
    zaxis=dict(title="f(x, y)", rangemode="tozero"),
    aspectmode="cube",
    camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
)
fig_anim.update_layout(
    title=dict(
        text=f"Sampling progress: 0%  |  Elapsed: 0.0s / {animation_seconds:.1f}s",
        x=0.5,
        xanchor="center",
    ),
    height=700,
    margin=dict(l=0, r=30, t=100, b=70),
    legend=dict(orientation="h", y=-0.08),
    scene=scene_style,
    scene2=scene_style,
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.02,
            y=1.12,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": int(animation_seconds * 1000 / frame_count), "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate",
                        },
                    ],
                ),
                dict(
                    label="❚❚ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                ),
            ],
        )
    ],
    sliders=[
        dict(
            active=0,
            x=0.16,
            y=-0.02,
            len=0.78,
            currentvalue={"prefix": "Frame: "},
            steps=[
                dict(
                    label=str(i),
                    method="animate",
                    args=[
                        [str(i)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                )
                for i in range(frame_count + 1)
            ],
        )
    ],
)

st.plotly_chart(
    fig_anim,
    use_container_width=True,
    config={"displayModeBar": True, "scrollZoom": True},
)

st.caption(
    f"Each algorithm displays up to {animation_limit:,} representative samples. "
    "Use Play/Pause or the frame slider to control the animation. "
    "The chart animation is client-side; actual algorithm execution times are shown above."
)

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
st.dataframe(comparison_df.style.set_properties(subset=["Absolute Error"], **{"background-color": "rgba(245, 158, 11, 0.18)", "font-weight": "bold"}), use_container_width=True, hide_index=True)

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
with st.container(key="hlerrorchart"):
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

