"""
Plotly figure builders.

모든 그래프는 알고리즘이 실제로 반환한 데이터(points, kinds, nodes, splits,
estimate, elapsed)만 사용한다. 화면 성능을 위해 산점도에는 실제 샘플 중 일부를
표시용으로만 추출(subsample)하며, 적분 계산에는 영향을 주지 않는다.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MC_COLOR = "#1f77b4"      # blue
MISER_COLOR = "#ff7f0e"   # orange
MISER_PILOT_COLOR = "#8c564b"
REF_COLOR = "#555555"     # gray
CURVE_COLOR = "#222222"

ALGO_COLORS = {"Standard Monte Carlo": MC_COLOR, "MISER": MISER_COLOR}
TEMPLATE = "plotly_white"


def display_subset(n_total, max_points, seed=0):
    """Indices of real samples to draw. Display-only; never used for estimates."""
    if n_total <= max_points:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=max_points, replace=False))


def leaf_boxes(result):
    return [nd["box"] for nd in result.get("nodes", []) if nd["leaf"]]


# ---------------------------------------------------------------- sampling 1D

def histogram_counts(result, bounds, bins):
    x = result["points"][:, 0]
    counts, edges = np.histogram(x, bins=bins, range=tuple(bounds[0]))
    return counts, edges


def sampling_figure_1d(integrand, bounds, result, algo, max_points, y_range,
                       hist_max, bins=50):
    """Top: f(x) with real sample positions. Bottom: sample-density histogram."""
    color = ALGO_COLORS[algo]
    a, b = bounds[0]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
    )

    xs = np.linspace(a, b, 400)
    fig.add_trace(go.Scatter(
        x=xs, y=integrand.scalar(xs), mode="lines", name="f(x)",
        line=dict(color=CURVE_COLOR, width=2),
    ), row=1, col=1)

    pts = result["points"][:, 0]
    vals = result["values"]
    idx = display_subset(len(pts), max_points)
    kinds = result.get("kinds")

    if kinds is None:
        fig.add_trace(go.Scattergl(
            x=pts[idx], y=vals[idx], mode="markers", name="Samples",
            marker=dict(size=4, color=color, opacity=0.45),
        ), row=1, col=1)
    else:
        k = kinds[idx]
        for kind, name, c in ((1, "Production samples", color),
                              (0, "Pilot samples", MISER_PILOT_COLOR)):
            sel = idx[k == kind]
            fig.add_trace(go.Scattergl(
                x=pts[sel], y=vals[sel], mode="markers", name=name,
                marker=dict(size=4 if kind else 6, color=c,
                            opacity=0.45 if kind else 0.8,
                            symbol="circle" if kind else "x"),
            ), row=1, col=1)

    counts, edges = histogram_counts(result, bounds, bins)
    fig.add_trace(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
        marker=dict(color=color, line=dict(width=0)), opacity=0.8,
        name="Samples per bin (all)",
    ), row=2, col=1)

    if kinds is not None:
        # One trace per row (add_vline per boundary is very slow in Plotly).
        cuts = sorted({xv for box in leaf_boxes(result) for xv in box[0]
                       if a < xv < b})
        for row, (y0, y1) in ((1, y_range), (2, (0, hist_max * 1.08))):
            lx, ly = [], []
            for xv in cuts:
                lx += [xv, xv, None]
                ly += [y0, y1, None]
            fig.add_trace(go.Scatter(
                x=lx, y=ly, mode="lines", hoverinfo="skip",
                name="MISER partition boundaries", showlegend=(row == 1),
                line=dict(color=MISER_COLOR, width=1, dash="dot"),
            ), row=row, col=1)

    fig.update_yaxes(range=y_range, title_text="f(x)", row=1, col=1)
    fig.update_yaxes(range=[0, hist_max * 1.08], title_text="Samples / bin",
                     row=2, col=1)
    fig.update_xaxes(range=[a, b], row=1, col=1)
    fig.update_xaxes(range=[a, b], title_text="x", row=2, col=1)
    fig.update_layout(
        template=TEMPLATE, height=560, margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.08, x=0),
        bargap=0,
    )
    return fig


# ---------------------------------------------------------------- sampling 2D

def _circle(circle):
    cx, cy, r = circle
    theta = np.linspace(0, 2 * np.pi, 180)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _split_lines(splits):
    lx, ly = [], []
    for ev in splits:
        box = ev["box"]
        if ev["axis"] == 0:
            lx += [ev["midpoint"], ev["midpoint"], None]
            ly += [box[1][0], box[1][1], None]
        else:
            lx += [box[0][0], box[0][1], None]
            ly += [ev["midpoint"], ev["midpoint"], None]
    return lx, ly


def sampling_figure_2d(result, algo, max_points, circle, area_mode=False):
    color = ALGO_COLORS[algo]
    pts = result["points"]
    idx = display_subset(len(pts), max_points)
    kinds = result.get("kinds")
    fig = go.Figure()

    if area_mode:
        # Indicator integrand: colour each real sample by inside / outside.
        inside = result["values"][idx] > 0.5
        sel_kind = kinds[idx] if kinds is not None else np.ones(len(idx), int)
        for mask, name, c in ((inside & (sel_kind == 1), "Inside circle", color),
                              (~inside & (sel_kind == 1), "Outside circle", "#b0b0b0")):
            fig.add_trace(go.Scattergl(
                x=pts[idx[mask], 0], y=pts[idx[mask], 1], mode="markers",
                name=name, marker=dict(size=4, color=c, opacity=0.55)))
        if kinds is not None:
            sel = idx[sel_kind == 0]
            fig.add_trace(go.Scattergl(
                x=pts[sel, 0], y=pts[sel, 1], mode="markers", name="Pilot samples",
                marker=dict(size=6, color=MISER_PILOT_COLOR, opacity=0.6,
                            symbol="x")))
    elif kinds is None:
        fig.add_trace(go.Scattergl(
            x=pts[idx, 0], y=pts[idx, 1], mode="markers", name="Samples",
            marker=dict(size=4, color=color, opacity=0.5),
        ))
    if kinds is not None and not area_mode:
        k = kinds[idx]
        for kind, name, c in ((1, "Production samples", color),
                              (0, "Pilot samples", MISER_PILOT_COLOR)):
            sel = idx[k == kind]
            fig.add_trace(go.Scattergl(
                x=pts[sel, 0], y=pts[sel, 1], mode="markers", name=name,
                marker=dict(size=4 if kind else 6, color=c,
                            opacity=0.5 if kind else 0.85,
                            symbol="circle" if kind else "x"),
            ))
    if kinds is not None:
        lx, ly = _split_lines(result["splits"])
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="lines", name="Split boundaries",
            line=dict(color="#a0522d", width=1.4), hoverinfo="skip",
        ))

    cx, cy = _circle(circle)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                             name="Circle" if area_mode else "Village boundary",
                             line=dict(color=CURVE_COLOR, width=2),
                             hoverinfo="skip"))
    fig.update_xaxes(range=[0, 1], title_text="x", constrain="domain")
    fig.update_yaxes(range=[0, 1], title_text="y", scaleanchor="x",
                     constrain="domain")
    fig.update_layout(template=TEMPLATE, height=560,
                      margin=dict(l=50, r=20, t=40, b=40),
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


# ---------------------------------------------------------------- partitioning

def _hover(nd):
    pv = "—" if nd["pilot_var"] is None else f"{nd['pilot_var']:.3e}"
    status = "leaf (stopped splitting)" if nd["leaf"] else "split further"
    return (f"Region #{nd['id']} · depth {nd['depth']}<br>"
            f"Allocated evaluations: {nd['count']:,}<br>"
            f"Pilot variance estimate: {pv}<br>Status: {status}")


def partition_figure_1d(integrand, bounds, regions, max_density, f_range=None):
    """Bars = regions at the selected depth; height = evaluations per unit length."""
    a, b = bounds[0]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    lo = np.array([r["box"][0][0] for r in regions])
    hi = np.array([r["box"][0][1] for r in regions])
    counts = np.array([r["count"] for r in regions])
    dens = counts / (hi - lo)
    fig.add_trace(go.Bar(
        x=(lo + hi) / 2, y=dens, width=hi - lo,
        marker=dict(color=counts, colorscale="Oranges", cmin=0,
                    cmax=max(counts.max(), 1),
                    line=dict(color="#7f3f00", width=1.2),
                    colorbar=dict(title="Allocated<br>evaluations", x=1.08)),
        hovertext=[_hover(r) for r in regions], hoverinfo="text",
        text=[f"{c:,}" for c in counts] if len(regions) <= 16 else None,
        textposition="inside", name="Region",
    ), secondary_y=False)
    xs = np.linspace(a, b, 400)
    fig.add_trace(go.Scatter(x=xs, y=integrand.scalar(xs), mode="lines",
                             name="f(x)", line=dict(color=CURVE_COLOR, width=2)),
                  secondary_y=True)
    fig.update_xaxes(range=[a, b], title_text="x")
    fig.update_yaxes(title_text="Evaluations per unit length",
                     range=[0, max_density * 1.08], secondary_y=False)
    fig.update_yaxes(title_text="f(x)", secondary_y=True, showgrid=False,
                     range=f_range)
    fig.update_layout(template=TEMPLATE, height=460, bargap=0,
                      margin=dict(l=50, r=20, t=30, b=40),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def partition_figure_2d(regions, max_density, circle):
    fig = go.Figure()
    for r in regions:
        (x0, x1), (y0, y1) = r["box"]
        d = r["count"] / ((x1 - x0) * (y1 - y0))
        t = min(d / max_density, 1.0) if max_density > 0 else 0.0
        # Orange shade proportional to evaluation density.
        fill = f"rgba(255,127,14,{0.08 + 0.82 * t:.3f})"
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode="lines",
            fill="toself", fillcolor=fill,
            line=dict(color="#7f3f00", width=1),
            hovertext=_hover(r) + f"<br>Evaluation density: {d:,.0f} / unit area",
            hoverinfo="text", hoveron="fills", showlegend=False,
        ))
    if circle is not None:  # 원이 없는 시나리오도 그리기 위해 수정
        cx, cy = _circle(circle)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="Circle boundary",
                                 line=dict(color=CURVE_COLOR, width=2, dash="dash"),
                                 hoverinfo="skip"))
    fig.update_xaxes(range=[0, 1], title_text="x", constrain="domain")
    fig.update_yaxes(range=[0, 1], title_text="y", scaleanchor="x",
                     constrain="domain")
    fig.update_layout(template=TEMPLATE, height=520,
                      margin=dict(l=50, r=20, t=30, b=40))
    return fig


def depth_summary_figure(nodes):
    """Number of regions and leaves created at each recursion depth."""
    depths = np.array([nd["depth"] for nd in nodes])
    leaves = np.array([nd["leaf"] for nd in nodes])
    ds = np.arange(depths.max() + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ds, y=[np.sum((depths == d) & ~leaves) for d in ds],
                         name="Split further", marker_color=MISER_COLOR))
    fig.add_trace(go.Bar(x=ds, y=[np.sum((depths == d) & leaves) for d in ds],
                         name="Leaf (production sampling)",
                         marker_color="#ffbb78"))
    fig.update_layout(template=TEMPLATE, barmode="stack", height=300,
                      title="Regions created at each depth",
                      xaxis_title="Recursion depth", yaxis_title="Regions",
                      margin=dict(l=50, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=-0.3))
    return fig


# ---------------------------------------------------------------- analysis

def _lines(rows, ykey, title, ytitle, log_y=False, reference=None):
    fig = go.Figure()
    for algo, color in ALGO_COLORS.items():
        rr = [r for r in rows if r["algorithm"] == algo]
        fig.add_trace(go.Scatter(
            x=[r["n"] for r in rr], y=[r[ykey] for r in rr],
            mode="lines+markers", name=algo, line=dict(color=color, width=2.5),
            marker=dict(size=8),
        ))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dash"),
                      annotation_text=f"Reference = {reference:.6g}",
                      annotation_position="top left")
    fig.update_xaxes(type="log", title_text="Number of samples")
    fig.update_yaxes(title_text=ytitle, type="log" if log_y else "linear")
    fig.update_layout(template=TEMPLATE, title=title, height=430,
                      margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def convergence_figure(rows, reference):
    return _lines(rows, "estimate", "Estimated integral vs number of samples",
                  "Estimated integral", reference=reference)


def error_figure(rows):
    rows = [dict(r, abs_error=max(r["abs_error"], 1e-16)) for r in rows]
    return _lines(rows, "abs_error", "Absolute error vs number of samples "
                  "(single run per size)", "Absolute error (log)", log_y=True)


def repeated_figure(rows, key, title, ytitle):
    rows = [dict(r, **{key: max(r[key], 1e-16)}) for r in rows]
    return _lines(rows, key, title, ytitle, log_y=True)


def time_figure(rows, key="time", title="Execution time vs number of samples"):
    fig = go.Figure()
    for algo, color in ALGO_COLORS.items():
        rr = [r for r in rows if r["algorithm"] == algo]
        fig.add_trace(go.Bar(
            x=[f"{r['n']:,}" for r in rr], y=[r[key] * 1000 for r in rr],
            name=algo, marker_color=color,
        ))
    fig.update_layout(template=TEMPLATE, barmode="group", title=title,
                      height=430, xaxis_title="Number of samples",
                      yaxis_title="Execution time (ms)",
                      margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def accuracy_time_figure(rows, tkey, ekey, title, ytitle):
    fig = go.Figure()
    for algo, color in ALGO_COLORS.items():
        rr = [r for r in rows if r["algorithm"] == algo]
        fig.add_trace(go.Scatter(
            x=[r[tkey] * 1000 for r in rr],
            y=[max(r[ekey], 1e-16) for r in rr],
            mode="lines+markers+text", name=algo,
            text=[f"{r['n']//1000}k" for r in rr], textposition="top center",
            line=dict(color=color, width=2), marker=dict(size=9),
        ))
    fig.update_xaxes(type="log", title_text="Execution time (ms, log)")
    fig.update_yaxes(type="log", title_text=ytitle)
    fig.update_layout(template=TEMPLATE, title=title, height=430,
                      margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def final_comparison_figure(mc_abs, ms_abs, mc_time, ms_time):
    """Side-by-side bars: absolute error and execution time of the final run."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                        subplot_titles=("Absolute error", "Execution time (ms)"))
    names = ["Plain Monte Carlo", "MISER"]
    colors = [MC_COLOR, MISER_COLOR]
    fig.add_trace(go.Bar(x=names, y=[mc_abs, ms_abs], marker_color=colors,
                         text=[f"{mc_abs:.2e}", f"{ms_abs:.2e}"],
                         textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[mc_time * 1000, ms_time * 1000],
                         marker_color=colors,
                         text=[f"{mc_time * 1000:.2f}", f"{ms_time * 1000:.2f}"],
                         textposition="outside", showlegend=False), row=1, col=2)
    fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(range=[0, max(mc_abs, ms_abs) * 1.25 or 1], row=1, col=1)
    fig.update_yaxes(range=[0, max(mc_time, ms_time) * 1000 * 1.25 or 1], row=1, col=2)
    fig.update_layout(template=TEMPLATE, height=400,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig
