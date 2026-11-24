"""
Live circle-sampling animation component.

Python 쪽은 *실제 알고리즘 실행 결과*를 실행 순서 그대로 직렬화해서
브라우저 컴포넌트(visualization/live_component/index.html)에 넘긴다.
재생은 브라우저에서만 일어나므로 Streamlit 스크립트는 블로킹되지 않는다.
재생 제어 버튼은 없으며, 사이드바의 Run Comparison 이 새 run_id 를 만들면
컴포넌트가 처음부터 자동 재생한다. 재생이 끝나면 컴포넌트가 run_id 를
값으로 돌려주고, app.py 는 그 신호로 최종 통계를 표시한다.

- 샘플 순서: run_*()가 반환한 points 배열 순서 = 실제 함수 평가 순서.
- 원 내부 판정: run_*()가 반환한 values (= circle_indicator, 0/1) 를 그대로 사용.
- MISER 분할선: splits[i]["after_eval"] 시점(해당 분할을 결정한 파일럿
  샘플을 모두 평가한 직후)에 표시.
- MISER 리프 영역: nodes[...]["eval_start"] 부터 count 개의 본 샘플.
"""

import pathlib
import shutil

import numpy as np
import plotly
import streamlit.components.v1 as components

_COMPONENT_DIR = pathlib.Path(__file__).parent / "live_component"

# Default playback: the whole recorded run is replayed in this many seconds.
PLAYBACK_SECONDS = 15


def _ensure_plotly_js():
    """Serve the plotly.js bundled with the installed plotly package (offline)."""
    src = pathlib.Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    dst = _COMPONENT_DIR / "plotly.min.js"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)


_ensure_plotly_js()
_live_component = components.declare_component("live_sampling",
                                                path=str(_COMPONENT_DIR))


def _r(a, digits=6):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def _empty_algo():
    return {"x": [], "y": [], "v": [], "elapsed": 0.0, "estimate": 0.0}


# 원이 없는·연속값 시나리오도 보내기 위해 추가
def _overlay(circle):
    return list(circle) if circle is not None else None


def _values(res, binary):
    return (res["values"].astype(int).tolist() if binary
            else _r(res["values"], 5))


def blank_payload(circle, reference):
    """Idle state before the first run: only the square and the circle."""
    ms = _empty_algo()
    ms.update(kind=[], leaves=[], splits=[], nodes=[])
    return {"run_id": 0, "n": 0, "circle": _overlay(circle), "binary": True,
            "vmax": 1.0, "heat": [], "reference": reference,
            "seconds": PLAYBACK_SECONDS, "mc": _empty_algo(), "ms": ms}


def build_payload(run_id, integrand, mc, ms, reference):
    """Serialize the real circle-indicator runs for the browser animation."""
    n = int(mc["evaluations"])
    assert n == ms["evaluations"]
    binary = bool(integrand.area_mode)
    vmax = float(max(mc["values"].max(), ms["values"].max(), 1e-12))

    leaves = []
    for nd in sorted((nd for nd in ms["nodes"] if nd["leaf"]),
                     key=lambda nd: nd["eval_start"]):
        size = float(np.prod([hi - lo for lo, hi in nd["box"]]))
        (x0, x1), (y0, y1) = nd["box"]
        leaves.append([nd["eval_start"], nd["count"], size, x0, x1, y0, y1,
                       nd["depth"], nd["estimate"]])

    splits = []
    for ev in ms["splits"]:
        (bx0, bx1), (by0, by1) = ev["box"]
        if ev["axis"] == 0:
            seg = [ev["midpoint"], by0, ev["midpoint"], by1]
        else:
            seg = [bx0, ev["midpoint"], bx1, ev["midpoint"]]
        # 분할할 때 % 비율을 보여주기 위해 추가
        share_left = ev["n_left"] / max(ev["n_left"] + ev["n_right"], 1)
        splits.append([ev["after_eval"]] + seg + [ev["depth"], ev["axis"],
                                                  bx0, bx1, by0, by1, share_left])

    nodes = sorted([nd["eval_at_create"], nd["depth"]] for nd in ms["nodes"])

    # 배경 히트맵을 그리기 위해 추가
    heat = []
    if ms.get("dim") == 2:
        g = np.linspace(0.0, 1.0, 100)
        gx, gy = np.meshgrid(g, g)
        h = integrand.func(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        heat = _r(h / max(float(np.max(np.abs(h))), 1e-12), 3)

    return {
        "run_id": run_id,
        "n": n,
        "circle": _overlay(integrand.circle),
        "binary": binary,
        "vmax": vmax,
        "heat": heat,
        "reference": reference,
        "seconds": PLAYBACK_SECONDS,
        "mc": {
            "x": _r(mc["points"][:, 0]), "y": _r(mc["points"][:, 1]),
            "v": _values(mc, binary),
            "elapsed": mc["elapsed"], "estimate": mc["estimate"],
        },
        "ms": {
            "x": _r(ms["points"][:, 0]), "y": _r(ms["points"][:, 1]),
            "v": _values(ms, binary),
            "kind": ms["kinds"].astype(int).tolist(),
            "leaves": leaves, "splits": splits, "nodes": nodes,
            "elapsed": ms["elapsed"], "estimate": ms["estimate"],
        },
    }


def live_sampling(payload, key="live-sampling"):
    """Render the player. Returns the run_id once its animation has finished."""
    return _live_component(payload=payload, key=key, default=None)
