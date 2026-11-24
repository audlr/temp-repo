# 시연용 시나리오 4개(원 넓이·동시 부도·간섭 무늬·별빛)를 위해 추가
import cmath
import math
from statistics import NormalDist

import numpy as np

_ND = NormalDist()
_erf = np.vectorize(math.erf)


def _phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


PEAK_CENTER = (0.40, 0.40)
PEAK_SIGMA = 0.03


def peak(points):
    dx = points[:, 0] - PEAK_CENTER[0]
    dy = points[:, 1] - PEAK_CENTER[1]
    return np.exp(-(dx * dx + dy * dy) / (2.0 * PEAK_SIGMA**2))


def _gauss_1d(mu, s=PEAK_SIGMA):
    k = math.sqrt(2.0) * s
    return s * math.sqrt(math.pi / 2.0) * (math.erf((1 - mu) / k) - math.erf((0 - mu) / k))


def peak_exact():
    return _gauss_1d(PEAK_CENTER[0]) * _gauss_1d(PEAK_CENTER[1])


PD = 0.02
RHO = 0.30
_PD_Z = _ND.inv_cdf(PD)
_S = math.sqrt(1.0 - RHO**2)


def joint_default(points):
    u1, u2 = points[:, 0], points[:, 1]
    out = np.zeros(len(points))
    first = u1 < PD
    if np.any(first):
        z1 = np.array([_ND.inv_cdf(min(max(u, 1e-15), 1 - 1e-15)) for u in u1[first]])
        limit = _phi((_PD_Z - RHO * z1) / _S)
        out[first] = (u2[first] < limit).astype(float)
    return out


def joint_default_exact(grid=200_001):
    z = np.linspace(_PD_Z - 12.0, _PD_Z, grid)
    y = np.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi) * _phi((_PD_Z - RHO * z) / _S)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(z)) / 2.0)


CIRCLE = (0.70, 0.30, 0.15)


def circle(points):
    cx, cy, r = CIRCLE
    d2 = (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2
    return (d2 <= r**2).astype(float)


def circle_exact():
    return math.pi * CIRCLE[2] ** 2


FRINGE_K = 100.0


def interference(points):
    return 1.0 + np.cos(FRINGE_K * (points[:, 0] + points[:, 1]))


def interference_exact():
    one = (cmath.exp(1j * FRINGE_K) - 1.0) / (1j * FRINGE_K)
    return 1.0 + (one * one).real


SCENARIOS = {
    "circle": {
        "label": "Circle area (Monte Carlo basics)",
        "func": circle,
        "exact": circle_exact,
        "binary": True,
        "overlay": CIRCLE,
        "description": (
            "Area of the circle centred at (0.70, 0.30) with radius 0.15. "
            "f = 1 inside the circle and 0 outside, so the integral is πr²."
        ),
    },
    "joint_default": {
        "label": "Joint default of two firms (MISER strength)",
        "func": joint_default,
        "exact": joint_default_exact,
        "binary": True,
        "overlay": None,
        "description": (
            f"Probability that two firms both default (each PD = {PD:.0%}, "
            f"asset correlation = {RHO}, Gaussian copula). f = 1 if both default. "
            "The non-zero region is a small corner near (0, 0)."
        ),
    },
    "interference": {
        "label": "Interference fringes (limit: spread-out variation)",
        "func": interference,
        "exact": interference_exact,
        "binary": False,
        "overlay": None,
        "description": (
            "Average brightness of two-beam interference fringes, "
            f"f = 1 + cos({FRINGE_K:g}·(x + y)). The fringes cover the whole square, "
            "so every split has about the same variance."
        ),
    },
    "peak": {
        "label": "Starlight on a sensor (risk: pilot can miss a narrow peak)",
        "func": peak,
        "exact": peak_exact,
        "binary": False,
        "overlay": (PEAK_CENTER[0], PEAK_CENTER[1], 2 * PEAK_SIGMA),
        "description": (
            "Total light of a star imaged on a camera sensor. The star's image "
            "(point spread function) is modelled as a narrow 2-D Gaussian at "
            f"({PEAK_CENTER[0]:.2f}, {PEAK_CENTER[1]:.2f}), σ = {PEAK_SIGMA}. Same shape "
            "as a test integrand of the original MISER paper (Press & Farrar, 1990)."
        ),
    },
}


# 시나리오별 시연용 Seed·샘플 수 (main = Gausian-Peak, risk = monte-carlo-miser)
DEMO = {
    "circle": {"main": {"seed": 114, "n": 10_000}, "risk": {"seed": 146, "n": 10_000}},
    "joint_default": {"main": {"seed": 82, "n": 10_000}, "risk": {"seed": 76, "n": 10_000}},
    "interference": {"main": {"seed": 66, "n": 10_000}, "risk": {"seed": 6, "n": 10_000}},
    "peak": {"main": {"seed": 40, "n": 10_000}, "risk": {"seed": 7, "n": 10_000}},
}
