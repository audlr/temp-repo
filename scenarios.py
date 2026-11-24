"""
시연용 시나리오 4개 (두 저장소 공통 파일). 발표 순서: 원리 -> 강점 -> 한계 -> 위험성.

모두 단위 정사각형 [0,1]^2 위의 2D 적분이고, 정답(exact)이 있다.
func 는 (n, 2) 점 배열을 받아 n 개의 값을 돌려준다.

  circle        원 넓이                   MC 기본 원리 (원래 데모의 원)
  joint_default 두 회사 동시 부도 확률    MISER 강점 — 가우시안 코퓰라(Li 2000) 신용위험 모형
  interference  간섭 무늬 평균 밝기       반례1(한계) — 두 빛의 간섭 I = 1 + cos(위상차), 줄무늬가 전체에 퍼짐
  peak          별빛 총량                 반례2(위험성) — 별 이미지(2D 가우시안)의 총 밝기.
                                          MISER 원논문(Press & Farrar 1990)의 좁은 가우시안 테스트 함수와 같은 형태

의존성: numpy + 파이썬 표준 라이브러리(math, statistics) 만 사용.
"""

import cmath
import math
from statistics import NormalDist

import numpy as np

_ND = NormalDist()
_erf = np.vectorize(math.erf)


def _phi(x):
    """표준정규 누적분포 Φ(x) (벡터)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


# ------------------------------------------------------------ 1. 좁은 가우시안 봉우리
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


# ------------------------------------------------------------ 2. 두 회사 동시 부도
PD = 0.02     # 각 회사의 부도 확률
RHO = 0.30    # 자산 상관계수
_PD_Z = _ND.inv_cdf(PD)
_S = math.sqrt(1.0 - RHO**2)


def joint_default(points):
    """둘 다 부도면 1, 아니면 0.

    (u1, u2) 균등난수 -> z1 = Φ⁻¹(u1), z2 = ρ z1 + √(1-ρ²) Φ⁻¹(u2).
    회사1 부도  <=>  z1 < Φ⁻¹(PD)  <=>  u1 < PD
    회사2 부도  <=>  z2 < Φ⁻¹(PD)  <=>  u2 < Φ((Φ⁻¹(PD) - ρ z1) / √(1-ρ²))
    """
    u1, u2 = points[:, 0], points[:, 1]
    out = np.zeros(len(points))
    first = u1 < PD
    if np.any(first):
        z1 = np.array([_ND.inv_cdf(min(max(u, 1e-15), 1 - 1e-15)) for u in u1[first]])
        limit = _phi((_PD_Z - RHO * z1) / _S)
        out[first] = (u2[first] < limit).astype(float)
    return out


def joint_default_exact(grid=200_001):
    """P(Z1 < c, Z2 < c): z1 에 대한 1차원 적분(사다리꼴)."""
    z = np.linspace(_PD_Z - 12.0, _PD_Z, grid)
    y = np.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi) * _phi((_PD_Z - RHO * z) / _S)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(z)) / 2.0)


# ------------------------------------------------------------ 3. 원 넓이
CIRCLE = (0.70, 0.30, 0.15)   # 원래 데모의 원: 중심 (0.70, 0.30), 반지름 0.15


def circle(points):
    cx, cy, r = CIRCLE
    d2 = (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2
    return (d2 <= r**2).astype(float)


def circle_exact():
    return math.pi * CIRCLE[2] ** 2


# ------------------------------------------------------------ 4. 간섭 무늬
FRINGE_K = 100.0   # 대각선 방향 줄무늬 촘촘함 (정사각형 안에 약 16줄)


def interference(points):
    """세기가 같은 두 빛의 간섭: I = 1 + cos(위상차), 위상차 = k (x + y)."""
    return 1.0 + np.cos(FRINGE_K * (points[:, 0] + points[:, 1]))


def interference_exact():
    # ∫∫ cos(k(x+y)) = Re[((e^{ik} - 1) / (ik))^2]
    one = (cmath.exp(1j * FRINGE_K) - 1.0) / (1j * FRINGE_K)
    return 1.0 + (one * one).real


# ------------------------------------------------------------ 목록
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
        "overlay": (PEAK_CENTER[0], PEAK_CENTER[1], 2 * PEAK_SIGMA),  # 2σ 원
        "description": (
            "Total light of a star imaged on a camera sensor. The star's image "
            "(point spread function) is modelled as a narrow 2-D Gaussian at "
            f"({PEAK_CENTER[0]:.2f}, {PEAK_CENTER[1]:.2f}), σ = {PEAK_SIGMA}. Same shape "
            "as a test integrand of the original MISER paper (Press & Farrar, 1990)."
        ),
    },
}


# ------------------------------------------------------------ 시연 프리셋
# 시나리오를 고르면 앱이 아래 Seed·샘플 수로 바꿔 둔다.
# Seed 는 "운 좋은 극단값"이 아니라, 30번 반복한 평균 결과(RMSE 비율)에 가장 가까운
# 대표적인 실행을 골랐다(0~149 중 탐색). 예외: risk 앱의 peak 는 위험성을 보여주려고
# 30번 중 2번꼴로 나오는 실패 사례(Seed 7)를 일부러 골랐다.
#   main = Gausian-Peak 앱(파일럿 10%),  risk = monte-carlo-miser 앱(축별 파일럿 8~16개)
# focus 는 결과 화면에서 강조해서 보여줄 "볼 곳" 안내 문구다.
DEMO = {
    "circle": {
        "main": {"seed": 114, "n": 10_000,
                 "focus": "Basics: compare the Absolute Error. Plain MC estimates the area as "
                          "the share of points inside the circle; MISER puts more points on the "
                          "rows the circle crosses and is usually closer (about 3x over 30 runs)."},
        "risk": {"seed": 146, "n": 10_000,
                 "focus": "Basics: with only 8-16 pilot points per half, this MISER is no better "
                          "than plain MC here (about 0.8x over 30 runs). Compare the Absolute Error."},
    },
    "joint_default": {
        "main": {"seed": 82, "n": 10_000,
                 "focus": "Strength: look at the Absolute Error. Plain MC lands only a handful of "
                          "points in the tiny default corner and is off by ~20%; MISER sends its "
                          "samples there and is off by ~3% (about 6.5x over 30 runs)."},
        "risk": {"seed": 76, "n": 10_000,
                 "focus": "This MISER's few pilot points rarely hit the tiny corner, so it gains "
                          "nothing over plain MC (about 1.0x over 30 runs). Compare the Absolute Error."},
    },
    "interference": {
        "main": {"seed": 66, "n": 10_000,
                 "focus": "Limit: look at the Absolute Error - MISER is worse. Every split sees "
                          "about the same variance (~50/50), so it only spends samples on pilots "
                          "(about 0.67x over 30 runs)."},
        "risk": {"seed": 6, "n": 10_000,
                 "focus": "Limit: MISER is worse than plain MC (about 0.85x over 30 runs). "
                          "Every split sees about the same variance, so there is nothing to gain."},
    },
    "peak": {
        "main": {"seed": 40, "n": 10_000,
                 "focus": "Look at the Absolute Error: with enough pilot samples MISER finds the "
                          "narrow peak and is far more accurate (about 8.5x over 30 runs). Compare "
                          "with the risk app, where the same scenario fails."},
        "risk": {"seed": 7, "n": 10_000,
                 "focus": "Risk: look at the MISER Final estimate - about twice the exact value "
                          "(92% error). The pilot points missed the peak, so the half that holds it "
                          "got only the minimum samples and was never split. This happens in about "
                          "2 of 30 runs."},
    },
}
