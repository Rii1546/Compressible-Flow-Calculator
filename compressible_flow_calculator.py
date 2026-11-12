"""
compressible_flow_calculator.py  --  single-file edition
=========================================================

One-dimensional compressible-flow calculator for a calorically perfect gas,
combining in ONE file:

  PART 1  library      isentropic flow, normal shock, oblique shock,
                       Prandtl-Meyer expansion, Rayleigh flow, Fanno flow
  PART 2  command line validation, plots, one-shot summary, interactive menu
  PART 3  test-suite   116 pytest tests (library + command line)

Usage
-----
    python compressible_flow_calculator.py                # validate, plots, menu
    python compressible_flow_calculator.py --validate     # table self-check
    python compressible_flow_calculator.py --mach 2.5     # one-shot summary
    python compressible_flow_calculator.py --test         # run all 116 tests
    pytest compressible_flow_calculator.py -q             # same, via pytest
    import compressible_flow_calculator as cf             # use as a library

Requires: numpy, scipy; matplotlib for plots; pytest for the tests.

References
----------
[1] Anderson, J.D., Modern Compressible Flow, 3rd ed., McGraw-Hill, 2003.
[2] Ames Research Staff, NACA Report 1135, "Equations, Tables, and Charts for
    Compressible Flow", 1953.
[3] Liepmann, H.W. and Roshko, A., Elements of Gasdynamics, Wiley, 1957.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

try:                                   # pytest is only needed for PART 3
    import pytest
except ImportError:                    # pragma: no cover
    pytest = None

cf = sys.modules[__name__]             # the CLI code below calls cf.<function>


# ============================================================================
# PART 1 - LIBRARY (formerly compressible_flow.py)
# ============================================================================

DEFAULT_GAMMA = 1.4


# --------------------------------------------------------------------------- #
# Result containers (NamedTuples can still be unpacked like plain tuples)
# --------------------------------------------------------------------------- #
class IsentropicState(NamedTuple):
    T0_T: float
    p0_p: float
    rho0_rho: float


class NormalShockState(NamedTuple):
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p02_p01: float


class ObliqueShockState(NamedTuple):
    beta_deg: float
    theta_deg: float
    Mn1: float
    Mn2: float
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p02_p01: float


class ExpansionState(NamedTuple):
    M2: float
    nu1_deg: float
    nu2_deg: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    mu1_deg: float  # leading Mach wave angle
    mu2_deg: float  # trailing Mach wave angle


class RayleighState(NamedTuple):
    T_Tstar: float
    p_pstar: float
    rho_rhostar: float
    V_Vstar: float
    T0_T0star: float
    p0_p0star: float


class FannoState(NamedTuple):
    T_Tstar: float
    p_pstar: float
    rho_rhostar: float
    V_Vstar: float
    p0_p0star: float
    fLstar_D: float  # Darcy friction factor * L* / D  (= 4 * Fanning f * L*/D)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_gamma(gamma):
    g = np.asarray(gamma, dtype=float)
    if g.ndim != 0 or not np.isfinite(g) or g <= 1.0:
        raise ValueError(f"gamma must be a finite scalar > 1, got {gamma!r}.")
    return float(g)


def _check_mach(M, *, minimum=0.0, inclusive=False, name="Mach number"):
    """Return M as a float array; require M > minimum (or >= if inclusive)."""
    arr = np.asarray(M, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {M!r}.")
    bad = np.any(arr < minimum) if inclusive else np.any(arr <= minimum)
    if bad:
        rel = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {rel} {minimum}, got {M!r}.")
    return arr


def _scalar(x):
    """Collapse 0-d arrays to Python floats, leave real arrays alone."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


# --------------------------------------------------------------------------- #
# 1. Isentropic flow
# --------------------------------------------------------------------------- #
def isentropic_ratios(M, gamma=DEFAULT_GAMMA) -> IsentropicState:
    """Stagnation-to-static ratios T0/T, p0/p, rho0/rho at Mach M (M >= 0)."""
    g = _check_gamma(gamma)
    M = _check_mach(M, inclusive=True)
    factor = 1.0 + 0.5 * (g - 1.0) * M**2
    return IsentropicState(
        _scalar(factor),
        _scalar(factor ** (g / (g - 1.0))),
        _scalar(factor ** (1.0 / (g - 1.0))),
    )


def area_mach_ratio(M, gamma=DEFAULT_GAMMA):
    """Area ratio A/A* for isentropic quasi-1D flow (M > 0)."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    term = (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M**2)
    return _scalar(term ** ((g + 1.0) / (2.0 * (g - 1.0))) / M)


def mach_from_area_ratio(A_Astar, supersonic=True, gamma=DEFAULT_GAMMA):
    """Invert A/A* -> M. ``supersonic`` selects the branch (M>1 or M<1)."""
    g = _check_gamma(gamma)
    A = float(A_Astar)
    if not np.isfinite(A) or A < 1.0 - 1e-12:
        raise ValueError(f"A/A* must be >= 1, got {A_Astar!r}.")
    # NB: A/A* - 1 ~ (M-1)^2 near the sonic point, so the tolerance must be a pure
    # absolute one (np.isclose's default rtol would corrupt M by ~1e-3 here).
    if abs(A - 1.0) <= 1e-12:
        return 1.0
    f = lambda m: area_mach_ratio(m, g) - A
    if supersonic:
        hi = 2.0
        while f(hi) < 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14)
    return brentq(f, 1e-9, 1.0, xtol=1e-14)


# --------------------------------------------------------------------------- #
# 2. Normal shock
# --------------------------------------------------------------------------- #
def normal_shock(M1, gamma=DEFAULT_GAMMA) -> NormalShockState:
    """Property ratios across a stationary normal shock (M1 >= 1)."""
    g = _check_gamma(gamma)
    M1 = _check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1")
    M1sq = M1**2
    M2 = np.sqrt((1 + 0.5 * (g - 1) * M1sq) / (g * M1sq - 0.5 * (g - 1)))
    p2_p1 = 1 + (2 * g / (g + 1)) * (M1sq - 1)
    rho2_rho1 = ((g + 1) * M1sq) / ((g - 1) * M1sq + 2)
    T2_T1 = p2_p1 / rho2_rho1
    p02_p01 = (rho2_rho1 ** (g / (g - 1))) * (
        ((g + 1) / (2 * g * M1sq - (g - 1))) ** (1 / (g - 1))
    )
    return NormalShockState(*(_scalar(v) for v in (M2, p2_p1, rho2_rho1, T2_T1, p02_p01)))


def shock_entropy_rise(M1, gamma=DEFAULT_GAMMA):
    """Dimensionless entropy rise (s2 - s1)/R = -ln(p02/p01) across a normal shock."""
    return _scalar(-np.log(normal_shock(M1, gamma).p02_p01))


def pitot_pressure_ratio(M1, gamma=DEFAULT_GAMMA):
    """Rayleigh Pitot formula: p02/p1 for a Pitot probe in supersonic flow (M1 >= 1)."""
    g = _check_gamma(gamma)
    M1 = _check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1")
    M1sq = M1**2
    a = (0.5 * (g + 1) * M1sq) ** (g / (g - 1))
    b = (2 * g * M1sq / (g + 1) - (g - 1) / (g + 1)) ** (1 / (g - 1))
    return _scalar(a / b)


def mach1_from_pressure_ratio(p2_p1, gamma=DEFAULT_GAMMA):
    """Upstream Mach number that produces a given static-pressure jump p2/p1 >= 1."""
    g = _check_gamma(gamma)
    r = float(p2_p1)
    if not np.isfinite(r) or r < 1.0:
        raise ValueError(f"p2/p1 across a shock must be >= 1, got {p2_p1!r}.")
    return float(np.sqrt(1.0 + (g + 1.0) / (2.0 * g) * (r - 1.0)))


def mach1_from_downstream_mach(M2, gamma=DEFAULT_GAMMA):
    """Upstream Mach number for a given post-shock Mach number M2."""
    g = _check_gamma(gamma)
    M2 = float(M2)
    lo = np.sqrt((g - 1.0) / (2.0 * g))
    if not (lo <= M2 <= 1.0):
        raise ValueError(
            f"Post-shock Mach number must lie in [{lo:.4f}, 1] for gamma={g}, got {M2!r}."
        )
    return float(np.sqrt((1 + 0.5 * (g - 1) * M2**2) / (g * M2**2 - 0.5 * (g - 1))))


# --------------------------------------------------------------------------- #
# 3. Oblique shock
# --------------------------------------------------------------------------- #
def mach_angle_deg(M):
    """Mach angle mu = arcsin(1/M) in degrees (M >= 1)."""
    M = _check_mach(M, minimum=1.0, inclusive=True)
    return _scalar(np.degrees(np.arcsin(1.0 / M)))


def theta_from_beta(M1, beta_deg, gamma=DEFAULT_GAMMA):
    """Flow-deflection angle theta (deg) for shock-wave angle beta (deg): the theta-beta-M relation."""
    g = _check_gamma(gamma)
    M1 = _check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1")
    beta = np.radians(np.asarray(beta_deg, dtype=float))
    tan_theta = (
        2.0
        / np.tan(beta)
        * (M1**2 * np.sin(beta) ** 2 - 1.0)
        / (M1**2 * (g + np.cos(2.0 * beta)) + 2.0)
    )
    return _scalar(np.degrees(np.arctan(tan_theta)))


def max_deflection_angle(M1, gamma=DEFAULT_GAMMA):
    """Return (theta_max_deg, beta_at_theta_max_deg) for an attached oblique shock."""
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1, minimum=1.0, name="Upstream Mach number M1 (must be > 1)"))
    M2sq = M1**2
    sin2b = (1.0 / (g * M2sq)) * (
        0.25 * (g + 1) * M2sq
        - 1.0
        + np.sqrt((g + 1) * (1 + 0.5 * (g - 1) * M2sq + (g + 1) / 16.0 * M2sq**2))
    )
    beta_max = np.arcsin(np.sqrt(sin2b))
    return float(theta_from_beta(M1, np.degrees(beta_max), g)), float(np.degrees(beta_max))


def oblique_shock(M1, theta_deg=None, beta_deg=None, weak=True, gamma=DEFAULT_GAMMA) -> ObliqueShockState:
    """
    Solve an oblique shock given M1 and EITHER the deflection angle ``theta_deg``
    or the wave angle ``beta_deg`` (exactly one must be supplied).

    ``weak`` chooses the weak (default) or strong solution when theta is given.
    Raises ValueError if theta exceeds the detachment limit theta_max(M1).
    """
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1, minimum=1.0, name="Upstream Mach number M1 (must be > 1)"))
    if (theta_deg is None) == (beta_deg is None):
        raise ValueError("Provide exactly one of theta_deg or beta_deg.")

    mu = float(mach_angle_deg(M1))
    if beta_deg is None:
        theta = float(theta_deg)
        if theta < 0.0:
            raise ValueError("theta must be >= 0 (mirror the geometry for expansions).")
        theta_max, beta_tmax = max_deflection_angle(M1, g)
        if theta > theta_max + 1e-12:
            raise ValueError(
                f"theta = {theta:.3f} deg exceeds theta_max = {theta_max:.3f} deg at "
                f"M1 = {M1:g}: the shock detaches (no attached oblique solution)."
            )
        if theta == 0.0:
            beta = mu if weak else 90.0
        else:
            f = lambda b: float(theta_from_beta(M1, b, g)) - theta
            if weak:
                beta = brentq(f, mu, beta_tmax, xtol=1e-13)
            else:
                beta = brentq(f, beta_tmax, 90.0, xtol=1e-13)
    else:
        beta = float(beta_deg)
        if not (mu - 1e-12 <= beta <= 90.0 + 1e-12):
            raise ValueError(
                f"beta must lie between the Mach angle ({mu:.3f} deg) and 90 deg, got {beta:g}."
            )
        theta = float(theta_from_beta(M1, beta, g))

    b = np.radians(beta)
    Mn1 = max(M1 * np.sin(b), 1.0)
    ns = normal_shock(Mn1, g)
    M2 = ns.M2 / np.sin(b - np.radians(theta))
    return ObliqueShockState(
        float(beta), float(theta), float(Mn1), float(ns.M2), float(M2),
        float(ns.p2_p1), float(ns.rho2_rho1), float(ns.T2_T1), float(ns.p02_p01),
    )


# --------------------------------------------------------------------------- #
# 4. Prandtl-Meyer expansion
# --------------------------------------------------------------------------- #
def prandtl_meyer(M, gamma=DEFAULT_GAMMA):
    """Prandtl-Meyer function nu(M) in degrees (M >= 1)."""
    g = _check_gamma(gamma)
    M = _check_mach(M, minimum=1.0, inclusive=True)
    a = np.sqrt((g + 1) / (g - 1))
    s = np.sqrt(M**2 - 1)
    nu = a * np.arctan(np.sqrt((g - 1) / (g + 1)) * s) - np.arctan(s)
    return _scalar(np.degrees(nu))


def prandtl_meyer_max(gamma=DEFAULT_GAMMA):
    """Limiting turn angle nu_max = (pi/2)(sqrt((g+1)/(g-1)) - 1), in degrees."""
    g = _check_gamma(gamma)
    return float(90.0 * (np.sqrt((g + 1) / (g - 1)) - 1.0))


def mach_from_pm_angle(nu_target_deg, gamma=DEFAULT_GAMMA):
    """Invert the Prandtl-Meyer function: Mach number whose nu(M) equals the target."""
    g = _check_gamma(gamma)
    nu = float(nu_target_deg)
    nu_max = prandtl_meyer_max(g)
    if not np.isfinite(nu) or nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu_target_deg!r}.")
    if nu >= nu_max:
        raise ValueError(
            f"nu = {nu:g} deg is not reachable: the limit for gamma={g:g} is {nu_max:.4f} deg."
        )
    if nu == 0.0:
        return 1.0
    f = lambda m: float(prandtl_meyer(m, g)) - nu
    hi = 2.0
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("Could not bracket the Mach number (nu too close to the limit).")
    return brentq(f, 1.0, hi, xtol=1e-13)


def prandtl_meyer_expansion(M1, turn_deg, gamma=DEFAULT_GAMMA) -> ExpansionState:
    """Isentropic expansion of supersonic flow M1 turned through ``turn_deg`` degrees."""
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1"))
    turn = float(turn_deg)
    if turn < 0.0:
        raise ValueError("Expansion turn angle must be >= 0.")
    nu1 = float(prandtl_meyer(M1, g))
    nu2 = nu1 + turn
    M2 = mach_from_pm_angle(nu2, g)
    t1, p1, r1 = isentropic_ratios(M1, g)
    t2, p2, r2 = isentropic_ratios(M2, g)
    return ExpansionState(
        M2, nu1, nu2,
        p1 / p2, r1 / r2, t1 / t2,  # static ratios: (p0/p)_1 / (p0/p)_2 etc.
        float(mach_angle_deg(M1)), float(mach_angle_deg(M2)),
    )


# --------------------------------------------------------------------------- #
# 5. Rayleigh flow (frictionless, constant-area, heat addition)
# --------------------------------------------------------------------------- #
def rayleigh_ratios(M, gamma=DEFAULT_GAMMA) -> RayleighState:
    """Rayleigh-flow ratios relative to the sonic (*) state, M > 0."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    d = 1.0 + g * M**2
    T_Ts = (g + 1) ** 2 * M**2 / d**2
    p_ps = (g + 1) / d
    rho_rs = d / ((g + 1) * M**2)
    T0_T0s = (g + 1) * M**2 * (2 + (g - 1) * M**2) / d**2
    p0_p0s = p_ps * ((2 + (g - 1) * M**2) / (g + 1)) ** (g / (g - 1))
    return RayleighState(*(_scalar(v) for v in (T_Ts, p_ps, rho_rs, 1.0 / rho_rs, T0_T0s, p0_p0s)))


def mach_from_rayleigh_T0_ratio(T0_T0star, supersonic, gamma=DEFAULT_GAMMA):
    """Invert T0/T0* -> M on the subsonic or supersonic branch."""
    g = _check_gamma(gamma)
    r = float(T0_T0star)
    if not (0.0 < r <= 1.0 + 1e-12):
        raise ValueError(f"T0/T0* must be in (0, 1], got {T0_T0star!r}.")
    if abs(r - 1.0) <= 1e-12:  # pure absolute tolerance: T0/T0* - 1 ~ (M-1)^2 near M = 1
        return 1.0
    f = lambda m: float(rayleigh_ratios(m, g).T0_T0star) - r
    if supersonic:
        floor = (g + 1) * (g - 1) / g**2  # limit as M -> infinity
        if r <= floor:
            raise ValueError(f"No supersonic solution: T0/T0* must exceed {floor:.4f}.")
        hi = 2.0
        while f(hi) > 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14)
    return brentq(f, 1e-9, 1.0, xtol=1e-14)


def rayleigh_exit(M1, T02_T01, gamma=DEFAULT_GAMMA):
    """
    Exit Mach number after heating (T02/T01 > 1) or cooling (< 1) a Rayleigh duct.
    The flow stays on its original branch; heating beyond the sonic limit
    raises ValueError (thermal choking).
    """
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1))
    ratio = float(T02_T01)
    if ratio <= 0.0:
        raise ValueError("T02/T01 must be positive.")
    target = float(rayleigh_ratios(M1, g).T0_T0star) * ratio
    if target > 1.0 + 1e-12:
        max_ratio = 1.0 / float(rayleigh_ratios(M1, g).T0_T0star)
        raise ValueError(
            f"Thermally choked: maximum T02/T01 for M1={M1:g} is {max_ratio:.4f}, got {ratio:g}."
        )
    return mach_from_rayleigh_T0_ratio(min(target, 1.0), supersonic=M1 > 1.0, gamma=g)


# --------------------------------------------------------------------------- #
# 6. Fanno flow (adiabatic, constant-area, wall friction)
# --------------------------------------------------------------------------- #
def fanno_parameter(M, gamma=DEFAULT_GAMMA):
    """Friction parameter f*L*/D (Darcy f, i.e. 4 x Fanning) needed to reach M = 1."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    M2 = M**2
    val = (1 - M2) / (g * M2) + (g + 1) / (2 * g) * np.log((g + 1) * M2 / (2 + (g - 1) * M2))
    return _scalar(val)


def fanno_ratios(M, gamma=DEFAULT_GAMMA) -> FannoState:
    """Fanno-flow ratios relative to the sonic (*) state, M > 0."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    e = 2 + (g - 1) * M**2
    T_Ts = (g + 1) / e
    p_ps = np.sqrt((g + 1) / e) / M
    rho_rs = np.sqrt(e / (g + 1)) / M
    p0_p0s = (e / (g + 1)) ** ((g + 1) / (2 * (g - 1))) / M
    return FannoState(
        *(_scalar(v) for v in (T_Ts, p_ps, rho_rs, 1.0 / rho_rs, p0_p0s)),
        fanno_parameter(M, g),
    )


def mach_from_fanno_parameter(fLstar_D, supersonic, gamma=DEFAULT_GAMMA):
    """Invert f*L*/D -> M on the subsonic or supersonic branch."""
    g = _check_gamma(gamma)
    p = float(fLstar_D)
    if p < 0.0 or not np.isfinite(p):
        raise ValueError(f"f L*/D must be >= 0, got {fLstar_D!r}.")
    if p == 0.0:
        return 1.0
    f = lambda m: float(fanno_parameter(m, g)) - p
    if supersonic:
        pmax = (g + 1) / (2 * g) * np.log((g + 1) / (g - 1)) - 1.0 / g
        if p >= pmax:
            raise ValueError(f"No supersonic solution: f L*/D must be below {pmax:.4f}.")
        hi = 2.0
        while f(hi) < 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14)
    return brentq(f, 1e-9, 1.0, xtol=1e-14)


def fanno_exit(M1, fL_D, gamma=DEFAULT_GAMMA):
    """
    Exit Mach number of a Fanno duct with friction parameter f*L/D (Darcy f).
    Raises ValueError if the duct is longer than L*, i.e. the flow chokes.
    """
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1))
    if fL_D < 0.0:
        raise ValueError("f L/D must be >= 0.")
    remaining = float(fanno_parameter(M1, g)) - float(fL_D)
    if remaining < -1e-12:
        raise ValueError(
            f"Choked: duct length f L/D = {fL_D:g} exceeds f L*/D = "
            f"{float(fanno_parameter(M1, g)):.4f} for M1 = {M1:g}."
        )
    return mach_from_fanno_parameter(max(remaining, 0.0), supersonic=M1 > 1.0, gamma=g)


# ============================================================================
# PART 2 - COMMAND-LINE INTERFACE (formerly main.py)
# ============================================================================

FIG_DIR = "figures"


# --------------------------------------------------------------------------- #
# Self-validation (asserts, so a regression fails loudly)
# --------------------------------------------------------------------------- #
def run_validation(verbose=True) -> bool:
    checks = []

    def check(label, computed, expected, tol):
        ok = abs(computed - expected) <= tol
        checks.append(ok)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} computed={computed:.4f}  expected={expected:.4f}")

    if verbose:
        print("=" * 72)
        print("VALIDATION AGAINST PUBLISHED TABLES (gamma = 1.4)")
        print("=" * 72)

    s = cf.isentropic_ratios(2.0)
    check("Isentropic M=2   T0/T", s.T0_T, 1.8000, 5e-5)
    check("Isentropic M=2   p0/p", s.p0_p, 7.8244, 5e-4)
    check("Isentropic M=2   A/A*", cf.area_mach_ratio(2.0), 1.6875, 5e-5)

    n = cf.normal_shock(2.0)
    check("Normal shock M1=2  M2", n.M2, 0.5774, 5e-5)
    check("Normal shock M1=2  p2/p1", n.p2_p1, 4.5000, 5e-5)
    check("Normal shock M1=2  T2/T1", n.T2_T1, 1.6875, 5e-5)
    check("Normal shock M1=2  p02/p01", n.p02_p01, 0.7209, 5e-5)

    o = cf.oblique_shock(2.0, theta_deg=10.0)
    check("Oblique M=2,theta=10  beta", o.beta_deg, 39.31, 0.01)
    check("Oblique M=2,theta=10  M2", o.M2, 1.641, 2e-3)
    check("Oblique M=2,theta=10  p2/p1", o.p2_p1, 1.707, 2e-3)

    check("Prandtl-Meyer M=2  nu", cf.prandtl_meyer(2.0), 26.380, 2e-3)
    check("Inverse PM nu=26.38 -> M", cf.mach_from_pm_angle(26.380), 2.0, 1e-3)

    r = cf.rayleigh_ratios(2.0)
    check("Rayleigh M=2  T0/T0*", r.T0_T0star, 0.7934, 5e-5)
    check("Rayleigh M=2  p/p*", r.p_pstar, 0.3636, 5e-5)

    f = cf.fanno_ratios(2.0)
    check("Fanno M=2  fL*/D", f.fLstar_D, 0.3050, 5e-5)
    check("Fanno M=2  p/p*", f.p_pstar, 0.4082, 5e-5)

    ok = all(checks)
    if verbose:
        print("-" * 72)
        print(f"  {sum(checks)}/{len(checks)} checks passed")
        print("=" * 72 + "\n")
    return ok


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _finish(fig, plt, name, show):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def make_plots(gamma=1.4, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = dict(linestyle="--", alpha=0.5)

    # 1. isentropic ratios + A/A*
    M = np.linspace(0.05, 5, 400)
    s = cf.isentropic_ratios(M, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M, s.T0_T, label=r"$T_0/T$")
    ax.plot(M, s.p0_p, label=r"$p_0/p$")
    ax.plot(M, s.rho0_rho, label=r"$\rho_0/\rho$")
    ax.plot(M, cf.area_mach_ratio(M, gamma), "k--", label=r"$A/A^*$")
    ax.set_yscale("log")
    ax.set(xlabel="Mach number", ylabel="Ratio (log scale)",
           title=f"Isentropic flow ratios vs. Mach number ($\\gamma$={gamma})")
    ax.legend(); ax.grid(True, which="both", **grid)
    _finish(fig, plt, "isentropic_curves.png", show)

    # 2. normal shock
    M1 = np.linspace(1, 5, 400)
    n = cf.normal_shock(M1, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M1, n.M2, label=r"$M_2$")
    ax.plot(M1, n.p2_p1, label=r"$p_2/p_1$")
    ax.plot(M1, n.T2_T1, label=r"$T_2/T_1$")
    ax.plot(M1, n.rho2_rho1, label=r"$\rho_2/\rho_1$")
    ax.plot(M1, n.p02_p01, label=r"$p_{02}/p_{01}$")
    ax.set(xlabel="Upstream Mach number $M_1$", ylabel="Ratio / downstream Mach number",
           title="Normal-shock relations vs. upstream Mach number")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "normal_shock_curves.png", show)

    # 3. Prandtl-Meyer
    M = np.linspace(1, 10, 400)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M, cf.prandtl_meyer(M, gamma), color="darkorange")
    ax.axhline(cf.prandtl_meyer_max(gamma), color="gray", ls=":",
               label=rf"$\nu_{{max}}$ = {cf.prandtl_meyer_max(gamma):.2f}$^\circ$")
    ax.set(xlabel="Mach number", ylabel=r"Prandtl-Meyer angle $\nu$ (deg)",
           title="Prandtl-Meyer function vs. Mach number")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "prandtl_meyer_curve.png", show)

    # 4. theta-beta-M chart
    fig, ax = plt.subplots(figsize=(8, 6))
    for Mach in (1.2, 1.5, 2, 3, 5, 10):
        mu = float(cf.mach_angle_deg(Mach))
        beta = np.linspace(mu, 90, 500)
        theta = cf.theta_from_beta(Mach, beta, gamma)
        line, = ax.plot(theta, beta, label=f"M$_1$={Mach}")
        tmax, bmax = cf.max_deflection_angle(Mach, gamma)
        ax.plot(tmax, bmax, "o", color=line.get_color(), ms=4)
    ax.set(xlabel=r"Deflection angle $\theta$ (deg)", ylabel=r"Wave angle $\beta$ (deg)",
           title=r"$\theta$-$\beta$-M chart (dots: $\theta_{max}$)", xlim=(0, 50), ylim=(0, 90))
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "theta_beta_M_chart.png", show)

    # 5. Rayleigh
    M = np.linspace(0.05, 5, 400)
    r = cf.rayleigh_ratios(M, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    for y, lab in ((r.T_Tstar, r"$T/T^*$"), (r.p_pstar, r"$p/p^*$"), (r.T0_T0star, r"$T_0/T_0^*$"),
                   (r.p0_p0star, r"$p_0/p_0^*$"), (r.V_Vstar, r"$V/V^*$")):
        ax.plot(M, y, label=lab)
    ax.set(xlabel="Mach number", ylabel="Ratio to sonic state", ylim=(0, 3),
           title="Rayleigh flow (heat addition, constant area)")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "rayleigh_curves.png", show)

    # 6. Fanno
    M = np.linspace(0.05, 5, 400)
    f = cf.fanno_ratios(M, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    for y, lab in ((f.T_Tstar, r"$T/T^*$"), (f.p_pstar, r"$p/p^*$"), (f.p0_p0star, r"$p_0/p_0^*$"),
                   (f.V_Vstar, r"$V/V^*$"), (f.fLstar_D, r"$fL^*/D$")):
        ax.plot(M, y, label=lab)
    ax.set(xlabel="Mach number", ylabel="Ratio to sonic state", ylim=(0, 4),
           title="Fanno flow (adiabatic, constant area, friction)")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "fanno_curves.png", show)

    print(f"Saved 6 figures to ./{FIG_DIR}/")


# --------------------------------------------------------------------------- #
# Printing helpers
# --------------------------------------------------------------------------- #
def print_summary(M, gamma=1.4):
    print(f"\n--- Isentropic relations (M = {M:g}, gamma = {gamma:g}) ---")
    s = cf.isentropic_ratios(M, gamma)
    print(f"  T0/T = {s.T0_T:.4f}   p0/p = {s.p0_p:.4f}   rho0/rho = {s.rho0_rho:.4f}")
    if M > 0:
        print(f"  A/A* = {cf.area_mach_ratio(M, gamma):.4f}")
    if M >= 1:
        n = cf.normal_shock(M, gamma)
        print(f"\n--- Normal shock (M1 = {M:g}) ---")
        print(f"  M2 = {n.M2:.4f}   p2/p1 = {n.p2_p1:.4f}   rho2/rho1 = {n.rho2_rho1:.4f}")
        print(f"  T2/T1 = {n.T2_T1:.4f}   p02/p01 = {n.p02_p01:.4f}   "
              f"(s2-s1)/R = {cf.shock_entropy_rise(M, gamma):.4f}")
        print(f"  Pitot p02/p1 = {cf.pitot_pressure_ratio(M, gamma):.4f}")
        print(f"\n--- Prandtl-Meyer (M = {M:g}) ---")
        print(f"  nu = {cf.prandtl_meyer(M, gamma):.4f} deg   (nu_max = {cf.prandtl_meyer_max(gamma):.2f} deg)")
        if M > 1:
            print(f"  Mach angle mu = {cf.mach_angle_deg(M):.3f} deg   "
                  f"theta_max (oblique) = {cf.max_deflection_angle(M, gamma)[0]:.3f} deg")
    else:
        print("\n(Shock and Prandtl-Meyer relations need M >= 1 - skipped.)")
    if M > 0:
        r = cf.rayleigh_ratios(M, gamma)
        f = cf.fanno_ratios(M, gamma)
        print(f"\n--- Rayleigh flow (M = {M:g}) ---")
        print(f"  T/T* = {r.T_Tstar:.4f}   p/p* = {r.p_pstar:.4f}   T0/T0* = {r.T0_T0star:.4f}   p0/p0* = {r.p0_p0star:.4f}")
        print(f"\n--- Fanno flow (M = {M:g}) ---")
        print(f"  T/T* = {f.T_Tstar:.4f}   p/p* = {f.p_pstar:.4f}   p0/p0* = {f.p0_p0star:.4f}   fL*/D = {f.fLstar_D:.4f}")
    print()


def _ask_float(prompt, default=None):
    raw = input(prompt).strip()
    if not raw and default is not None:
        return default
    return float(raw)


def interactive_menu(gamma=1.4):
    menu = (
        "\n" + "=" * 60 + "\nCOMPRESSIBLE FLOW CALCULATOR\n" + "=" * 60 + "\n"
        "  1  Full summary at one Mach number\n"
        "  2  Oblique shock (M1 + wedge angle theta)\n"
        "  3  Prandtl-Meyer expansion (M1 + turn angle)\n"
        "  4  Isentropic: Mach number from A/A*\n"
        "  5  Rayleigh duct: exit Mach for T02/T01\n"
        "  6  Fanno duct: exit Mach for f L/D\n"
        "  7  Change gamma (currently {g:g})\n"
        "  q  Quit\n"
    )
    while True:
        print(menu.format(g=gamma))
        try:
            choice = input("Choose an option: ").strip().lower()
            if choice in ("q", "quit", "exit", ""):
                break
            if choice == "1":
                print_summary(_ask_float("Mach number: "), gamma)
            elif choice == "2":
                M1 = _ask_float("Upstream Mach number M1: ")
                th = _ask_float("Deflection angle theta (deg): ")
                branch = input("Solution branch - weak or strong [weak]: ").strip().lower() or "weak"
                o = cf.oblique_shock(M1, theta_deg=th, weak=(branch != "strong"), gamma=gamma)
                print(f"\n  beta = {o.beta_deg:.3f} deg   Mn1 = {o.Mn1:.4f}   M2 = {o.M2:.4f}")
                print(f"  p2/p1 = {o.p2_p1:.4f}   rho2/rho1 = {o.rho2_rho1:.4f}   "
                      f"T2/T1 = {o.T2_T1:.4f}   p02/p01 = {o.p02_p01:.4f}")
            elif choice == "3":
                M1 = _ask_float("Upstream Mach number M1: ")
                turn = _ask_float("Turn angle (deg): ")
                e = cf.prandtl_meyer_expansion(M1, turn, gamma)
                print(f"\n  M2 = {e.M2:.4f}   nu1 = {e.nu1_deg:.3f}   nu2 = {e.nu2_deg:.3f} deg")
                print(f"  p2/p1 = {e.p2_p1:.4f}   rho2/rho1 = {e.rho2_rho1:.4f}   T2/T1 = {e.T2_T1:.4f}")
                print(f"  fan between Mach waves {e.mu1_deg:.2f} deg and {e.mu2_deg:.2f} deg")
            elif choice == "4":
                A = _ask_float("A/A* (>= 1): ")
                print(f"\n  Subsonic   M = {cf.mach_from_area_ratio(A, supersonic=False, gamma=gamma):.4f}")
                print(f"  Supersonic M = {cf.mach_from_area_ratio(A, supersonic=True, gamma=gamma):.4f}")
            elif choice == "5":
                M1 = _ask_float("Inlet Mach number: ")
                ratio = _ask_float("Stagnation-temperature ratio T02/T01: ")
                print(f"\n  Exit Mach number = {cf.rayleigh_exit(M1, ratio, gamma):.4f}")
            elif choice == "6":
                M1 = _ask_float("Inlet Mach number: ")
                fld = _ask_float("Friction parameter f L/D (Darcy f): ")
                print(f"\n  Exit Mach number = {cf.fanno_exit(M1, fld, gamma):.4f}")
            elif choice == "7":
                new = _ask_float("New gamma (> 1): ")
                cf.isentropic_ratios(1.0, new)  # validates
                gamma = new
            else:
                print("Unknown option.")
        except ValueError as exc:
            print(f"\n  Input error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print()
            break
    print("Exiting. Figures (if generated) are in ./" + FIG_DIR + "/")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Compressible flow calculator (calorically perfect gas).")
    ap.add_argument("--mach", type=float, help="print a one-shot summary at this Mach number")
    ap.add_argument("--gamma", type=float, default=1.4, help="ratio of specific heats (default 1.4)")
    ap.add_argument("--validate", action="store_true", help="run validation only")
    ap.add_argument("--plots", action="store_true", help="generate figures only")
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    ap.add_argument("--no-interactive", action="store_true", help="skip the interactive menu")
    ap.add_argument("--show", action="store_true", help="display plot windows (blocks until closed)")
    ap.add_argument("--test", action="store_true", help="run the built-in test-suite (needs pytest)")
    a = ap.parse_args(argv)

    try:
        cf.isentropic_ratios(1.0, a.gamma)  # validate gamma early
    except ValueError as exc:
        ap.error(str(exc))

    if a.test:
        if pytest is None:
            ap.error("pytest is not installed: pip install pytest")
        return int(pytest.main([os.path.abspath(__file__), "-q"]))
    if a.validate:
        return 0 if run_validation() else 1
    if a.plots:
        make_plots(a.gamma, a.show)
        return 0
    if a.mach is not None:
        try:
            print_summary(a.mach, a.gamma)
        except ValueError as exc:
            ap.error(str(exc))
        return 0

    ok = run_validation()
    if not a.no_plots:
        make_plots(a.gamma, a.show)
    if not a.no_interactive:
        interactive_menu(a.gamma)
    return 0 if ok else 1


# ============================================================================
# PART 3 - TEST-SUITE (formerly test_compressible_flow.py and test_main.py)
# ============================================================================

if pytest is not None:

    cli = cf          # the command-line functions live in this same module

    G = 1.4


    # ------------------------------------------------------------------ isentropic
    class TestIsentropic:
        @pytest.mark.parametrize(
            "M, T0_T, p0_p, rho0_rho, A_Astar",
            [
                (0.5, 1.0500, 1.1862, 1.1297, 1.3398),
                (1.0, 1.2000, 1.8929, 1.5774, 1.0000),
                (2.0, 1.8000, 7.8244, 4.3469, 1.6875),
                (3.0, 2.8000, 36.733, 13.119, 4.2346),  # rho0/rho = 2.8**2.5
            ],
        )
        def test_table_values(self, M, T0_T, p0_p, rho0_rho, A_Astar):
            s = cf.isentropic_ratios(M)
            assert s.T0_T == pytest.approx(T0_T, abs=5e-5)
            assert s.p0_p == pytest.approx(p0_p, rel=2e-4)
            assert s.rho0_rho == pytest.approx(rho0_rho, rel=2e-4)
            assert cf.area_mach_ratio(M) == pytest.approx(A_Astar, rel=2e-4)

        def test_ideal_gas_consistency(self):
            # p0/p = (rho0/rho) * (T0/T) for an ideal gas
            s = cf.isentropic_ratios(1.7, 1.3)
            assert s.p0_p == pytest.approx(s.rho0_rho * s.T0_T)

        def test_zero_mach_is_stagnation(self):
            assert tuple(cf.isentropic_ratios(0.0)) == (1.0, 1.0, 1.0)

        @pytest.mark.parametrize("M", [0.3, 0.9, 1.5, 4.0, 12.0])
        def test_area_inverse_roundtrip(self, M):
            A = cf.area_mach_ratio(M)
            assert cf.mach_from_area_ratio(A, supersonic=M > 1) == pytest.approx(M, rel=1e-9)

        def test_area_inverse_sonic(self):
            assert cf.mach_from_area_ratio(1.0) == 1.0

        def test_vectorised(self):
            M = np.array([0.5, 1.0, 2.0])
            out = cf.isentropic_ratios(M)
            assert out.p0_p.shape == (3,)
            assert out.p0_p[2] == pytest.approx(7.8244, rel=2e-4)

        @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
        def test_rejects_bad_mach(self, bad):
            with pytest.raises(ValueError):
                cf.isentropic_ratios(bad)

        @pytest.mark.parametrize("g", [1.0, 0.9, -1.4, np.nan])
        def test_rejects_bad_gamma(self, g):
            with pytest.raises(ValueError):
                cf.isentropic_ratios(2.0, g)

        def test_area_ratio_zero_mach_rejected(self):
            # the original code raised ZeroDivisionError here
            with pytest.raises(ValueError):
                cf.area_mach_ratio(0.0)

        def test_area_inverse_rejects_below_one(self):
            with pytest.raises(ValueError):
                cf.mach_from_area_ratio(0.9)


    # ---------------------------------------------------------------- normal shock
    class TestNormalShock:
        @pytest.mark.parametrize(
            "M1, M2, p, rho, T, p0",
            [
                (2.0, 0.5774, 4.5000, 2.6667, 1.6875, 0.7209),
                (3.0, 0.4752, 10.333, 3.8571, 2.6790, 0.3283),
                (5.0, 0.4152, 29.000, 5.0000, 5.8000, 0.06172),
            ],
        )
        def test_table_values(self, M1, M2, p, rho, T, p0):
            s = cf.normal_shock(M1)
            assert s.M2 == pytest.approx(M2, abs=1e-4)
            assert s.p2_p1 == pytest.approx(p, rel=1e-4)
            assert s.rho2_rho1 == pytest.approx(rho, rel=1e-4)
            assert s.T2_T1 == pytest.approx(T, rel=1e-4)
            assert s.p02_p01 == pytest.approx(p0, rel=2e-3)

        def test_sonic_limit_is_no_shock(self):
            assert all(v == pytest.approx(1.0) for v in cf.normal_shock(1.0))

        def test_subsonic_rejected(self):
            with pytest.raises(ValueError):
                cf.normal_shock(0.8)

        def test_downstream_subsonic_and_loses_pressure(self):
            for M1 in (1.1, 1.5, 2.5, 10.0):
                s = cf.normal_shock(M1)
                assert s.M2 < 1.0
                assert 0.0 < s.p02_p01 < 1.0

        def test_strong_shock_density_limit(self):
            # rho2/rho1 -> (g+1)/(g-1) = 6 as M1 -> infinity
            assert cf.normal_shock(1e3).rho2_rho1 == pytest.approx(6.0, rel=1e-3)

        def test_entropy_rise_positive(self):
            assert cf.shock_entropy_rise(2.0) == pytest.approx(-np.log(0.7209), rel=2e-3)
            assert cf.shock_entropy_rise(1.0) == pytest.approx(0.0, abs=1e-12)

        def test_pitot_ratio_table(self):
            assert cf.pitot_pressure_ratio(2.0) == pytest.approx(5.6404, rel=1e-4)

        def test_pitot_consistent_with_shock_plus_isentropic(self):
            M1 = 2.7
            s = cf.normal_shock(M1)
            p01_p1 = cf.isentropic_ratios(M1).p0_p
            assert cf.pitot_pressure_ratio(M1) == pytest.approx(s.p02_p01 * p01_p1)

        def test_inverse_relations(self):
            s = cf.normal_shock(3.3)
            assert cf.mach1_from_pressure_ratio(s.p2_p1) == pytest.approx(3.3)
            assert cf.mach1_from_downstream_mach(s.M2) == pytest.approx(3.3)

        def test_inverse_bounds(self):
            with pytest.raises(ValueError):
                cf.mach1_from_pressure_ratio(0.5)
            with pytest.raises(ValueError):
                cf.mach1_from_downstream_mach(0.2)  # below sqrt((g-1)/2g) = 0.378

        def test_vectorised_shock(self):
            # the original implementation failed on array input
            s = cf.normal_shock(np.array([2.0, 3.0]))
            assert s.p2_p1[1] == pytest.approx(10.333, rel=1e-4)


    # ---------------------------------------------------------------- oblique shock
    class TestObliqueShock:
        def test_table_M2_theta10(self):
            s = cf.oblique_shock(2.0, theta_deg=10.0)
            assert s.beta_deg == pytest.approx(39.31, abs=0.01)
            assert s.M2 == pytest.approx(1.641, abs=2e-3)
            assert s.p2_p1 == pytest.approx(1.707, abs=2e-3)

        def test_textbook_example_M2_theta20(self):
            # Anderson Example 4.1-style case: M1 = 2, theta = 20 deg -> beta ~ 53.4 deg
            s = cf.oblique_shock(2.0, theta_deg=20.0)
            assert s.beta_deg == pytest.approx(53.4, abs=0.1)
            assert s.Mn1 == pytest.approx(1.60, abs=0.01)
            assert s.M2 == pytest.approx(1.21, abs=0.01)
            # The textbook rounds Mn1 to 1.60 -> p2/p1 = 2.82; the exact Mn1 = 1.607 gives 2.843.
            assert s.p2_p1 == pytest.approx(2.82, abs=0.03)
            assert s.p2_p1 == pytest.approx(1 + 2.8 / 2.4 * (s.Mn1**2 - 1), rel=1e-12)

        def test_max_deflection_values(self):
            theta_max, _ = cf.max_deflection_angle(2.0)
            assert theta_max == pytest.approx(22.97, abs=0.02)
            assert cf.max_deflection_angle(1e4)[0] == pytest.approx(45.58, abs=0.05)

        def test_detached_shock_rejected(self):
            with pytest.raises(ValueError, match="detach"):
                cf.oblique_shock(2.0, theta_deg=25.0)

        def test_weak_vs_strong_branches(self):
            weak = cf.oblique_shock(3.0, theta_deg=15.0, weak=True)
            strong = cf.oblique_shock(3.0, theta_deg=15.0, weak=False)
            assert weak.beta_deg < strong.beta_deg
            assert weak.M2 > 1.0       # weak solution: supersonic downstream
            assert strong.M2 < 1.0     # strong solution: subsonic downstream
            assert strong.p2_p1 > weak.p2_p1
            # both must reproduce the same deflection angle
            for s in (weak, strong):
                assert cf.theta_from_beta(3.0, s.beta_deg) == pytest.approx(15.0, abs=1e-9)

        def test_beta_input_roundtrip(self):
            s1 = cf.oblique_shock(2.5, theta_deg=12.0)
            s2 = cf.oblique_shock(2.5, beta_deg=s1.beta_deg)
            assert s2.theta_deg == pytest.approx(12.0, abs=1e-9)
            assert s2.M2 == pytest.approx(s1.M2)

        def test_normal_shock_limit(self):
            # beta = 90 deg must reduce to the normal shock
            o = cf.oblique_shock(2.0, beta_deg=90.0)
            n = cf.normal_shock(2.0)
            assert o.theta_deg == pytest.approx(0.0, abs=1e-9)
            assert o.p2_p1 == pytest.approx(n.p2_p1)
            assert o.M2 == pytest.approx(n.M2)

        def test_zero_deflection_is_mach_wave(self):
            s = cf.oblique_shock(2.0, theta_deg=0.0)
            assert s.beta_deg == pytest.approx(30.0)  # arcsin(1/2)
            assert s.p2_p1 == pytest.approx(1.0)

        def test_requires_exactly_one_angle(self):
            with pytest.raises(ValueError):
                cf.oblique_shock(2.0)
            with pytest.raises(ValueError):
                cf.oblique_shock(2.0, theta_deg=10.0, beta_deg=40.0)

        def test_beta_below_mach_angle_rejected(self):
            with pytest.raises(ValueError):
                cf.oblique_shock(2.0, beta_deg=20.0)

        def test_tangential_velocity_conserved(self):
            # V_t is continuous across an oblique shock: check with temperatures (a ~ sqrt(T))
            M1 = 3.0
            s = cf.oblique_shock(M1, theta_deg=14.0)
            b, th = np.radians(s.beta_deg), np.radians(s.theta_deg)
            a2_a1 = np.sqrt(s.T2_T1)
            Vt1 = M1 * np.cos(b)
            Vt2 = s.M2 * a2_a1 * np.cos(b - th)
            assert Vt2 == pytest.approx(Vt1, rel=1e-9)

        def test_mach_angle(self):
            assert cf.mach_angle_deg(2.0) == pytest.approx(30.0)
            assert cf.mach_angle_deg(1.0) == pytest.approx(90.0)


    # ---------------------------------------------------------------- Prandtl-Meyer
    class TestPrandtlMeyer:
        @pytest.mark.parametrize("M, nu", [(1.0, 0.0), (2.0, 26.380), (3.0, 49.757), (5.0, 76.920)])
        def test_table_values(self, M, nu):
            assert cf.prandtl_meyer(M) == pytest.approx(nu, abs=2e-3)

        def test_limit(self):
            assert cf.prandtl_meyer_max() == pytest.approx(130.454, abs=1e-3)
            assert cf.prandtl_meyer(1e6) == pytest.approx(130.454, abs=1e-2)

        def test_monotonic(self):
            nu = cf.prandtl_meyer(np.linspace(1.0, 10.0, 200))
            assert np.all(np.diff(nu) > 0)

        @pytest.mark.parametrize("M", [1.0001, 1.5, 2.0, 5.0, 20.0])
        def test_inverse_roundtrip(self, M):
            assert cf.mach_from_pm_angle(cf.prandtl_meyer(M)) == pytest.approx(M, rel=1e-8)

        def test_inverse_zero_and_unreachable(self):
            assert cf.mach_from_pm_angle(0.0) == 1.0   # failed in the original code
            assert cf.mach_from_pm_angle(130.0) > 100  # close to the limit, still solvable
            with pytest.raises(ValueError):
                cf.mach_from_pm_angle(131.0)
            with pytest.raises(ValueError):
                cf.mach_from_pm_angle(-1.0)

        def test_subsonic_rejected(self):
            with pytest.raises(ValueError):
                cf.prandtl_meyer(0.9)

        def test_expansion_fan(self):
            e = cf.prandtl_meyer_expansion(2.0, 10.0)
            assert e.nu2_deg == pytest.approx(36.380, abs=2e-3)
            assert e.M2 == pytest.approx(2.386, abs=2e-3)
            assert e.p2_p1 < 1.0 and e.T2_T1 < 1.0 and e.rho2_rho1 < 1.0
            assert e.mu2_deg < e.mu1_deg
            # isentropic: stagnation conditions unchanged
            p0 = cf.isentropic_ratios(2.0).p0_p / cf.isentropic_ratios(e.M2).p0_p
            assert e.p2_p1 == pytest.approx(p0)

        def test_zero_turn_is_identity(self):
            e = cf.prandtl_meyer_expansion(3.0, 0.0)
            assert e.M2 == pytest.approx(3.0)
            assert e.p2_p1 == pytest.approx(1.0)


    # --------------------------------------------------------------------- Rayleigh
    class TestRayleigh:
        def test_sonic_reference(self):
            assert all(v == pytest.approx(1.0) for v in cf.rayleigh_ratios(1.0))

        @pytest.mark.parametrize(
            "M, T, p, T0, p0",
            [
                (0.5, 0.7901, 1.7778, 0.6914, 1.1141),
                (2.0, 0.5289, 0.3636, 0.7934, 1.5031),
            ],
        )
        def test_table_values(self, M, T, p, T0, p0):
            r = cf.rayleigh_ratios(M)
            assert r.T_Tstar == pytest.approx(T, abs=1e-4)
            assert r.p_pstar == pytest.approx(p, abs=1e-4)
            assert r.T0_T0star == pytest.approx(T0, abs=1e-4)
            assert r.p0_p0star == pytest.approx(p0, abs=1e-4)

        def test_velocity_density_reciprocal(self):
            r = cf.rayleigh_ratios(2.0)
            assert r.rho_rhostar * r.V_Vstar == pytest.approx(1.0)
            assert r.V_Vstar == pytest.approx(1.4545, abs=1e-4)

        @pytest.mark.parametrize("M", [0.2, 0.7, 1.8, 6.0])
        def test_inverse_roundtrip(self, M):
            r = cf.rayleigh_ratios(M).T0_T0star
            assert cf.mach_from_rayleigh_T0_ratio(r, supersonic=M > 1) == pytest.approx(M, rel=1e-8)

        def test_supersonic_floor(self):
            with pytest.raises(ValueError):
                cf.mach_from_rayleigh_T0_ratio(0.40, supersonic=True)  # floor ~0.4898

        def test_heating_accelerates_subsonic_flow(self):
            M2 = cf.rayleigh_exit(0.3, 2.0)
            assert 0.3 < M2 < 1.0

        def test_heating_decelerates_supersonic_flow(self):
            M2 = cf.rayleigh_exit(3.0, 1.5)
            assert 1.0 < M2 < 3.0

        def test_cooling_reverses_trend(self):
            assert cf.rayleigh_exit(0.5, 0.8) < 0.5

        def test_thermal_choking(self):
            ratio_choke = 1.0 / cf.rayleigh_ratios(0.5).T0_T0star
            assert cf.rayleigh_exit(0.5, ratio_choke * 0.999999) == pytest.approx(1.0, abs=1e-2)
            with pytest.raises(ValueError, match="choked"):
                cf.rayleigh_exit(0.5, ratio_choke * 1.01)


    # ------------------------------------------------------------------------ Fanno
    class TestFanno:
        def test_sonic_reference(self):
            f = cf.fanno_ratios(1.0)
            assert all(v == pytest.approx(1.0) for v in f[:5])
            assert f.fLstar_D == pytest.approx(0.0, abs=1e-12)

        @pytest.mark.parametrize(
            "M, T, p, rho, V, p0, fL",
            [
                (0.5, 1.1429, 2.1381, 1.8708, 0.5345, 1.3398, 1.0691),
                (2.0, 0.6667, 0.4082, 0.6124, 1.6330, 1.6875, 0.3050),
            ],
        )
        def test_table_values(self, M, T, p, rho, V, p0, fL):
            f = cf.fanno_ratios(M)
            assert f.T_Tstar == pytest.approx(T, abs=1e-4)
            assert f.p_pstar == pytest.approx(p, abs=1e-4)
            assert f.rho_rhostar == pytest.approx(rho, abs=1e-4)
            assert f.V_Vstar == pytest.approx(V, abs=1e-4)
            assert f.p0_p0star == pytest.approx(p0, abs=1e-4)
            assert f.fLstar_D == pytest.approx(fL, abs=1e-4)

        def test_p0_ratio_equals_area_ratio(self):
            # For Fanno flow p0/p0* is identical to the isentropic A/A*
            for M in (0.3, 0.8, 1.5, 3.0):
                assert cf.fanno_ratios(M).p0_p0star == pytest.approx(cf.area_mach_ratio(M))

        def test_ideal_gas_consistency(self):
            # p = rho R T  =>  p/p* = (rho/rho*)(T/T*)
            f = cf.fanno_ratios(0.6)
            assert f.p_pstar == pytest.approx(f.rho_rhostar * f.T_Tstar)

        def test_supersonic_limit(self):
            assert cf.fanno_parameter(1e6) == pytest.approx(0.8215, abs=1e-3)

        @pytest.mark.parametrize("M", [0.1, 0.5, 0.9, 1.5, 3.0, 10.0])
        def test_inverse_roundtrip(self, M):
            p = cf.fanno_parameter(M)
            assert cf.mach_from_fanno_parameter(p, supersonic=M > 1) == pytest.approx(M, rel=1e-7)

        def test_inverse_rejects_beyond_limit(self):
            with pytest.raises(ValueError):
                cf.mach_from_fanno_parameter(0.9, supersonic=True)

        def test_subsonic_friction_accelerates_to_sonic(self):
            M2 = cf.fanno_exit(0.5, 0.5)
            assert 0.5 < M2 < 1.0
            assert cf.fanno_exit(0.5, cf.fanno_parameter(0.5)) == pytest.approx(1.0)

        def test_supersonic_friction_decelerates(self):
            M2 = cf.fanno_exit(3.0, 0.2)
            assert 1.0 < M2 < 3.0

        def test_choked_duct(self):
            with pytest.raises(ValueError, match="Choked"):
                cf.fanno_exit(0.5, 1.2)

        def test_exit_additivity(self):
            # marching in two legs must equal one long leg
            M_mid = cf.fanno_exit(0.3, 0.4)
            M_end_two = cf.fanno_exit(M_mid, 0.5)
            M_end_one = cf.fanno_exit(0.3, 0.9)
            assert M_end_two == pytest.approx(M_end_one, rel=1e-8)


    # ------------------------------------------------- remaining input-validation paths
    class TestInputValidation:
        def test_oblique_negative_theta(self):
            with pytest.raises(ValueError):
                cf.oblique_shock(2.0, theta_deg=-5.0)

        def test_negative_expansion_turn(self):
            with pytest.raises(ValueError):
                cf.prandtl_meyer_expansion(2.0, -1.0)

        def test_rayleigh_ratio_domain(self):
            with pytest.raises(ValueError):
                cf.mach_from_rayleigh_T0_ratio(1.5, supersonic=False)
            with pytest.raises(ValueError):
                cf.rayleigh_exit(0.5, -1.0)

        def test_fanno_negative_parameter(self):
            with pytest.raises(ValueError):
                cf.mach_from_fanno_parameter(-0.1, supersonic=False)
            with pytest.raises(ValueError):
                cf.fanno_exit(0.5, -0.1)

        def test_pm_inverse_extremely_close_to_limit(self):
            # bracket expansion must either succeed or fail with a clear ValueError, never hang
            try:
                M = cf.mach_from_pm_angle(cf.prandtl_meyer_max() - 1e-9)
                assert M > 1e3
            except ValueError:
                pass


    # ------------------------------------------- regression: near-sonic inverse accuracy
    class TestNearSonicRegression:
        """np.isclose(..., atol=1e-12) used the default rtol=1e-5 and returned M=1.0 for
        M = 1.001 (error 1e-3). Quantities such as A/A* - 1 are quadratic in (M - 1)."""

        @pytest.mark.parametrize("M", [1.0005, 1.001, 1.01, 0.999, 0.99])
        def test_area_inverse_near_sonic(self, M):
            A = cf.area_mach_ratio(M)
            assert cf.mach_from_area_ratio(A, supersonic=M > 1) == pytest.approx(M, rel=1e-6)

        @pytest.mark.parametrize("M", [1.0005, 1.001, 1.01, 0.999, 0.99])
        def test_rayleigh_inverse_near_sonic(self, M):
            r = cf.rayleigh_ratios(M).T0_T0star
            assert cf.mach_from_rayleigh_T0_ratio(r, supersonic=M > 1) == pytest.approx(M, rel=1e-6)

        def test_full_sweep_supersonic_area_inverse(self):
            worst = max(
                abs(cf.mach_from_area_ratio(cf.area_mach_ratio(m), True) - m) / m
                for m in np.linspace(1.001, 30, 120)
            )
            assert worst < 1e-8


    def test_validation_passes(capsys):
        assert cli.run_validation(verbose=True) is True
        assert "16/16 checks passed" in capsys.readouterr().out


    def test_one_shot_summary(capsys):
        assert cli.main(["--mach", "2.0", "--no-plots"]) == 0
        out = capsys.readouterr().out
        assert "p2/p1 = 4.5000" in out and "nu = 26.3798" in out


    def test_subsonic_summary_skips_shock(capsys):
        cli.main(["--mach", "0.5"])
        assert "skipped" in capsys.readouterr().out


    @pytest.mark.parametrize("argv", [["--gamma", "0.8", "--mach", "2"], ["--mach", "-1"]])
    def test_bad_arguments_exit_cleanly(argv):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2


    def test_plots_written(tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "FIG_DIR", str(tmp_path))
        assert cli.main(["--plots"]) == 0
        names = set(os.listdir(tmp_path))
        assert {"isentropic_curves.png", "normal_shock_curves.png", "prandtl_meyer_curve.png",
                "theta_beta_M_chart.png", "rayleigh_curves.png", "fanno_curves.png"} <= names


    def test_interactive_menu_handles_errors(monkeypatch, capsys):
        # oblique shock past detachment -> message, not a crash; then quit
        inputs = iter(["2", "2", "25", "weak", "4", "0.5", "q"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        cli.interactive_menu()
        out = capsys.readouterr().out
        assert "detaches" in out and "A/A* must be >= 1" in out


    def test_interactive_menu_survives_eof(monkeypatch):
        def boom(*_):
            raise EOFError
        monkeypatch.setattr("builtins.input", boom)
        cli.interactive_menu()  # must return, not raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
