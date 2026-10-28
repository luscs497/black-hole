"""Validações físicas (M = 1, G = c = 1)."""
import math

import numpy as np
import pytest

from blackhole.geodesic import (CAPTURED, ESCAPED, horizon_radius, isco_radius,
                                null_pr, orbit_equation_deflection, trace_simple, hamiltonian)
from blackhole.tracer import camera_frame, emitter_ut, trace_pixel, NPARAM, P_A, P_RH, \
    P_RIN, P_ROUT, P_STEP, P_MAXSTEPS, P_RESC, P_HMAX, P_SKYB, P_MAXCROSS

B_CRIT = 3 * math.sqrt(3)
EQ = math.pi / 2


def shoot(b, a=0.0, r0=1e4, step=0.01, max_steps=200_000, r_stop=None):
    """Fóton no plano equatorial vindo de longe com parâmetro de impacto b = L/E."""
    pr = null_pr(r0, EQ, 0.0, b, a, -1.0)
    return trace_simple(r0, EQ, 0.0, pr, 0.0, b, a, 1.0, r_stop or r0 * 1.0000001, step, max_steps)


def critical_b(a, sign, lo, hi, iters=40):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        st = shoot(sign * mid, a)[0]
        if st == CAPTURED:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- horizonte
def test_horizon_radius():
    assert horizon_radius(0.0) == pytest.approx(2.0)
    assert horizon_radius(0.9) == pytest.approx(1 + math.sqrt(1 - 0.81))


def test_radial_ray_falls_to_horizon():
    st, r, *_ = shoot(0.0, r0=50.0)
    assert st == CAPTURED
    assert r == pytest.approx(2.0, abs=0.01)


# ------------------------------------------------------------ esfera de fótons
def test_photon_sphere_orbit_equation_equilibrium():
    # u'' + u = 3u² tem ponto fixo (órbita circular) em u = 1/3 -> r = 3M
    u = 1 / 3
    assert 3 * u * u - u == pytest.approx(0.0, abs=1e-15)


def test_near_critical_ray_skims_photon_sphere():
    # Um raio com b ligeiramente acima de b_c gira várias vezes perto de r = 3M
    st, r, th, ph, rmin, n = shoot(B_CRIT * (1 + 1e-7), step=0.005)
    assert st == ESCAPED
    assert rmin == pytest.approx(3.0, rel=2e-3)
    assert abs(ph) > 4 * math.pi  # deu voltas em torno do buraco


# ----------------------------------------------------------------- deflexão
@pytest.mark.parametrize("b", [300.0, 1000.0, 3000.0])
def test_weak_deflection(b):
    r0 = 1e7
    st, rf, th, ph, rmin, n = shoot(b, r0=r0, step=0.002, r_stop=r0)
    assert st == ESCAPED
    straight = math.pi - math.asin(b / r0) - math.asin(b / rf)
    defl_ham = abs(ph) - straight
    defl_orb, cap = orbit_equation_deflection(b, 1e-4)
    assert not cap
    # integrador Hamiltoniano == equação orbital u'' + u = 3u²
    assert defl_ham == pytest.approx(defl_orb, rel=5e-3)
    # limite de campo fraco 4M/b (+ correção de 2ª ordem 15πM²/4b²)
    assert defl_ham == pytest.approx(4 / b, rel=1.5 * (15 * math.pi / 16) / b + 3e-3)
    assert defl_ham == pytest.approx(4 / b + 15 * math.pi / (4 * b * b), rel=5e-3)


def test_orbit_equation_capture_threshold():
    assert orbit_equation_deflection(B_CRIT * 0.999, 1e-4)[1]
    assert not orbit_equation_deflection(B_CRIT * 1.001, 1e-4)[1]


# ------------------------------------------------------------------- sombra
def test_shadow_critical_impact_parameter():
    bc = critical_b(0.0, 1.0, 4.0, 6.0)
    assert bc == pytest.approx(B_CRIT, rel=1e-3)


def test_camera_sees_shadow_of_radius_bc():
    """O raio angular da sombra visto pela câmera corresponde a b_c = 3√3 M."""
    rc = 1000.0
    P = np.zeros(NPARAM)
    P[P_A] = 0.0; P[P_RH] = 2.0; P[P_RIN] = 1e9; P[P_ROUT] = 0.0  # sem disco
    P[P_STEP] = 0.01; P[P_MAXSTEPS] = 200_000; P[P_RESC] = 2 * rc; P[P_HMAX] = 0.1
    P[P_SKYB] = 1.0; P[P_MAXCROSS] = 0
    hfov = 2.0
    C = camera_frame(rc, EQ, 0.0, 0.0, hfov, 100, 100)
    sky = np.ones((4, 8, 3), dtype=np.float32)
    lut = np.ones((2048, 3)); tex = np.zeros((1, 4)); out = np.empty(3)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        code = trace_pixel(mid, 0.0, C, P, sky, lut, tex, out)
        if code % 4 == CAPTURED:
            lo = mid
        else:
            hi = mid
    psi = math.atan(0.5 * (lo + hi) * math.tan(math.radians(hfov) / 2))
    b_measured = rc * math.sin(psi) / math.sqrt(1 - 2 / rc)
    assert b_measured == pytest.approx(B_CRIT, rel=2e-3)


# ------------------------------------------------------------- disco / redshift
def test_isco():
    assert isco_radius(0.0) == pytest.approx(6.0)
    assert isco_radius(0.9) == pytest.approx(2.3209, rel=1e-3)
    assert isco_radius(1.0 - 1e-9) == pytest.approx(1.0, abs=0.01)


def test_redshift_face_on():
    # fóton com L = 0 (visto de cima, observador no infinito): g = sqrt(1 - 3M/r)
    # = redshift gravitacional sqrt(1-2M/r) × Doppler transversal
    for r in (6.0, 10.0, 20.0):
        g = 1.0 / emitter_ut(r, 0.0)
        assert g == pytest.approx(math.sqrt(1 - 3 / r))
        assert g < math.sqrt(1 - 2 / r)


def test_hamiltonian_constraint_initial():
    for a in (0.0, 0.9):
        for b in (2.0, 7.0, -4.0):
            pr = null_pr(30.0, 1.2, 0.7, b, a, -1.0)
            y = np.array([0.0, 30.0, 1.2, 0.0, pr, 0.7])
            assert abs(hamiltonian(y, b, a)) < 1e-12


# --------------------------------------------------------------------- Kerr
def kerr_equatorial_bc(a, prograde):
    r = 2 * (1 + math.cos(2 / 3 * math.acos(-a if prograde else a)))
    return abs((r * r * (3 - r) - a * a * (r + 1)) / (a * (r - 1)))


@pytest.mark.parametrize("a", [0.5, 0.9])
def test_kerr_shadow_asymmetry(a):
    b_pro = critical_b(a, 1.0, 1.5, 6.0)
    b_ret = critical_b(a, -1.0, 4.0, 8.0)
    assert b_pro == pytest.approx(kerr_equatorial_bc(a, True), rel=2e-3)
    assert b_ret == pytest.approx(kerr_equatorial_bc(a, False), rel=2e-3)
    assert b_pro < B_CRIT < b_ret  # sombra achatada/deslocada
