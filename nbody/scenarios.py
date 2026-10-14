r"""
Ready-to-use initial conditions.

Every function returns an ``NBodySystem``. Three families:

1. **From the original report** (SI units): ``sun_earth_moon``,
   ``four_body_report``, ``six_body_report``, ``random_system``.
2. **Famous solutions** (N-body units, G = 1): ``figure_eight``,
   ``lagrange_triangle``, ``pythagorean``.
3. **Larger systems**: ``solar_system`` (SI), ``plummer_cluster`` and
   ``uniform_sphere`` (N-body units).

About N-body units (Hénon, 1971)
--------------------------------
For abstract problems it is convenient to choose units with G = 1, total mass
M = 1 and a characteristic length of 1. Numbers stay of order one, products
like 1e30 * 1e-11 are avoided, and results hold at any physical scale (just
rescale: doubling all distances multiplies times by 2^{3/2}, as Kepler's third
law says).
"""

from __future__ import annotations

import numpy as np

from . import constants as c
from .forces import potential_energy
from .system import NBodySystem


# ===========================================================================
# 1. Scenarios from the original report
# ===========================================================================
def sun_earth_moon(center_of_mass_frame: bool = True) -> NBodySystem:
    """Coplanar Sun-Earth-Moon: Earth at aphelion, Moon at apogee.

    Fixes two details of Table 1 of the report: the Moon's mass (7.342e22 kg;
    the table lists 2e40 kg, about 10^10 solar masses) and Earth's speed at
    aphelion (29.29 km/s). With Earth and Moon moving but the Sun at rest the
    total momentum is not zero and the whole system drifts slowly; therefore
    we move to the CM frame by default.
    """
    x_e, v_e = c.EARTH_APHELION, c.EARTH_APHELION_SPEED
    sys = NBodySystem(
        masses=[c.M_SUN, c.M_EARTH, c.M_MOON],
        positions=[[0, 0, 0], [x_e, 0, 0], [x_e + c.MOON_APOGEE, 0, 0]],
        velocities=[[0, 0, 0], [0, v_e, 0], [0, v_e + c.MOON_APOGEE_SPEED, 0]],
        G=c.G_SI, names=["Sun", "Earth", "Moon"],
    )
    return sys.to_center_of_mass_frame() if center_of_mass_frame else sys


def four_body_report(G: float = c.G_SI) -> NBodySystem:
    """Four-body system of Table 2 of the report (reproduced as is).

    A very massive central body (2e40 kg, ~10^10 solar masses, like a galactic
    nucleus), two ~1 solar-mass bodies in the XY plane and a light fourth body
    in the perpendicular plane. Note: the speeds use sqrt(G m3 / x0(2)) with
    x0(2) = 1.496e14 m, TWICE the actual distance to the central body; that is
    why orbits 1 and 2 are ellipses (e = 0.5) instead of circles.
    """
    m_sun = 1.9891e30
    d = 1.496e14
    m = np.array([2.02 * m_sun, 0.98 * m_sun, 2e40, 2e20])
    pos = np.array([[0, 0, 0], [d, 0, 0], [d / 2, 0, 0], [d / 2, 0, 1e14]])
    v12 = np.sqrt(G * m[2] / pos[1, 0])
    v4 = np.sqrt(G * m[2] / pos[3, 2])
    vel = np.array([[0, -v12, 0], [0, v12, 0], [0, 0, 0], [-v4, 0, 0]])
    return NBodySystem(m, pos, vel, G=G)


def six_body_report() -> NBodySystem:
    """Six 1e10 kg bodies on the ±x, ±y, ±z semi-axes at 1e5 m (Section 4.3).

    Each speed (~2.1 mm/s) is that of a circular orbit around the next body if
    the two were alone. This is the chaos example of the report.
    """
    d, u = 1e5, 0.0015
    pos = d * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    vel = u * np.array([[1, 1, 0], [0, 1, 1], [-1, 0, 1], [-1, -1, 0], [0, -1, -1], [1, 0, -1]])
    return NBodySystem(np.full(6, 1e10), pos, vel, G=c.G_SI)


def random_system(n_bodies: int = 3, seed: int | None = None) -> NBodySystem:
    """Python version of the "Juego de azar" (game of chance) script: random initial conditions.

    Same ranges as the original: m in [0, 1e10] kg, r in [-1e5, 1e5] m per
    component, v in [-1.5, 1.5] mm/s per component. A seeded generator
    (``np.random.default_rng``) makes the experiment reproducible.
    """
    rng = np.random.default_rng(seed)
    m = 1e10 * rng.uniform(0.05, 1.0, n_bodies)       # avoid masses ~0
    pos = rng.uniform(-1e5, 1e5, (n_bodies, 3))
    vel = rng.uniform(-1.5e-3, 1.5e-3, (n_bodies, 3))
    return NBodySystem(m, pos, vel, G=c.G_SI)


# ===========================================================================
# 2. Famous solutions (G = 1)
# ===========================================================================
FIGURE_EIGHT_PERIOD = 6.325_913_985_3


def figure_eight() -> NBodySystem:
    """Figure-eight choreography (Moore 1993; Chenciner and Montgomery 2000).

    Three equal masses chase each other along the SAME figure-eight curve,
    one third of a period apart. It is an exact periodic solution of the
    three-body problem, found numerically almost 300 years after Newton, and
    it is stable (KAM) against small perturbations.
    """
    x1 = np.array([-0.970_004_36, 0.243_087_53])
    v3 = np.array([-0.932_407_37, -0.864_731_46])
    return NBodySystem(
        masses=[1.0, 1.0, 1.0],
        positions=[x1, -x1, [0.0, 0.0]],
        velocities=[-v3 / 2, -v3 / 2, v3],
        G=1.0, names=["A", "B", "C"],
    )


def lagrange_triangle(masses=(1.0, 1.0, 1.0), side: float = 1.0, G: float = 1.0) -> NBodySystem:
    """Lagrange's solution (1772): an equilateral triangle in rigid rotation.

    For ANY masses, if the bodies sit at the vertices of an equilateral
    triangle of side d and rotate about the CM with omega^2 = G M / d^3, the
    configuration is preserved. It is stable only if (Routh's criterion)
    27 (m1 m2 + m2 m3 + m3 m1) < M^2, i.e. if one mass dominates
    (Sun-Jupiter-Trojans). With equal masses it is UNSTABLE: round-off and
    truncation errors grow exponentially and the triangle breaks up.
    """
    m = np.asarray(masses, dtype=float)
    ang = np.deg2rad([90.0, 210.0, 330.0])
    pos = (side / np.sqrt(3.0)) * np.column_stack([np.cos(ang), np.sin(ang), np.zeros(3)])
    pos -= m @ pos / m.sum()                              # origin at the CM
    omega = np.sqrt(G * m.sum() / side ** 3)
    vel = omega * np.column_stack([-pos[:, 1], pos[:, 0], np.zeros(3)])   # v = omega z x r
    return NBodySystem(m, pos, vel, G=G)


def routh_stable(masses) -> bool:
    """Routh's criterion for the linear stability of the Lagrange triangle."""
    m1, m2, m3 = masses
    return 27 * (m1 * m2 + m2 * m3 + m3 * m1) < (m1 + m2 + m3) ** 2


def lagrange_growth_rate(masses, side: float = 1.0, G: float = 1.0) -> float:
    """Growth rate of the Lagrange triangle predicted by linear theory.

    Linearizing in the rotating frame, the exponents lambda satisfy
        lambda^4 + omega^2 lambda^2 + (27/4) beta omega^4 = 0,
        beta = (m1 m2 + m2 m3 + m3 m1) / M^2.
    When 27 beta > 1 the roots lambda^2 are complex and lambda has a positive
    real part (0 means linearly stable). For equal masses: omega / sqrt(2).
    """
    m = np.asarray(masses, dtype=float)
    M = m.sum()
    beta = (m[0] * m[1] + m[1] * m[2] + m[2] * m[0]) / M ** 2
    omega = np.sqrt(G * M / side ** 3)
    lam2 = np.roots([1.0, omega ** 2, 6.75 * beta * omega ** 4])     # roots in the variable lambda^2
    return float(max(np.max(np.sqrt(lam2.astype(complex)).real), 0.0))


def pythagorean() -> NBodySystem:
    """Burrau's Pythagorean problem (1913): masses 3, 4, 5 at rest on a 3-4-5 triangle.

    After several very close encounters a binary forms and the third body is
    ejected (Szebehely and Peters, 1967). A CHALLENGE for fixed-step
    integrators: during encounters the distance drops to ~1e-3 and h ~ 1e-6
    would be needed. See "Ideas for contributions" in the README.
    """
    return NBodySystem(masses=[3.0, 4.0, 5.0],
                       positions=[[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]],
                       velocities=np.zeros((3, 2)), G=1.0, names=["m=3", "m=4", "m=5"])


# ===========================================================================
# 3. Larger systems
# ===========================================================================
# (name, mass [kg], semi-major axis [AU], eccentricity, inclination [deg])
PLANETS = [
    ("Mercury", 3.3011e23, 0.387_10, 0.2056, 7.00),
    ("Venus",   4.8675e24, 0.723_33, 0.0068, 3.39),
    ("Earth",   5.9722e24, 1.000_00, 0.0167, 0.00),
    ("Mars",    6.4171e23, 1.523_68, 0.0934, 1.85),
    ("Jupiter", 1.8982e27, 5.2026,   0.0489, 1.30),
    ("Saturn",  5.6834e26, 9.5549,   0.0565, 2.49),
    ("Uranus",  8.6810e25, 19.2184,  0.0457, 0.77),
    ("Neptune", 1.024_13e26, 30.110, 0.0113, 1.77),
]


def solar_system(n_planets: int = 8, seed: int | None = 1) -> NBodySystem:
    """Sun + planets with real masses, semi-major axes, eccentricities and inclinations.

    Illustrative approximation (not the ephemeris of a specific date): each
    planet starts at perihelion with a random phase (reproducible through the
    seed), and its orbital plane is tilted by rotating about the Sun-perihelion
    line. The perihelion speed follows from the vis-viva equation:

        v^2 = G M (2/r - 1/a)  with  r = a(1 - e)  ->  v_p = sqrt(G M (1+e) / (a(1-e)))
    """
    rng = np.random.default_rng(seed)
    names, masses, pos, vel = ["Sun"], [c.M_SUN], [np.zeros(3)], [np.zeros(3)]
    for name, m, a_au, e, inc in PLANETS[:n_planets]:
        a = a_au * c.AU
        r_p = a * (1 - e)
        v_p = np.sqrt(c.G_SI * c.M_SUN * (1 + e) / r_p)
        th = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th), 0.0])            # perihelion direction
        w = np.array([-np.sin(th), np.cos(th), 0.0])           # in-plane velocity direction (perpendicular to u)
        i = np.deg2rad(inc)
        w = w * np.cos(i) + np.cross(u, w) * np.sin(i)         # Rodrigues: rotate by angle i about u
        names.append(name); masses.append(m); pos.append(r_p * u); vel.append(v_p * w)
    return NBodySystem(masses, pos, vel, G=c.G_SI, names=names).to_center_of_mass_frame()


def plummer_cluster(n_bodies: int = 100, seed: int | None = 0, softening: float = 0.02) -> NBodySystem:
    """Star cluster in equilibrium: Plummer sphere (1911) in N-body units.

    Density rho(r) ~ (1 + r^2/a^2)^{-5/2}. Sampling algorithm of Aarseth,
    Hénon and Wielen (1974):
      * radius: invert the cumulative mass M(r)/M = r^3/(1+r^2)^{3/2}
        -> r = (X^{-2/3} - 1)^{-1/2},  X ~ U(0,1)
      * speed: |v| = q * v_escape(r), with q drawn by rejection from
        g(q) ~ q^2 (1 - q^2)^{7/2}  (the Plummer distribution function)
      * isotropic directions.
    Then rescaled to standard units (G = M = 1, E = -1/4) and moved to the CM.
    The cluster is born virialized: 2K/|U| ~ 1 from t = 0.
    """
    rng = np.random.default_rng(seed)

    def isotropic(n):
        cos_t = rng.uniform(-1, 1, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        sin_t = np.sqrt(1 - cos_t ** 2)
        return np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])

    # Radii (the outermost 0.1 % is discarded to avoid r -> infinity)
    X = rng.uniform(1e-10, 0.999, n_bodies)
    r = 1.0 / np.sqrt(X ** (-2.0 / 3.0) - 1.0)

    # Speeds by von Neumann rejection sampling
    q = np.empty(n_bodies)
    filled = 0
    while filled < n_bodies:
        qq = rng.uniform(0, 1, n_bodies)
        yy = rng.uniform(0, 0.1, n_bodies)                      # max of g(q) ~ 0.092
        ok = qq[yy < qq ** 2 * (1 - qq ** 2) ** 3.5]
        take = min(ok.size, n_bodies - filled)
        q[filled:filled + take] = ok[:take]
        filled += take
    v = q * np.sqrt(2.0) * (1.0 + r ** 2) ** -0.25

    scale = 3 * np.pi / 16                                     # Plummer radius in standard units
    pos = isotropic(n_bodies) * (r * scale)[:, None]
    vel = isotropic(n_bodies) * (v / np.sqrt(scale))[:, None]
    m = np.full(n_bodies, 1.0 / n_bodies)
    return NBodySystem(m, pos, vel, G=1.0, softening=softening).to_center_of_mass_frame()


def uniform_sphere(n_bodies: int = 100, virial_ratio: float = 0.0, seed: int | None = 0,
                   softening: float = 0.05) -> NBodySystem:
    """Homogeneous sphere of radius 1 with random, rescaled velocities.

    ``virial_ratio`` = initial 2K/|U|. With 0 ("cold collapse") the stars start
    at rest, the cluster collapses, bounces and, through *violent relaxation*
    (Lynden-Bell, 1967), settles into a quasi-stationary state with
    2K/|U| ~ 1. The virial theorem "in action".
    """
    rng = np.random.default_rng(seed)
    pts = []
    while sum(len(p) for p in pts) < n_bodies:                 # rejection sampling in the cube
        cand = rng.uniform(-1, 1, (2 * n_bodies, 3))
        pts.append(cand[np.sum(cand ** 2, axis=1) <= 1.0])
    pos = np.concatenate(pts)[:n_bodies]
    m = np.full(n_bodies, 1.0 / n_bodies)
    vel = rng.normal(size=(n_bodies, 3))
    sys = NBodySystem(m, pos, vel, G=1.0, softening=softening).to_center_of_mass_frame()
    # Rescale |v| to obtain the requested virial ratio
    U = potential_energy(sys.positions, sys.masses, sys.G, sys.softening)
    K = 0.5 * np.sum(sys.masses * np.sum(sys.velocities ** 2, axis=1))
    sys.velocities *= np.sqrt(virial_ratio * abs(U) / (2 * K)) if K > 0 else 0.0
    return sys
