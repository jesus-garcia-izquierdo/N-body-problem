r"""
Time integrators for  dr/dt = v,  dv/dt = a(r).

All of them share the same signature, so they can be swapped with one argument:

    step(x, v, a, h, accel) -> (x_new, v_new, a_new)

    x, v, a : np.ndarray (N, 3)   position, velocity and acceleration at t_n
    h       : float               time step
    accel   : function (N,3)->(N,3) returning a(x)

Why ``a`` goes in and ``a_new`` comes out: the force evaluation is O(N^2) and
by far the most expensive operation. Verlet needs a(t_n) and a(t_{n+1}); since
a(t_{n+1}) of this step is a(t_n) of the next one, we keep it and reuse it
(1 force evaluation per step instead of 2). The MATLAB code recomputed it.

Summary (details and derivations in docs/NUMERICS.md)
-----------------------------------------------------
| method            | order | symplectic | time-reversible | forces/step |
|-------------------|-------|------------|-----------------|-------------|
| euler             |   1   |     no     |       no        |      1      |
| symplectic_euler  |   1   |    yes     |       no        |      1      |
| verlet            |   2   |    yes     |      yes        |      1      |
| yoshida4          |   4   |    yes     |      yes        |      3      |
| rk4               |   4   |     no     |       no        |      4      |

"Symplectic" = preserves phase-space volume (Liouville's theorem) and, in
practice, the energy does NOT drift: it oscillates, bounded, around the exact
value for extremely long times. RK4 is more accurate in the short term, but its
energy error grows without bound: for thousands of orbits, Verlet wins. See
examples/06_compare_integrators.py.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .forces import AccelerationFunction

State = tuple[np.ndarray, np.ndarray, np.ndarray]
StepFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, float, AccelerationFunction], State]


# ---------------------------------------------------------------------------
# Explicit Euler (the simplest; useful as a counter-example)
# ---------------------------------------------------------------------------
def euler_step(x, v, a, h, accel) -> State:
    """r_{n+1} = r_n + v_n h ;  v_{n+1} = v_n + a_n h.

    Local error O(h^2), global O(h). For orbits the energy grows
    systematically: planets spiral outwards. Shown only as what NOT to do.
    """
    x_new = x + v * h
    v_new = v + a * h
    return x_new, v_new, accel(x_new)


# ---------------------------------------------------------------------------
# Symplectic (semi-implicit) Euler: swapping two lines fixes almost everything
# ---------------------------------------------------------------------------
def symplectic_euler_step(x, v, a, h, accel) -> State:
    """v_{n+1} = v_n + a_n h ;  r_{n+1} = r_n + v_{n+1} h.

    Same formal accuracy as Euler (order 1) but symplectic: the energy no
    longer drifts. The simplest example that the *structure* of a method
    matters more than its order.
    """
    v_new = v + a * h
    x_new = x + v_new * h
    return x_new, v_new, accel(x_new)


# ---------------------------------------------------------------------------
# Velocity Verlet (the method of the original report)
# ---------------------------------------------------------------------------
def velocity_verlet_step(x, v, a, h, accel) -> State:
    """Equations (3)-(5) of the original report:

        r_{n+1} = r_n + v_n h + 1/2 a_n h^2
        a_{n+1} = a(r_{n+1})
        v_{n+1} = v_n + 1/2 (a_n + a_{n+1}) h

    Equivalent to "half kick - drift - half kick" (leapfrog):
        v_{n+1/2} = v_n + 1/2 a_n h ;  r_{n+1} = r_n + v_{n+1/2} h ;  v_{n+1} = v_{n+1/2} + 1/2 a_{n+1} h
    Second order, symplectic and time-reversible.
    """
    x_new = x + v * h + 0.5 * a * (h * h)
    a_new = accel(x_new)
    v_new = v + 0.5 * (a + a_new) * h
    return x_new, v_new, a_new


# ---------------------------------------------------------------------------
# Fourth-order Yoshida: three Verlet steps with "magic" weights
# ---------------------------------------------------------------------------
_CBRT2 = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)          # ~  1.3512  (forward step, longer than h)
_YOSHIDA_W0 = -_CBRT2 / (2.0 - _CBRT2)      # ~ -1.7024  (a step BACKWARDS in time!)


def yoshida4_step(x, v, a, h, accel) -> State:
    """Symmetric composition Verlet(w1 h) o Verlet(w0 h) o Verlet(w1 h).

    Idea (Yoshida, 1990): Verlet is symmetric, so its error contains only odd
    powers of h: h^3, h^5, ... Chaining three Verlet steps with weights that
    satisfy 2 w1 + w0 = 1 (total advance h) and 2 w1^3 + w0^3 = 0 (the h^3 term
    cancels) lowers the local error to O(h^5) -> a 4th-order method that is
    still symplectic, because it composes symplectic maps.
    """
    for w in (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1):
        x, v, a = velocity_verlet_step(x, v, a, w * h, accel)
    return x, v, a


# ---------------------------------------------------------------------------
# Classical 4th-order Runge-Kutta (very accurate short-term, NOT symplectic)
# ---------------------------------------------------------------------------
def rk4_step(x, v, a, h, accel) -> State:
    """RK4 applied to the first-order system y = (r, v), y' = (v, a(r)).

    k1 = f(y_n), k2 = f(y_n + h/2 k1), k3 = f(y_n + h/2 k2), k4 = f(y_n + h k3)
    y_{n+1} = y_n + h/6 (k1 + 2 k2 + 2 k3 + k4)
    ``a`` (the velocity part of k1) comes from the previous step: 4 forces per step.
    """
    k1x, k1v = v, a
    k2x, k2v = v + 0.5 * h * k1v, accel(x + 0.5 * h * k1x)
    k3x, k3v = v + 0.5 * h * k2v, accel(x + 0.5 * h * k2x)
    k4x, k4v = v + h * k3v, accel(x + h * k3x)
    x_new = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_new, v_new, accel(x_new)


# ---------------------------------------------------------------------------
# Registry (choose an integrator by name)
# ---------------------------------------------------------------------------
class IntegratorInfo(NamedTuple):
    step: StepFunction
    order: int
    symplectic: bool
    force_evals: int
    label: str


INTEGRATORS: dict[str, IntegratorInfo] = {
    "euler":            IntegratorInfo(euler_step, 1, False, 1, "Explicit Euler"),
    "symplectic_euler": IntegratorInfo(symplectic_euler_step, 1, True, 1, "Symplectic Euler"),
    "verlet":           IntegratorInfo(velocity_verlet_step, 2, True, 1, "Velocity Verlet"),
    "yoshida4":         IntegratorInfo(yoshida4_step, 4, True, 3, "Yoshida-4"),
    "rk4":              IntegratorInfo(rk4_step, 4, False, 4, "Runge-Kutta 4"),
}


def get_integrator(name: str) -> IntegratorInfo:
    try:
        return INTEGRATORS[name]
    except KeyError:
        raise ValueError(f"Unknown integrator '{name}'. Options: {list(INTEGRATORS)}") from None
