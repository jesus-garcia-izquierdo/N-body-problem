# From the original MATLAB project to this package

This repository grew out of the course project *Problema de los n-cuerpos* (Ginés Haro Pérez
and Jesús García Izquierdo, April 2023), written in MATLAB. The original code is kept in
[`legacy_matlab/`](../legacy_matlab) (with its original Spanish comments) for comparison.

The physics and the numerical method (velocity Verlet) are the same. What changes is the
**organization**, the **efficiency** and the **scope**.

## Piece-by-piece correspondence

| Original MATLAB | Python | Comment |
|---|---|---|
| `Gravedad_clasica.m` (2 bodies) | `simulate(...)` with N = 2 | No special case needed |
| `Gravedad_clasica_2.m` | `simulation.simulate` + `integrators.velocity_verlet_step` | Time loop separated from the integrator step |
| function `a(M, Δx, Δy, Δz)` | `forces.accelerations` | Vectorized over all bodies at once |
| function `Epot1` and its double loop | `diagnostics.potential_energy` | Sum over pairs $i<j$, no division by 2 |
| `px = m'*vx` | `diagnostics.linear_momentum` | Returns the vector, not only its modulus |
| "Momento angular" block | `diagnostics.angular_momentum` | `np.cross` over all bodies and instants |
| `Juego_de_azar.m` | `scenarios.random_system(seed=...)` | Seeded: reproducible |
| matrices `x`, `y`, `z` (body × instant) | `Trajectory.positions` (instant × body × 3) | A single 3D array |
| `plot(t, E - mean(E))` | `diagnostics.relative_error` | Error relative to the **initial** value |
| MATLAB figure windows | `nbody/plotting.py`, `nbody/video.py` | Publication figures and 3D MP4 videos |

## Efficiency improvements

1. **Memory allocated up front.** `x = [x x1]` copies the whole matrix at every step: the total
   cost grows as $T^2$. Now memory is allocated with `np.empty` and each step writes into its
   slot ($\mathcal O(T)$).
2. **One force evaluation per step instead of two.** The acceleration at $t_{n+1}$ is reused as
   the one at $t_n$ in the next step.
3. **No loops over bodies.** The separations of all pairs are computed with broadcasting (see
   NUMERICS.md §8).
4. **Selective storage** (`save_every`): a small step is only needed for accuracy, not for
   plotting.

## Things that have been fixed or clarified

Reviewing old code calmly always reveals details. None of them changes the conclusions of the
report, but they are worth knowing.

**In the code**

* *Silent collisions.* Self-interaction was removed by searching for `NaN` and setting them
  to 0. But if two **different** bodies coincided, their force would also be `NaN` and would be
  set to zero without warning. Now the diagonal is excluded explicitly, and a real collision
  produces a `NaN` that propagates and is noticed instead of hidden.
* A non-integer `np = tf/h` makes `for i=1:np` truncate silently; `t(i+1)=t(i)+h` accumulates
  round-off. Now `n_steps = ceil(t_final/h)` and $t_n = n\,h$.
* In function `a`, `az` was not transposed (unlike `ax` and `ay`); since it is only summed
  afterwards, the result was not affected.
* `Gravedad_clasica_3.m` was a draft: the potential energy was evaluated only at the initial
  instant inside the loop and the angular momentum used only body 1. It has not been ported.
* `Juego_de_azar.m` computed `v1 … v3z` but did not use them, and the comment "1 año" (1 year)
  accompanies `tf = 100*31536000` (100 years).

**In the report**

* Table 1: the Moon's mass is listed as `2e40` kg (about $10^{10}$ solar masses); the real value
  is $7.342\times10^{22}$ kg. Earth's speed at aphelion is 29.29 km/s (the table lists 28.76 km/s).
* Figure 1: Earth's orbit has eccentricity close to **0** ($e = 0.0167$), not 1 ($e=1$ would be
  a parabola).
* Equations (8) and (10): the third term repeats $p_y$ ($L_y$); it should be $p_z$ ($L_z$).
* Section 4.2: the speeds use $\sqrt{Gm_3/x_0(2)}$ with $x_0(2)$ twice the real distance to the
  central body, so the orbits are not circular but ellipses with $e = 0.5$ (confirmed by
  example 2).
* Section 4.3: $x_{\text{body 1}} = 9.9\times10^4$ m is a 1 % change (not 0.1 %);
  $v_y = 0.016$ m/s and $m = 9.99\times10^4$ kg look like typos for $0.0015015$ m/s and
  $9.99\times10^9$ kg.
* Subtracting the mean to visualize conservation hides slow drifts (the mean depends on the
  whole series). $|Q(t)-Q(0)|/|Q(0)|$ on a log scale is more informative.
* The virial theorem only applies to bound, bounded systems, in the centre-of-mass frame (see
  THEORY.md §6).

## What has been added

* Four more integrators (Euler, symplectic Euler, Yoshida-4, RK4) to understand *why* Verlet
  works.
* A quantitative measure of chaos: Lyapunov exponent (Benettin) and the $h$ vs $h/2$ comparison.
* Classic scenarios: figure-eight, Lagrange triangle (with the theoretical instability rate
  verified), Solar System with Kepler's third law, Plummer clusters and cold collapse.
* Gravitational softening, N-body units, 3D MP4 videos, a command-line interface, automated
  tests and the PDF reports in English and Spanish.
