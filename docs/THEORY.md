# Theory: the gravitational N-body problem

This document collects all the physics used by the code, in a self-contained way.
Every new concept is preceded by *why* we need it. A longer, illustrated version with
all the results is in the PDF report ([English](../documents_english) ·
[Spanish](../documents_spanish)).

**Contents**

1. [Setting up the problem](#1-setting-up-the-problem)
2. [Two bodies: the Kepler problem](#2-two-bodies-the-kepler-problem)
3. [Why there is no general solution for N ≥ 3](#3-why-there-is-no-general-solution-for-n--3)
4. [Hamiltonian formulation and phase space](#4-hamiltonian-formulation-and-phase-space)
5. [Conservation laws](#5-conservation-laws)
6. [The virial theorem](#6-the-virial-theorem)
7. [Special solutions of the three-body problem](#7-special-solutions-of-the-three-body-problem)
8. [Chaos and Lyapunov exponents](#8-chaos-and-lyapunov-exponents)
9. [Gravitational softening](#9-gravitational-softening)
10. [N-body units](#10-n-body-units)
11. [References](#11-references)

---

## 1. Setting up the problem

We have $N$ point masses $m_i$ at positions $\mathbf r_i(t)$ that interact only through
Newtonian gravity. The force exerted by $j$ on $i$ is

$$
\mathbf F_{ij} = G\,\frac{m_i m_j}{|\mathbf r_j-\mathbf r_i|^{3}}\,(\mathbf r_j-\mathbf r_i),
$$

attractive (it points from $i$ towards $j$) with magnitude $G m_i m_j / r_{ij}^2$. By the
superposition principle the equation of motion of each body is

$$
\boxed{\;\ddot{\mathbf r}_i = G\sum_{j\neq i} m_j\,\frac{\mathbf r_j-\mathbf r_i}{|\mathbf r_j-\mathbf r_i|^{3}}, \qquad i=1,\dots,N\;}
$$

These are $3N$ coupled second-order differential equations or, equivalently, $6N$
first-order ones for $(\mathbf r_i,\mathbf v_i)$. Given initial positions and velocities the
solution exists and is unique (as long as there are no collisions), but **we cannot write
it down** in general. That is why we integrate numerically.

> In the code: `nbody/forces.py::accelerations` evaluates the right-hand side for all bodies at once.

## 2. Two bodies: the Kepler problem

The only solvable case, and our main way to validate the code.

**Separating the centre of mass.** Define the centre of mass
$\mathbf R = (m_1\mathbf r_1+m_2\mathbf r_2)/M$ and the relative position
$\mathbf r=\mathbf r_2-\mathbf r_1$. Adding the two equations,
$M\ddot{\mathbf R}=\mathbf F_{12}+\mathbf F_{21}=0$: the CM moves in a straight line.
Subtracting them,

$$
\mu\,\ddot{\mathbf r} = -\frac{G m_1 m_2}{r^3}\,\mathbf r, \qquad \mu=\frac{m_1 m_2}{m_1+m_2}\ \text{(reduced mass)}.
$$

A two-body problem reduces to **one** body of mass $\mu$ in a central field. Since the
force is central, $\mathbf L$ is conserved, the motion is planar, and the solution is a
conic (an ellipse if $E<0$).

**Results used in the examples:**

* *Vis-viva equation* (energy conservation along the orbit):
  $v^2 = GM\left(\dfrac{2}{r}-\dfrac{1}{a}\right)$, with $a$ the semi-major axis. It gives the
  perihelion speed in `scenarios.solar_system` and explains why the orbits of example 2 have
  $e=0.5$.
* *Kepler's third law*: $T^2 = \dfrac{4\pi^2}{GM}a^3$. Example 7 measures $T$ in the
  simulation and recovers the slope $3/2$ on a log-log plot.
* *Circular orbit*: $v_c=\sqrt{GM/r}$ (equate $v^2/r$ with $GM/r^2$). The rule the original
  report used to choose initial speeds.

## 3. Why there is no general solution for N ≥ 3

To "solve" a dynamical system with $n$ degrees of freedom one generally needs enough
conserved quantities (first integrals) to reduce it to quadratures. The N-body problem has
only **10 classical integrals**: energy (1), linear momentum (3), angular momentum (3) and
the uniform motion of the CM (3). For $N=2$ (12 dimensions) they suffice; for $N=3$
(18 dimensions) they do not.

* **Bruns (1887)** proved that there are no further *algebraic* integrals.
* **Poincaré (1890)**, in the memoir that won King Oscar II's prize, showed that there are
  no additional analytic integrals either, and in the process discovered the behaviour we
  now call **chaos** (§8).
* **Sundman (1912)** found a convergent series for $N=3$, but it converges so slowly
  (about $10^{8\,000\,000}$ terms would be needed) that it is useless in practice.

Conclusion: the only general route is numerical integration, and the way to trust it is to
check that it respects what we do know: the conservation laws (§5).

## 4. Hamiltonian formulation and phase space

*Why we need it:* to understand why some integrators conserve energy in the long run and
others do not (see [NUMERICS.md](NUMERICS.md)) we must look at the dynamics in **phase
space**, the $6N$-dimensional space of all positions and momenta
$(\mathbf q,\mathbf p)$. A point in that space is a complete state of the system; its
evolution is a curve.

The Hamiltonian (total energy written in terms of $\mathbf q$ and $\mathbf p$) is

$$
H(\mathbf q,\mathbf p) = \underbrace{\sum_i \frac{|\mathbf p_i|^2}{2m_i}}_{T(\mathbf p)}\;
\underbrace{-\sum_{i<j}\frac{G m_i m_j}{|\mathbf q_i-\mathbf q_j|}}_{V(\mathbf q)},
$$

and Hamilton's equations $\dot{\mathbf q}_i=\partial H/\partial\mathbf p_i=\mathbf p_i/m_i$,
$\dot{\mathbf p}_i=-\partial H/\partial\mathbf q_i$ reproduce Newton's law.

Two properties of the Hamiltonian flow that good integrators imitate:

1. **Liouville's theorem**: the flow preserves volume in phase space (a "drop" of initial
   conditions deforms but keeps its volume). More strongly, it preserves the *symplectic
   form* $\sum_i d\mathbf q_i\wedge d\mathbf p_i$ (intuitively: the projected areas on every
   $(q_k,p_k)$ plane). An integrator with this property is called **symplectic**.
2. **Time reversibility**: if at some instant all velocities are reversed, the system
   retraces its history backwards.

Note that $H=T(\mathbf p)+V(\mathbf q)$ is **separable**: the kinetic part depends only on
momenta and the potential part only on positions. That is what makes Verlet possible
(NUMERICS.md §4).

## 5. Conservation laws

**Noether's theorem** associates a conserved quantity with every continuous symmetry. An
isolated N-body system has three:

| Symmetry | Conserved quantity | Formula |
|---|---|---|
| Space translation | Linear momentum | $\mathbf P=\sum_i m_i\mathbf v_i$ |
| Rotation | Angular momentum | $\mathbf L=\sum_i \mathbf r_i\times m_i\mathbf v_i$ |
| Time translation | Energy | $E=K+U$ |

Direct proofs (useful because they show *what* is needed numerically):

* $\dot{\mathbf P}=\sum_i\sum_{j\ne i}\mathbf F_{ij}=0$ because $\mathbf F_{ij}=-\mathbf F_{ji}$
  (third law). **Any** integrator using exactly antisymmetric pair forces conserves
  $\mathbf P$ up to round-off.
* $\dot{\mathbf L}=\sum_i \mathbf r_i\times\mathbf F_i=\sum_{i<j}(\mathbf r_i-\mathbf r_j)\times\mathbf F_{ij}=0$
  because $\mathbf F_{ij}\parallel(\mathbf r_i-\mathbf r_j)$ (**central** forces). Verlet
  conserves $\mathbf L$ exactly (it is a variational integrator); RK4 does not.
* $\dot E = \sum_i \mathbf v_i\cdot(m_i\dot{\mathbf v}_i + \nabla_i U) = 0$. No fixed-step
  integrator conserves $E$ exactly; symplectic ones keep it bounded (NUMERICS.md §3).

**Potential energy of the system.** Summed once per **pair**:

$$
U=-\sum_{i<j}\frac{G m_i m_j}{r_{ij}} = -\frac12\sum_{i}\sum_{j\ne i}\frac{G m_i m_j}{r_{ij}}.
$$

The original report used the second form (sum everything and divide by 2). The code uses
the first, which needs half the operations.

**Centre of mass.** Since $\mathbf P$ is constant, the CM moves with constant velocity
$\mathbf V_{cm}=\mathbf P/M$. Working in the CM frame ($\mathbf P=0$) is convenient because:
1. the bodies do not "leave" the plot;
2. the kinetic energy splits as $K = \tfrac12 M V_{cm}^2 + K_{\text{internal}}$, and only the
   internal part enters the virial theorem.

## 6. The virial theorem

*Why it matters:* it relates time averages of $K$ and $U$ without solving anything. In
astrophysics it is used to "weigh" clusters and galaxies (Zwicky applied it in 1933 to the
Coma cluster and inferred the existence of dark matter).

**Derivation.** Define the *virial* $\mathcal G=\sum_i \mathbf p_i\cdot\mathbf r_i$.
Differentiating:

$$
\frac{d\mathcal G}{dt}=\sum_i m_i v_i^2+\sum_i \mathbf F_i\cdot\mathbf r_i = 2K+\sum_i \mathbf F_i\cdot\mathbf r_i .
$$

Since $U$ is a homogeneous function of degree $-1$ of the positions
($U(\lambda\mathbf r)=\lambda^{-1}U(\mathbf r)$), Euler's theorem for homogeneous functions
gives $\sum_i \mathbf r_i\cdot\nabla_i U = -U$, i.e. $\sum_i\mathbf F_i\cdot\mathbf r_i = U$.
Averaging over an interval $\tau$:

$$
\frac{\mathcal G(\tau)-\mathcal G(0)}{\tau} = 2\langle K\rangle+\langle U\rangle .
$$

If the motion is **bounded** (bound system, nothing escapes), $\mathcal G$ is bounded and the
left-hand side vanishes as $\tau\to\infty$:

$$
\boxed{\;2\langle K\rangle = -\langle U\rangle\;}\qquad\Longleftrightarrow\qquad Q\equiv\frac{2\langle K\rangle}{-\langle U\rangle}=1 .
$$

This is equation (2) of the original report. Conditions for it to hold in a simulation:

1. Bound system ($E<0$) with no escapers. In the six-body system of the report some bodies
   fly away, and then $Q\neq1$.
2. $K$ measured in the CM frame (otherwise $\tfrac12MV_{cm}^2$ inflates $\langle K\rangle$).
3. A long average (or an integer number of periods for periodic orbits).

Useful consequence: in equilibrium $E=\langle K\rangle+\langle U\rangle=\tfrac12\langle U\rangle=-\langle K\rangle$.

Example 8 shows the virial theorem "in action": a cluster that starts at rest ($Q=0$)
collapses and settles at $Q\approx 1$ (violent relaxation).

## 7. Special solutions of the three-body problem

Although there is no general solution, families of exact solutions are known:

* **Euler (1767)**: the three bodies on a line, rotating rigidly.
* **Lagrange (1772)**: an equilateral triangle in rigid rotation with $\omega^2 = GM/d^3$, for
  **any** masses. Linearizing about it (in the rotating frame) the exponents $\lambda$ satisfy
  $$\lambda^4+\omega^2\lambda^2+\tfrac{27}{4}\beta\,\omega^4=0,\qquad \beta=\frac{m_1m_2+m_2m_3+m_3m_1}{M^2}.$$
  The roots are purely imaginary (stable) only if $27\beta<1$: **Routh's criterion**.
  Sun-Jupiter-Trojan satisfies it (hence the Trojans at L4 and L5); three equal masses do not,
  and the growth rate is $\operatorname{Re}\lambda=\omega/\sqrt2$. Example 5 measures it in
  the simulation and matches theory to 3 digits.
* **Figure-eight (Moore 1993; Chenciner and Montgomery 2000)**: three equal masses run along
  the same ∞-shaped curve. It has $\mathbf L=0$ and is stable. Since 2013 hundreds of new
  periodic orbits have been found (Šuvakov and Dmitrašinović) with numerical searches like
  this one.
* **Pythagorean problem (Burrau 1913)**: masses 3, 4 and 5 at rest on the vertices of a
  3-4-5 triangle. After a chaotic dance with very close encounters, a binary forms and the
  third body is ejected. A very hard test for integrators (see "Ideas for contributions" in
  the README).

## 8. Chaos and Lyapunov exponents

*Motivation:* the original report concluded that long-term predictions are impossible. We
want to **quantify** "long-term".

Let $\boldsymbol\delta(t)$ be the phase-space difference between two solutions that start a
distance $\delta_0$ apart. Linearizing the dynamics,
$\dot{\boldsymbol\delta}=J(t)\,\boldsymbol\delta$ with $J$ the Jacobian matrix. For almost
every initial direction,

$$
\lambda_{\max}=\lim_{t\to\infty}\lim_{\delta_0\to0}\frac1t\ln\frac{|\boldsymbol\delta(t)|}{\delta_0}
$$

exists and is called the **maximal Lyapunov exponent**.

* $\lambda_{\max}=0$: regular motion (trajectories separate at most polynomially, e.g.
  linearly because their periods differ slightly).
* $\lambda_{\max}>0$: **chaos**. An initial error is multiplied by $e$ every *Lyapunov time*
  $t_L=1/\lambda_{\max}$.

**Prediction horizon.** If the initial conditions are known with error $\delta_0$ and we
tolerate an error $\Delta$, we can predict up to $t_{\text{pred}}\approx t_L\ln(\Delta/\delta_0)$.
The dependence is **logarithmic**: measuring a thousand times better only adds ~7 Lyapunov
times. For the Solar System $t_L\approx5$ million years (Laskar, 1989), so the Earth's orbit
cannot be predicted precisely beyond a few tens of millions of years.

**Benettin's method** (implemented in `nbody/chaos.py`): a "shadow" trajectory is
integrated a distance $\delta_0$ away; every so often its separation $d_k$ is measured,
$\ln(d_k/\delta_0)$ is accumulated and the shadow is rescaled back to distance $\delta_0$
along the same direction. The separation never saturates and
$\lambda\approx\frac1t\sum_k\ln(d_k/\delta_0)$.

**Physical chaos vs. numerical error.** In a chaotic system the truncation error also grows
exponentially: two runs with $h$ and $h/2$ diverge (example 3). Is the simulation useful,
then? Yes, in a statistical sense: the *shadowing lemma* guarantees (under some hypotheses)
that the numerical trajectory stays close to **some** exact trajectory with slightly
different initial conditions. Conserved quantities and statistical properties are reliable
even when the specific trajectory is not.

**KAM theory** (Kolmogorov-Arnold-Moser): when an integrable system is slightly perturbed,
most regular orbits survive (on invariant tori). That is why the figure-eight orbit and the
Trojans are stable: they are surrounded by KAM tori. Chaos appears near resonances and when
the perturbation is large.

## 9. Gravitational softening

When two bodies approach, $1/r^2\to\infty$ and a fixed-step integrator produces an artificial
"kick". In cluster and galaxy simulations (where particles stand for many stars and the
collective field is what matters) one replaces

$$
\frac{1}{r}\;\longrightarrow\;\frac{1}{\sqrt{r^2+\varepsilon^2}}, \qquad
\mathbf a_i = G\sum_{j\ne i} m_j\frac{\mathbf r_j-\mathbf r_i}{\left(|\mathbf r_j-\mathbf r_i|^2+\varepsilon^2\right)^{3/2}} .
$$

This is equivalent to replacing every point mass by a Plummer sphere of radius
$\varepsilon$. The force is still $-\nabla U$ of the softened potential (the test
`test_softened_force_is_consistent_with_softened_potential` checks it), so the softened
energy is conserved. Planetary systems use $\varepsilon=0$.

## 10. N-body units

The equations depend only on the combination $GM$. Rescaling lengths by $L$ and masses by
$M$, the natural time is $T=\sqrt{L^3/(GM)}$ (Kepler's third law). Choosing $G=M=1$ and $L$
such that $E=-1/4$ (Hénon units, standard in stellar dynamics), all numbers are of order 1,
which improves readability and avoids working with $10^{30}\cdot10^{-11}$. Results are
converted to any physical scale at the end.

## 11. References

* Aarseth, S. J. (2003). *Gravitational N-Body Simulations*. Cambridge University Press.
* Benettin, G., Galgani, L., Giorgilli, A., Strelcyn, J.-M. (1980). *Meccanica* 15, 9–30.
* Binney, J., Tremaine, S. (2008). *Galactic Dynamics*, 2nd ed. Princeton.
* Chenciner, A., Montgomery, R. (2000). *Annals of Mathematics* 152, 881–901.
* Goldstein, H., Poole, C., Safko, J. *Classical Mechanics*, 3rd ed.
* Laskar, J. (1989). *Nature* 338, 237–238.
* Lynden-Bell, D. (1967). *MNRAS* 136, 101.
* Moore, C. (1993). *Physical Review Letters* 70, 3675.
* Šuvakov, M., Dmitrašinović, V. (2013). *Physical Review Letters* 110, 114301.
* Szebehely, V., Peters, C. F. (1967). *Astronomical Journal* 72, 876.
