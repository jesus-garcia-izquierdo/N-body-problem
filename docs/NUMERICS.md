# Numerical methods: from the differential equation to the code

**Contents**

1. [From an ODE to a discrete map](#1-from-an-ode-to-a-discrete-map)
2. [Euler: why it fails for orbits](#2-euler-why-it-fails-for-orbits)
3. [Symplectic integrators and the shadow Hamiltonian](#3-symplectic-integrators-and-the-shadow-hamiltonian)
4. [Velocity Verlet: three derivations](#4-velocity-verlet-three-derivations)
5. [Higher order without losing structure: Yoshida](#5-higher-order-without-losing-structure-yoshida)
6. [Runge-Kutta 4: accurate but not symplectic](#6-runge-kutta-4-accurate-but-not-symplectic)
7. [Choosing the time step h](#7-choosing-the-time-step-h)
8. [Computational cost and vectorization](#8-computational-cost-and-vectorization)
9. [Round-off errors](#9-round-off-errors)
10. [Rendering the videos](#10-rendering-the-videos)
11. [Beyond this repository](#11-beyond-this-repository)

---

## 1. From an ODE to a discrete map

We write the system as $\dot{\mathbf y}=\mathbf f(\mathbf y)$ with
$\mathbf y=(\mathbf r,\mathbf v)$ and $\mathbf f=(\mathbf v,\mathbf a(\mathbf r))$. An
integrator is a rule $\mathbf y_{n+1}=\Phi_h(\mathbf y_n)$ that approximates the exact flow
$\varphi_h$ (the solution after a time $h$).

* **Local error**: $\|\Phi_h(\mathbf y)-\varphi_h(\mathbf y)\|=\mathcal O(h^{p+1})$.
* **Global error** after $n=t/h$ steps: the local errors add up, $n\cdot h^{p+1}\sim t\,h^{p}$.
  The method is of **order $p$**.

Halving $h$ reduces the global error by $2^p$. Example 6 and the test
`test_convergence_orders` check this for the five methods.

But the order only describes what happens as $h\to0$ at fixed $t$. For orbits we want
$t\to\infty$ at fixed $h$, and there what matters is **structure**.

## 2. Euler: why it fails for orbits

Explicit Euler: $\mathbf r_{n+1}=\mathbf r_n+h\mathbf v_n$, $\mathbf v_{n+1}=\mathbf v_n+h\mathbf a_n$.

Take the simplest model of an orbit, the harmonic oscillator $\ddot x=-\omega^2x$ (a circular
orbit projected on one axis). The step is linear:

$$
\begin{pmatrix}x_{n+1}\\ v_{n+1}\end{pmatrix}=
\underbrace{\begin{pmatrix}1 & h\\ -\omega^2h & 1\end{pmatrix}}_{A_{\text{Euler}}}
\begin{pmatrix}x_{n}\\ v_{n}\end{pmatrix},
\qquad \det A_{\text{Euler}}=1+\omega^2h^2>1 .
$$

The determinant is the factor by which **areas** in phase space change at each step. Being
larger than 1, the orbit spirals outwards and the energy $E\propto v^2+\omega^2x^2$ grows by
$(1+\omega^2h^2)$ **every step**: exponential growth, without bound. This is what example 6
shows.

**Symplectic Euler** only changes the order: first the velocity, then the position with the
*new* velocity:

$$
A_{\text{symp}}=\begin{pmatrix}1-\omega^2h^2 & h\\ -\omega^2h & 1\end{pmatrix},\qquad \det A_{\text{symp}}=1 .
$$

It preserves area exactly. Still first order, but the energy no longer drifts: it oscillates
around the correct value. Moral: **structure matters more than order**.

## 3. Symplectic integrators and the shadow Hamiltonian

*Motivated definition.* The exact flow of a Hamiltonian system preserves the symplectic form
(in 1D: area in the $(x,p)$ plane; see THEORY.md §4). A **symplectic** integrator is one whose
map $\Phi_h$ preserves it too: mathematically, its Jacobian $M=\partial\Phi_h/\partial\mathbf y$
satisfies $M^{\mathsf T}JM=J$, with $J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}$. In 1D this is
exactly $\det M=1$.

*Why that gives bounded energy.* A deep result (backward error analysis) states that a
symplectic integrator of order $p$ solves **exactly** (up to exponentially small terms) the
dynamics of a modified or **shadow** Hamiltonian

$$
\tilde H = H + h^{p}H_{p}+h^{p+1}H_{p+1}+\dots
$$

Since the numerical trajectory conserves $\tilde H$, and $\tilde H$ differs from $H$ by
$\mathcal O(h^p)$, the error in $H$ stays **bounded** by $\mathcal O(h^p)$ for times
exponentially long in $1/h$ (Benettin and Giorgilli 1994; Hairer, Lubich and Wanner 2006).
That is what every energy-error plot in the examples shows: it oscillates but does not grow.

A visible effect of the shadow Hamiltonian: in example 5 the "stable" Lagrange triangle shows
a periodic deformation of ~1e-6. It is not an instability: the circular orbit of $\tilde H$
has a slightly different radius from that of $H$.

## 4. Velocity Verlet: three derivations

The method of the original report (equations 3-5):

$$
\begin{aligned}
\mathbf r_{n+1} &= \mathbf r_n + h\,\mathbf v_n + \tfrac{h^2}{2}\mathbf a_n,\\
\mathbf a_{n+1} &= \mathbf a(\mathbf r_{n+1}),\\
\mathbf v_{n+1} &= \mathbf v_n + \tfrac{h}{2}\left(\mathbf a_n+\mathbf a_{n+1}\right).
\end{aligned}
$$

**(a) Taylor.** The position is the Taylor expansion up to $h^2$. For the velocity,
$\mathbf v(t+h)-\mathbf v(t)=\int_t^{t+h}\mathbf a\,dt'$, and that integral is approximated by
the trapezoidal rule (error $\mathcal O(h^3)$).

**(b) Leapfrog (kick-drift-kick).** Introducing the half-step velocity
$\mathbf v_{n+1/2}=\mathbf v_n+\frac h2\mathbf a_n$, the same algorithm reads

1. *kick*: $\mathbf v_{n+1/2}=\mathbf v_n+\frac h2\mathbf a(\mathbf r_n)$
2. *drift*: $\mathbf r_{n+1}=\mathbf r_n+h\,\mathbf v_{n+1/2}$
3. *kick*: $\mathbf v_{n+1}=\mathbf v_{n+1/2}+\frac h2\mathbf a(\mathbf r_{n+1})$

**(c) Splitting the Hamiltonian** (the one that explains all its properties). With
$H=T(\mathbf p)+V(\mathbf q)$, each part on its own is integrated **exactly**:
* the flow of $T$ is uniform straight-line motion (*drift*);
* the flow of $V$ only changes the momenta with frozen forces (*kick*).

Both are exact Hamiltonian flows, hence symplectic; so is their composition. The
Baker-Campbell-Hausdorff formula states that the **symmetric** composition
$e^{\frac h2 V}e^{hT}e^{\frac h2V}$ approximates $e^{h(T+V)}$ with an $\mathcal O(h^3)$ error
per step. Consequences:

| Property | Why |
|---|---|
| Second order | local error $\mathcal O(h^3)$ (symmetric BCH) |
| Symplectic | composition of exact Hamiltonian flows |
| Time-reversible | the composition is symmetric: $\Phi_{-h}=\Phi_h^{-1}$ |
| Conserves $\mathbf P$ and $\mathbf L$ exactly | every sub-step does (central pair forces) |
| 1 force evaluation per step | $\mathbf a_{n+1}$ is reused as the $\mathbf a_n$ of the next step |

The last row is an optimization over the original MATLAB code, which computed the
acceleration twice per step.

## 5. Higher order without losing structure: Yoshida

Can we have order 4 and stay symplectic? Yes, by composing Verlet steps. Since Verlet is
symmetric, its error contains only **odd** powers of $h$:
$\Phi_h=\varphi_h+h^3C+\mathcal O(h^5)$. We compose three steps with weights $w_1,w_0,w_1$:

$$
\Psi_h=\Phi_{w_1h}\circ\Phi_{w_0h}\circ\Phi_{w_1h}.
$$

To advance $h$ in total: $2w_1+w_0=1$. To cancel the $h^3$ error: $2w_1^3+w_0^3=0$. The
solution (Yoshida 1990) is

$$
w_1=\frac{1}{2-2^{1/3}}\approx1.3512,\qquad w_0=-\frac{2^{1/3}}{2-2^{1/3}}\approx-1.7024 .
$$

$w_0<0$: the middle step goes **backwards** in time! The result is fourth order, symplectic
and reversible, with 3 force evaluations per step. In example 6 it is the clear winner at
equal cost.

## 6. Runge-Kutta 4: accurate but not symplectic

RK4 evaluates the derivative at four points and combines them with weights 1/6, 2/6, 2/6, 1/6
to cancel the errors up to $h^4$. It is excellent for general ODEs and in the short term it
is more accurate than Verlet. But it is not symplectic: the energy error has a
**systematic** component that accumulates linearly in time. For $10^6$ orbits (planetary
dynamics) it ends up losing to a lower-order symplectic method. RK4 also does not conserve
$\mathbf L$ exactly (only linear invariants such as $\mathbf P$).

## 7. Choosing the time step h

The step is limited by the **shortest dynamical time** of the system,
$t_{\text{dyn}}\sim\sqrt{r^3/(GM)}\sim r/v$:

* Sun-Earth-Moon: the Moon (27 days) → $h=1000$ s ≈ $T_{\text{Moon}}/2400$.
* Solar System: Mercury (88 days) → $h=0.5$ days.
* Figure-eight: $T\approx6.3$ → $h=10^{-3}$.

Rule of thumb for Verlet: $h\lesssim t_{\text{dyn}}/100$, then check that the energy error is
acceptable. **Always** repeat with $h/2$: if the results you care about change, $h$ was too
large.

**Close encounters.** When two bodies approach, $t_{\text{dyn}}$ collapses and a fixed step
is no longer valid. Options:
1. Softening $\varepsilon$ (THEORY.md §9) if the encounters are not physically relevant
   (clusters).
2. Adaptive step $h\propto\sqrt{r_{\min}^3/(GM)}$. Careful: changing $h$ naively breaks
   symplecticity (the energy drifts again). Techniques exist to avoid it: Sundman time
   transformations, block time steps, or very high-order integrators with error control
   such as IAS15.
3. Regularization (Kustaanheimo-Stiefel): a change of variables that removes the $1/r^2$
   singularity in binary encounters.

## 8. Computational cost and vectorization

Evaluating forces by direct summation costs $\mathcal O(N^2)$ operations (all pairs). In
Python a double `for` loop over bodies is very slow because Python interprets every
iteration. The solution is to express the computation with whole-array operations that NumPy
runs in C:

```python
dr = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]   # (N, N, 3): r_j - r_i
r2 = np.einsum("ijk,ijk->ij", dr, dr)                             # (N, N):    |r_j - r_i|^2
np.fill_diagonal(r2, np.inf)                                      # no self-interaction
acc = G * np.einsum("ij,ijk->ik", masses[None, :] * r2**-1.5, dr) # (N, 3):    sum over j
```

Timings measured in this repository (one CPU core, NumPy):

| System | N | steps | time |
|---|---|---|---|
| Sun-Earth-Moon, 2 years | 3 | 63 000 | ~1 s |
| Solar System, 165 years | 9 | 120 000 | ~2 s |
| Cold collapse | 400 | 4 000 | ~25 s |

For small $N$ the fixed cost of every NumPy call (~15 µs) dominates; for large $N$, the memory
of the $(N,N,3)$ array (600 MB for $N=5000$). Natural next steps to scale up:

* **Numba / JAX / C++**: compile the whole integration loop.
* **Barnes-Hut** (octree, $\mathcal O(N\log N)$): groups distant bodies into their centre of mass.
* **Fast multipole methods** ($\mathcal O(N)$) and **particle-mesh** with FFTs for cosmological
  simulations.
* **GPUs**: direct summation is perfectly parallel.

## 9. Round-off errors

With `float64` every operation introduces a relative error $\epsilon\approx10^{-16}$. In the
plots, $\mathbf P$ and $\mathbf L$ (which Verlet conserves exactly in exact arithmetic) show
errors of $10^{-15}$–$10^{-12}$ that grow slowly: the accumulation of round-off, typically
like a random walk $\propto\sqrt n\,\epsilon$, sometimes with a small systematic trend. It can
be reduced with compensated (Kahan) summation or by integrating in coordinates relative to a
central body (heliocentric or Jacobi), as professional codes do.

For the same reason the code uses $t_n=n\,h$ instead of accumulating $t\mathrel{+}=h$: the
latter adds up $10^5$ round-off errors.

## 10. Rendering the videos

`nbody/video.py` draws each frame with Matplotlib's 3D axes and pipes it to **ffmpeg**
(H.264, `yuv420p`, CRF 18 ≈ visually lossless, `+faststart` so the file streams on the web):

* frames are evenly spaced in time; the **trail** of each body uses every stored sample
  between the current frame and `trail` time units earlier, with an alpha ramp that fades
  towards the tail;
* bodies are a bright core plus a faint halo, with area ∝ mass^(1/3) (spheres of equal
  density);
* the camera **rotates** slowly: motion parallax is what lets the eye read depth in a 3D plot;
* all panels of a comparison video share the same limits and camera, so differences are real;
* the box has the **same scale on x, y and z**; flat systems can use `z_scale` to exaggerate
  inclinations (always stated in the subtitle).

## 11. Beyond this repository

* **REBOUND** (Rein and Liu 2012): the reference code for planetary dynamics, with WHFast
  (symplectic Wisdom-Holman) and IAS15 (adaptive, 15th order).
* **Hairer, Lubich, Wanner**, *Geometric Numerical Integration* (Springer, 2006): the reference
  on symplectic integrators.
* **Yoshida, H.** (1990). *Physics Letters A* 150, 262–268.
* **Verlet, L.** (1967). *Physical Review* 159, 98.
* **Swope, W. C. et al.** (1982). *J. Chem. Phys.* 76, 637 (velocity Verlet).
