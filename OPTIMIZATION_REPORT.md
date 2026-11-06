# Simulation speed: measurements and optimization options

## Second round: bit-identical optimizations

Rule for this round: the results of the four example cases must not change at all. Every change was
checked bitwise against the `master` code (a copy from git):
- table lookups on 4000 × 164 values,
- the refrigerant right-hand side and the full 172 × 172 Jacobian,
- 300 frost-solver steps on six segments with different iteration counts, sublimation and a step change,
- Case 1 and a 7.5-min frost run (CSV and all grid snapshots).

At the end, all four cases were rerun at full length: **every `results_case_N.csv` is byte-identical to the
committed file.**

| Change | Where | Why it is exact |
|---|---|---|
| One fused lookup of ρ, ∂ρ/∂p, ∂ρ/∂h, T that calls the compiled helpers of `RegularGridInterpolator` (`find_indices`, `evaluate_linear_2d`) directly; replaces four lookup functions | `refrigerant_modell_V2._lookup` | Same compiled routines, without the per-call Python overhead; falls back to the interpolator for NaN input or another SciPy version |
| Right-hand side for many states at once, BDF with `vectorized=True`: the 141 Jacobian columns take one call instead of 141. Evaporator and condenser share one batched balance builder (`flow_system`) | `refrigerant_modell_V2.update_all_segments` | Same expressions and operation order; row sums along contiguous axes and batched `gesv` checked to equal the 1-D versions bitwise |
| Compressor/valve evaluated once per distinct input in a batch | same | Same function, same inputs |
| `PropsSI` (≈450 µs per call) replaced by the existing `AbstractState` (≈12 µs) for T, quality and T_sat in the time loop | `Refrigerant.props_PH`, `T_sat` | Bitwise equal on 3000 states incl. two-phase |
| Frost of a whole row solved at once: one shared Picard iteration `_picard_rows` for the edge and fin/tube models. Each segment keeps its own relaxation factor and convergence; loop invariants computed once; LAPACK `gtsv` called directly | `Frostmodell_V1.py`, `simulator_V2.py` (row phases: frost condition → row solves → air) | Same per-segment iteration; `exp`/`pow` of the saturation formula checked equal for all array lengths |
| Surface fluxes computed once per segment (`segment_flows`) instead of twice | `Frostmodell_V1.py` | Same formulas |

**Run times** (4-core cloud container, all four cases running at the same time, `OPENBLAS_NUM_THREADS=1`):

| Case | Thesis code (your PC) | After round 1 | **After round 2** |
|---|---|---|---|
| 1, 10 min | 6.5 min | 2.5 min | **0.3 min** |
| 2, 60 min | 570.5 min | 31.4 min | **3.3 min** |
| 3, 60 min | 700.7 min | 38.6 min | **4.0 min** |
| 4, 90 min | – | 73.4 min | **8.5 min** |

Same machine and load (two runs at a time), `master` vs. this round: Case 1 107 s → 17 s, 7.5-min frost
run 158 s → 26 s (≈6×). Per frost step: ~0.56 s → ~0.07 s.

**Code size:** `Framework/` 4901 → 4098 lines (−803). `Frostmodell_V1.py` 901 → 331,
`refrigerant_modell_V2.py` 744 → 587, `simulator_V2.py` 576 → 500. Unused code was removed: the
all-angles edge solver, the `q_dot_*` helpers, duplicate air-enthalpy helpers, commented-out blocks and
four lookup functions. The per-segment entry points `New_edge_state_seg_at_90` and
`New_finn_and_tube_state_seg` remain as one-segment wrappers.

**Where the time goes now** (40 s of frost, 6.8 s in total, `master`: 52 s): 74 % refrigerant solver,
mainly the batched LAPACK solves of the 91 × 91 condenser systems for the Jacobian; 14 % frost; the rest
is air, recording and the console. Further speed-ups would have to change the rounding: block-solving
the condenser (its wall and water rows are diagonal), a sparse or analytic Jacobian, or looser solver
tolerances. They are not compatible with bit-identical results.

**Reproducibility note:** bitwise reproduction requires the same BLAS thread setting as the reference
run (`OPENBLAS_NUM_THREADS=1` for the committed results), because multi-threaded BLAS changes the rounding
in the linear solves.

---

## 0. Status: implemented optimizations (first round)

O1, O2, O3 and O8 below are implemented. O5 was tested and not adopted. Sections 1–3 are the original
analysis; their numbers refer to the code before these changes.

| # | Change | Where |
|---|---|---|
| O1 | Segments run one after another instead of in a `ThreadPoolExecutor` | `runtime/simulator_V2.py` |
| O2 | Adaptive under-relaxation: start with `OMEGA_T_START = 0.8`, halve whenever the residual grows, never below the previous fixed 0.1. Same tolerance, so the converged results are unchanged | `models/Frostmodell_V1.py` (fin/tube and 90° edge solver) |
| O3 | Tridiagonal `solve_banded` instead of CSR + `spsolve`; vectorised saturation humidity (`p_ws_Pa`, the IAPWS equations CoolProp uses: identical below 0 °C, < 1e-4 above) instead of one CoolProp call per node | `models/Frostmodell_V1.py` |
| O8 | CoolProp grids of the property tables cached in `~/.cache/mse_vt1_framework` (keyed by fluid, ranges and CoolProp version; written atomically) | `models/derivatives_of_rho.py`, `refrigerant_modell_V2.py` |

Bug fixed on the way: `drho_dP_dH(load_path=...)` failed because `T_interp` was never defined in that
branch.

**Measured effect**

| Measurement | Before | After |
|---|---|---|
| Fin/tube + edge frost solve per segment and step (150 s of frost growth incl. a jump in boundary conditions) | 54.8 ms, ~70 iterations | **1.8 ms**, ~7 iterations, identical results (6 significant digits) |
| 40 s with frost on all 40 segments | 579 s (thread pool), 226 s (serial) | **52 s** |
| 7.5-min frost reference run (humidity step at 5 min, temperature ramp at 6 min) | 68 min | **6 min**; all quantities within 0.3 %, frost thickness within 0.04 % |
| `Simulator()` start-up | 13.7 s | **0.03 s** (from the second run on) |
| Case 1 (no frost) | – | identical results (deviation 0.000 %) |

**Example cases, full length** (before: `Simulation_time.png` of the original runs on your machine; after:
this branch in the 4-core cloud container, with Cases 2 and 3 running in parallel to other simulations):

| Case | Simulated | Before | After | Speed-up |
|---|---|---|---|---|
| 1 (F0), no frost | 10 min | 6.5 min | 2.5 min | ~2.6× (start-up cache; no frost to speed up) |
| 2 (F1), frost | 60 min | 570.5 min | **31.4 min** | **~18×** |
| 3 (F2), frost + temperature ramp | 60 min | 700.7 min | **38.6 min** | **~18×** |
| 4 (F3), new, 90 min of changing conditions | 90 min | – | **73.4 min** | – (not run before) |

The machines differ, so these ratios are indicative. The same-machine comparisons above (frost solver
×30, frost reference run ×11) are the precise numbers.

**O5 (larger `dt_max`) was not adopted.** Against the 7.5-min frost reference:

| dt_max | 0.5 s | 1 s | 2 s |
|---|---|---|---|
| Steps | 1221 | 825 | 630 |
| Max frost thickness at the end | 0.02 % | 5.6 % | 7.0 % |
| Q_evap, max deviation | 0.26 % | 2.6 % | 1.9 % |

The frost thickness deviates by more than the 1 % limit, so the default stays at 0.5 s. `GridShape.dt_max`
can still be raised for quick studies.

**Where the time goes now** (profile of the 40 s frost run, 52 s in total):

- **78 %: refrigerant solver** (`update_all_segments`). About 70 % of its time is the numerical Jacobian of
  the BDF solver: ~1.5 Jacobians per time step, each needing one RHS evaluation per state (141 states),
  so ~295 RHS evaluations per step.
  - ~28 % of the total runtime is `RegularGridInterpolator` for the property tables (385 000 calls at
    ~38 µs, mostly argument checking).
  - ~13 % is the compressor model (two HEOS state updates per RHS evaluation).
- **19 %: frost and air** for all 40 segments (was > 95 % before).

**Next options, not implemented:**
- **Analytic or vectorised Jacobian:** Jacobian columns in one vectorised RHS call (`vectorized=True`)
  or partly analytic. Largest remaining gain (×3–5 on the refrigerant part), but a significant change.
- **One fused bilinear lookup** of ρ, ∂ρ/∂p, ∂ρ/∂h and T instead of four `RegularGridInterpolator` calls
  (measured 1.4–2× per call; about −20 % total).
- **Solver tolerances** (`rtol=1e-6`, `atol=1e-8` on states of order 1e5–1e6): looser values mean fewer
  steps and Jacobians. Needs the same validation as O5.
- O4, O6, O7 from section 2 still apply. They now only affect the remaining ~19 %.
- When running several simulations in parallel, set `OPENBLAS_NUM_THREADS=1`. Otherwise every process
  starts one BLAS thread per core and they slow each other down.

---

*Original analysis (before the optimizations):*

This report lists where the time goes and what would make the simulation significantly faster, with
measured numbers where possible. All measurements are on the fixed code of this branch, in a 4-core
cloud container (Python 3.13, pinned `requirements.txt`). Some runs shared the CPU with other
simulations, so treat absolute times as indicative. The ratios (A vs. B under the same conditions) are
what matter.

## 1. Where the time goes

| Part | Measured cost |
|---|---|
| Time step **without** frost (Case 1, 1210 steps) | ~0.12 s/step (2.5 min for 10 min of simulated time) |
| Time step **with** frost (example geometry: 40 segments, 20 of them with an edge model) | ~2.4 s/step serial, ~6 s/step with the current thread pool |
| one `New_finn_and_tube_state_seg` call | 48 ms, ~70 Picard iterations (0.7 ms/iteration) |
| one `New_edge_state_seg_at_90` call | 25 ms, ~58 iterations |
| refrigerant update (BDF) | 26 ms/step |
| `Simulator()` start-up (property tables, 2 × 240 000 `PropsSI` calls) | 13 s per simulator |
| one grid snapshot (every 100 steps) | 0.5 s and 19.7 MB of JSON |

The frost steps dominate. 40 × 48 ms + 20 × 25 ms ≈ 2.4 s, which is the whole step. In a 60-min case,
frost is active from minute 5, so roughly 6500 of the ~7400 steps of the committed Case 2 run are frost
steps. That is roughly 4–12 h of run time.

Grid size barely matters at the moment. The fin/tube solve takes 30 ms at `nx = 20`, 42 ms at 50 and 48 ms
at 100. The cost is the number of iterations times the fixed overhead per iteration (Python, building the
sparse matrix, CoolProp calls).

## 2. Options, ordered by expected gain per effort

### O1: Drop the thread pool, or use processes (×2.5 measured, trivial)

40 s simulated with frost from the start:

| Variant | Run time | Steps |
|---|---|---|
| current `ThreadPoolExecutor` | 579 s | 94 |
| same code, segments run one after another | **226 s** | 94 |

The 7.5-min Case 3 verification run (threaded) needed 68 min of wall time, with 50 min of *system* time
against 44 min of user time. That much kernel time is a typical sign of threads contending for the GIL.
The frost solvers are mostly Python and run under the GIL, so threads cannot run them in parallel.
Instead they add contention and context switches, plus a new pool for every row in every step. The
"ct" values printed per segment are inflated by threads waiting for each other.

**Change:** Run the segments of a row in a plain loop. To use several cores, use a
`ProcessPoolExecutor` (one persistent pool, segments as pickled state), or better, do O4.

### O2: Fewer Picard iterations in the frost solvers (×4–20 on the frost solve, small change)

Both solvers under-relax the temperature with `omega_T = 0.1`, which needs ~70 iterations per call.
Same state, same converged result (s_ft identical to 4 digits):

| omega_T | 0.1 | 0.3 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| iterations | 70 | 22 | 12 | 6 | 3 |
| time per call | 46 ms | 20 ms | 14 ms | 5 ms | 2 ms |

The comment in the code ("save stabil: 0.1") suggests that larger values were unstable in some
situation, probably at frost onset or with a large dt. Options, in increasing robustness:
- **Adaptive relaxation:** start at 0.8 and halve whenever the residual grows (a few lines).
- **Aitken Δ² acceleration** of the Picard iteration.
- **Newton on the coupled T–w system:** the source terms are smooth (`w_sat(T)` has an analytic
  derivative), so this converges in 2–4 iterations.

This needs a validation run at frost onset and with the largest dt.

### O3: Cheaper iterations (×2 per iteration, small change)

- **Tridiagonal solver:** each iteration builds two CSR matrices and calls `spsolve` twice. For N = 100,
  build plus solve takes **139 µs** with `csr_matrix` + `spsolve` and **20 µs** with
  `scipy.linalg.solve_banded`. A hand-written Thomas algorithm or a batched one (see O4) is cheaper still.
  The matrices are already assembled as diagonals, so only the last lines change.
- **Vectorised saturation humidity:** `w_sat_coolprop` loops over all nodes with one CoolProp call each
  (54 µs for 98 nodes). A NumPy formula takes 6 µs and agrees with CoolProp to 0.015 % between −30 and
  0 °C (Sonntag 1990 over ice, `p_ws = 100·exp(−6024.5282/T + 24.7219 + 1.0613868e−2·T −
  1.3198825e−5·T² − 0.49382577·ln T)` in Pa). The same goes for the `HAPropsSI` frost-condition check per
  segment and step in the simulator.

### O4: Solve a whole row of segments at once (×5–10 on top of O2/O3, medium effort)

After O2 and O3, the Python overhead per segment call dominates. The 20 segments of a row are independent
within a time step (row `ix` only needs the air leaving row `ix − 1`). Stacking their profiles into arrays
of shape `(n_seg_r, nx)` lets each Picard/Newton iteration be a single vectorised operation, with a
batched tridiagonal solve, for all segments. Per-segment control flow (frost condition, sign of `dw`)
becomes masks. This replaces 40–60 Python calls per step with about 2–3 vectorised calls and makes threads
unnecessary.

### O5: Larger time steps (×3.5 measured without frost, needs validation with frost)

`dt_max = 0.5 s`, while frost evolves over minutes and the refrigerant side uses an implicit BDF solver.
Case 1 with `dt_max = 2 s` (now a `GridShape` parameter, no code change needed):

| | dt_max = 0.5 s | dt_max = 2 s |
|---|---|---|
| steps for 10 min | 1210 | **350** |
| end values | – | identical within 0.1 % (Q_evap 7850.3 vs 7850.4 W, p_cond 13.348 vs 13.348 bar) |
| max. deviation after 60 s | – | Q_evap 5 W, superheat 0.04 K, p_cond 0.005 bar |

Frost thickness and air are advanced explicitly with the same dt, so check before relying on it: compare
`mean_s_ft`, `Q_evap` and the outlet air for a frost case at dt_max = 0.5 / 1 / 2 / 5 s. The dt reduction
around the profile ramps is already in place. An error-based step control (e.g. limit the change of
`s_ft` or `T_tube` per step) would let dt grow during slow frost growth and shrink automatically during
transients.

### O6: Edge model only where it matters (−20 % of frost time, small change)

The edge solve (20 × 25 ms per step, about 20 % of the frost time) only feeds the "frost blocks the
channel" stop criterion. `s_e`, `T_e` and `w_e` do not affect heat flows, air or refrigerant. Options:
compute it every N steps with `N·dt` in the growth update, compute it only once the fin/tube frost exceeds
some fraction of `s_max`, or make it optional (`gs.cal_edge`).

### O7: Coarser frost grid (×2–3 after O2/O3/O4, needs a grid study)

`nx = nr = 100` nodes for a frost layer that stays below ~2 mm. Once the per-iteration overhead is gone
(O3/O4), the cost scales with N. A short grid study (s_ft, Q at nx = 20/30/50/100) will most likely show
that 20–30 nodes are enough.

### O8: Start-up and output (fixed costs)

- **Property tables:** 13 s per `Simulator()` (about 480 000 `PropsSI` calls). `drho_dP_dH` already has
  `save_path`/`load_path`. Cache the tables in a file keyed by fluid and range, and load them in <1 s.
- **Snapshots:** 19.7 MB of JSON each (`T_e`, `w_e`, `rho_e`, `rho_a_e` are 100 × 90 per segment, even
  for rows where the edge model never runs), deep-copied and also kept in RAM. A 60-min run writes about
  1.5 GB. Store the edge fields only for `ix = 0`, use `float32`/`.npz` instead of JSON text, and do not
  keep the snapshots in RAM when streaming. Less I/O; mainly a memory and disk fix.
- **Console output:** every 20th step prints one line per segment, and every step prints several lines.
  In Jupyter, megabytes of cell output slow the browser down noticeably. Use a single status line or the
  `logging` module with a level.
- Refrigerant write-back: 80 `PropsSI` calls per step can use the existing `T_from_PH` table (small, part
  of the 26 ms).

## 3. Expected overall effect

The gains multiply, but only approximately, because each option shrinks a part that others also touch.
A realistic path:

| Step | Effort | Frost step (serial) |
|---|---|---|
| today (thread pool) | – | ~6 s |
| O1 serial loop | minutes | ~2.4 s |
| + O2 adaptive relaxation / Newton | hours | ~0.3–0.5 s |
| + O3 banded solver, vectorised w_sat | hours | ~0.15–0.25 s |
| + O6 edge model only when needed | hour | ~0.12–0.2 s |
| + O4 row-vectorised solver | 1–2 days | ~0.1 s (then the ~0.1 s/step outside the frost solvers dominates: air and CoolProp calls per segment, refrigerant, printing) |
| + O5 dt_max 2 s (if validated) | minutes | 3–4× fewer steps |

For a 60-min frost case (Case 2/3), that is from **many hours** today to roughly **10–30 minutes** with
O1–O3 + O5, and to a few minutes with O4. Recommended order: O1 → O2 → O3 → O5 (each validated against the
current results), then O4 if the simulation will be run often or for parameter studies.
