# Simulation speed: measurements and optimization options

Nothing in this report is implemented. It lists where the time goes and what would make the
simulation significantly faster, with measured numbers where possible. All measurements are on the
fixed code of this branch, in a 4-core cloud container (Python 3.13, pinned `requirements.txt`). Some
runs shared the CPU with other simulations, so treat absolute times as indicative. The ratios
(A vs. B under the same conditions) are what matter.

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
