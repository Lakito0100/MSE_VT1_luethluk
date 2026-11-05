# Simulation test & bug report

Scope: `Framework/` and `Examples/` (Legacy was not reviewed). Line numbers in sections 1–4 refer to
commit `c6fef2e`, the state before the fixes. Speed-up options are in `OPTIMIZATION_REPORT.md`.

## 0. Status: fixes applied on this branch

All bugs below are fixed on `claude/confident-wright-1pbmjz`, except where the table says otherwise.
Testing found three more problems (20–22), which are fixed as well.

| # | What was changed |
|---|---|
| 1 | `refrigerant_modell_V2.py`: `fill_vals = -delta_h ...` in the condenser and evaporator matrices. |
| 2 | `compressor_model` reuses one `AbstractState("HEOS")` created in `__init__`. No more per-call TTSE states. |
| 3 | `valve_model` and `compressor_model` use the scalar `rho_and_derivs`. `recorder.read_results_csv_json` also reads old CSVs with `[x]` cells without deprecated NumPy behaviour. |
| 4 | `air_modell.propagate_inplace`: the limiter uses `min` for cooling and `max` for heating. |
| 5 | Kays & London: `A_fr = n_fin·fin_pitch_cc·h_fin·n_seg_r·stacks`, `A_min = n_fin·fin_gap·(h_fin−d)·n_seg_r·stacks` (σ = 0.73 instead of being clipped to 1). Per your decision `dp_clean` stays unused and is marked as such in `config.py`. |
| 6 | After the fan solve, `input_cfg.v_a = m_dot / (rho_amb · A_free)`, with the same free-flow area as `propagate_inplace` (new `Air.free_flow_area_seg`). |
| 7 | New `GridShape` fields: `T_a_start/T_a_end/t_switch_T/ramp_T`, `RH_start/RH_end/t_switch_RH/ramp_RH`, `dt_window_switch`, `dt_min/dt_max/it_target`. The start values default to `cfg.T_a`/`cfg.RH`, so the examples behave as before. `RH` and `rho_amb` are updated together with `w_amb`/`T_a`, the profile is evaluated at `t + dt` (no one-step lag), and the dt reduction follows the configured switch times. |
| 8 | The valve controller runs once per time step before the integration and stays constant inside `rhs`. `compressor_model` is pure. `W_comp`, `EER` and `COP` are computed from the state at time t. `EER`/`COP` are NaN while the compressor is off. |
| 9 | All example `.py` and `.ipynb` files: `n_fin` added to the backup `m_dot`. |
| 10 | The SH filter uses `_valve_SH_filt` everywhere. |
| 11 | Default `path_variant="serpentine"`. Exact string comparison and a correct error message in `build_connection_path`. |
| 12 | `st.s_e[gs.ntheta - 1]` (tested with `ntheta = 60`). |
| 13 | `x_ref = NaN` for single phase. |
| 14 | `FinnTubedHX.l_tube()` returns `fin_pitch_cc`, so the segment tube length is `n_fin·fin_pitch_cc`. |
| 15 | `run()` works on copies of `gs`, `HP` and the model state, and creates a new recorder per run, so two calls give identical results (tested). The `gs` actually used is available as `sim.gs_run`. |
| 16 | Frost-model warning: a flag on the model, reported in the ordered segment output. Non-converged frost solves now also produce a warning. |
| 17 | `rho_a_e` and `rho_a_ft` are declared in `SimState` (unused `rho_a` removed), so they are written to and read from the JSONL. The snapshot size is an optimization item (`OPTIMIZATION_REPORT.md`). |
| 18 | Not a bug. Moved to `OPTIMIZATION_REPORT.md` (edge model). |
| 19 | `np.gradient` (one-sided at the edges) for the table derivatives; `H_min = 150 kJ/kg`. |
| 20 (new) | **The last time step was never stored as a grid snapshot.** The condition `t >= gs.t_end` cannot be true inside `while t <= gs.t_end`, so field plots at `t_idx=-1` showed a state up to 100 steps old, and runs with fewer than 100 steps had no snapshot at all, which crashes the field plots. Now `t + gs.dt > gs.t_end` also triggers a snapshot. |
| 21 (new) | `plot_logph_cycles` crashed with `IndexError` when the requested time window holds no data. It now raises a clear `ValueError`. |
| 22 (new) | `__pycache__/*.pyc` files were tracked in git and changed on every run. They are removed from the index and ignored in `.gitignore`. |
| 23 (optimization round) | **The simulation could stop without saying why.** Segment warnings and errors, including the frost-blockage stop and solver exceptions, were only printed on every `print_output_every_x_it`-th step. Found when the first Case 4 schedule blocked at 58.6 min without a message. Stop messages and errors are now printed on every step, stored in `Simulator.stop_reason` / `results.stop_reason`, and repeated at the end. |
| 24 (optimization round) | `drho_dP_dH(load_path=...)` failed with `NameError` because `T_interp` was never defined in that branch. |
| 25 (optimization round) | Example notebooks: `plot_time = int(max(data['t'])/gs.dt)` assumed a fixed dt and one snapshot per step, so the spatial-slice plots indexed beyond the ~70 stored snapshots. Now the last snapshot is used. The "dry-air density" plot extracted `rho_e` (frost density) instead of `rho_a_e`. |
| 26 (optimization round) | **Unphysically light frost at cold walls.** The density correlation `207·exp(0.266·T_s − 0.0615·T_w)` (Hermes et al. 2009, fitted for walls of about −16 … −4 °C) gives ~5 kg/m³ at −20 °C walls. A small deposited mass then turned into a very thick layer: −5 °C air at RH 0.4 on a −23 °C wall grew 0.46 mm in 10 min. Found in Example Case 4 (1200 rpm, evaporator at about −20 °C), where one segment blocked the fins. The density is now limited to 25 kg/m³ (`RHO_FROST_MIN`, the model's initial frost density): 0.11 mm in 10 min in the same test. Case 2 is unchanged. In Case 3 the bound applies after the drop to −5 °C (mean frost −16 %). |
| cleanups | `fields=("t",)`; `CorLammellenRohrWU.k` uses areas that exist; `RuntimeError` instead of `SystemExit` in the Hermes model; `max(n_inner, 1)` in the dt adaptation; dead `do_print` condition, `T_inlet_air_mean`, `Air.ft_model` and unused fan attributes removed; `HeatPump.N_cond` typed as `int`; humidity plot labels generated from `n_seg_l`. Not changed: `isv`/`h_sub` (both are config inputs), the unused `New_edge_state_seg` (may be useful later), and section 5 (modelling decisions). |

**Verification after the fixes**
- Smoke test (30 s, Case 1) with an empty CoolProp cache, NumPy `DeprecationWarning`s turned into errors (this stands in for new NumPy), and plotting directly after the run: passes, and all 23 plot files are written.
- Case 1, full 10 min, regenerated in `Examples/Example_Case_1 (F0)` (CSV and all plots). 1210 steps instead of 1401, 2.5 min run time. Energy balances: air side 7843 W vs. Q_evap 7850 W; Q_evap + W_comp = 11109 W vs. Q_cond 11102 W. The spurious air heating at start-up dropped from +0.53 °C to +0.13 °C. End values compared with the old results: Q_evap 7837 → 7850 W, Q_cond 11115 → 11102 W, COP 3.390 → 3.407, p_cond 13.59 → 13.35 bar, air mass flow 1.4648 → 1.4579 kg/s.
- `run()` called twice on the same objects gives identical results and leaves `gs`/`HP` untouched. `ntheta = 60` with frost works. `use_fan = False` uses the corrected mass flow (1.49 kg/s).
- Case 3 shortened to 7.5 min with the temperature ramp moved to 6 min (`t_switch_T = 6*60`). This exercises the humidity step, frost growth and the temperature ramp in one run. 1220 steps, no errors, no unconverged frost solves, no blocking stop. After the humidity step, frost grows (mean `s_ft` 0.013 mm at 5.5 min, 0.095 mm at 7.5 min), and the air is dried row by row (inlet 3.03 → 2.63 → 2.33 g/kg). The temperature ramp to −5 °C lowers the outlet air to −8.9 °C and Q_evap from 8.3 to 7.1 kW. The fan mass flow follows the denser cold air (1.458 → 1.481 kg/s; before the fix, `rho_amb` was never updated). The "negative moisture mass flow" note appeared 38 times, now in the ordered segment output.
- The committed results of Cases 2 and 3 (60 min) were **not** regenerated (many hours of run time). They were produced with the old code.

## 1. What was tested

| Test | Setup | Result |
|---|---|---|
| Example Case 1 (F0), full 10 min | pinned `requirements.txt` (Python 3.13, NumPy 2.3.5, SciPy 1.16.3, CoolProp 7.2.0) | Runs in 3.6 min, 1401 steps. Output matches the committed `results_case_1.csv` to ~1e-13. |
| Energy balance, Case 1 end state | | Air side `m_air·cp·ΔT = 1.4648·1006·5.318 = 7837 W` = refrigerant `Q_evap = 7837 W`. `Q_evap + W_comp = 11117 W` ≈ `Q_cond = 11115 W`. Balances close. |
| Example Case 2 (F1), shortened to 15 min | same | Still running when this report was written. Matches the committed run through the start of frost formation (t ≈ 5.5 min). After frost onset each step takes 5–12 s. |
| Example Case 3 (F2) | | Identical to Case 2 until the hard-coded temperature ramp at 30 min (checked in the committed CSVs: same `Q_evap` and `s_ft` at 600/1200/1790 s). A shortened run adds nothing, so it was stopped. |
| All 34 plot calls of the example script | via the `read_data` path (CSV + JSONL) | All OK. |
| Plotting directly after a simulation (`sim_run = True`, `plots = True`) | | **Crashes** at the first `plot_any(... data['EER'] ...)` (see bug 3). |
| Same examples with current packages | NumPy 2.5.3 / CoolProp 8.0 | **Crash in time step 1** (see bug 3). |
| Fresh machine (empty CoolProp table cache `~/.CoolProp`) | pinned packages | **Segfault in time step 2** (see bug 2). |

Extra checks done outside the simulation: the refrigerant matrix against an independent solution
(bug 1), the fan/pressure-drop model (bug 5), the air outlet limiter (bug 4), the property tables
(bug 19) and the connection-path defaults (bug 11).

## 2. Summary

| # | Severity | Where | Bug |
|---|---|---|---|
| 1 | High | `refrigerant_modell_V2.py:376, 438` | Sign error in the reduced mass/energy matrix (evaporator **and** condenser) |
| 2 | High | `refrigerant_modell_V2.py:297, 312` | Segfault on the first run on a fresh machine (new TTSE `AbstractState` on every call) |
| 3 | High | `refrigerant_modell_V2.py:281, 316` | `valve_model` / `compressor_model` return 1-element arrays instead of floats: crash on new NumPy, `[x]` cells in the CSV, plot crash |
| 4 | Medium | `air_modell.py:246-250` | Outlet-air limiter is wrong when the wall is warmer than the air: air jumps to the wall temperature |
| 5 | Medium | `air_modell.py:65-82` | Kays & London face area is missing `n_seg_r`, so σ is always clipped to 1 and the clean Δp is only about 5 Pa (`dp_clean` unused) |
| 6 | Medium | `simulator_V2.py:84-92` | Fan operating point changes `m_dot` but not the inlet `v_a`, so row 0 heat transfer ignores frost blockage |
| 7 | Medium | `simulator_V2.py:387-391, 411` | Dynamic profiles and the dt window are hard-coded and ignore the case config |
| 8 | Medium | `refrigerant_modell_V2.py:514-515, 321` | Valve controller state and `HP.W_comp` are updated inside the ODE right-hand side |
| 9 | Medium | all `Example_Case_*.py`, `m_dot = ...` | Backup air mass flow is missing `n_fin` (10× too small) |
| 10 | Low | `refrigerant_modell_V2.py:232-235, 728` | SH filter is never reset (`_SH_filt` vs `_valve_SH_filt`) |
| 11 | Low | `refrigerant_modell_V2.py:40`, `geometry.py:107, 120, 133` | Default `path_variant="row_serpentine"` is invalid; `in ("serpentine")` is a substring test |
| 12 | Low | `simulator_V2.py:179` | Hard-coded `st.s_e[89]` (breaks if `ntheta != 90`) |
| 13 | Low | `refrigerant_modell_V2.py:701-704` | `x_ref = -1` for single-phase states instead of NaN |
| 14 | Low | `simulator_V2.py:27` | Refrigerant segment length uses `(pitch + thickness)·n_fin` (+4%) |
| 15 | Low | `simulator_V2.py:266, 412-423` | `run()` mutates `gs` (`dt`, `t_end`) and keeps HP/controller state, so re-running in a notebook gives different results |
| 16 | Low | `Frostmodell_V1.py:847` | Warning printed from worker threads for every segment, every step |
| 17 | Low | `recorder.py` / `initializer.py` | `rho_a_e`, `rho_a_ft` are not dataclass fields, so they are lost in the JSONL; snapshots are ~18 MB each and also kept in RAM |
| 18 | Low (perf) | `simulator_V2.py:172-189` | Edge frost model costs roughly half of the frost CPU time, but its result only feeds the stop criterion |
| 19 | Low | `derivatives_of_rho.py:113-120` | Central differences give NaN at the table edges, which is exactly where states are clipped to |
| 20 | Cosmetic | various | Small cleanups (see section 4) |

## 3. Details

### Bug 1: Sign error in the reduced refrigerant matrix (High)

**Where:** `Framework/models/refrigerant_modell_V2.py:376` (condenser) and `:438` (evaporator)

```python
fill_vals = delta_h_broadcast * b_broadcast * lower_tri
```

**What:** The internal mass flows are eliminated using
`m_{k-1} = m_in − Σ_{j<k}(a_j·dP/dt + b_j·dh_j/dt)`. Both the `a_j` and the `b_j` term enter the energy
equation of cell k with the factor `(h_{k-1} − h_k)`. The `dP/dt` column (line 369/431) is correct
(`−(h_k − h_{k-1})·cumsum(a)`), but the `dh_j/dt` columns use `+(h_k − h_{k-1})·b_j`. The sign is flipped.

**Evidence:** I copied `solve_dp_dh_evaporator` verbatim and compared it with a direct solve of all N mass
and N energy balances (internal flows `m_k` as explicit unknowns). With the current code the per-cell
energy balances are violated by up to about 1.9 kW in the test case, and `dP/dt` is off by a factor of about 2.
With the sign flipped, the residuals are 1e-13 and the result matches the reference exactly.

Effect on Case 1 (10 min) with the fix applied to both matrices:

| Quantity | Max. difference during start-up (t < 15 s) | End value (orig → fixed) |
|---|---|---|
| Q_cond | 1014 W | 11115 → 11127 W |
| Q_evap | 437 W | 7837 → 7865 W |
| Superheat | 4.1 K | 8.00 → 8.00 K |
| p_cond | 0.20 bar | 13.59 → 13.41 bar |
| COP | 0.64 | 3.390 → 3.405 |

The steady state is almost unaffected because the wrong terms are multiplied by `dh/dt`, which goes to 0.
Every transient is affected, though: start-up, the humidity step, the temperature ramp and controller action.

**Fix:** In both places:

```python
fill_vals = -delta_h_broadcast * b_broadcast * lower_tri
```

A regression test should be added (direct solve vs. reduced solve, as in the appendix).

### Bug 2: Segfault on the first run on a fresh machine (High)

**Where:** `refrigerant_modell_V2.py:297` creates a new `AbstractState("TTSE&HEOS", ...)` on every
`compressor_model` call. That function is called in every ODE right-hand-side evaluation. The crash is at the
`TTSE.update(...)` call on line 312.

**What / evidence:** With an empty CoolProp cache (`~/.CoolProp/Tables` missing, which is always the case on
Binder, linked in the README, or on a new PC), the simulation segfaults reproducibly in time step 2. That is
the first step with RPM > 0. faulthandler output:

```
Fatal Python error: Segmentation fault
  File ".../refrigerant_modell_V2.py", line 312 in compressor_model
  File ".../refrigerant_modell_V2.py", line 514 in rhs
  ...
```

The second run works because the tables then exist. Creating a TTSE state per call is also slow.

**Fix:** Create the state once in `__init__` and reuse it. `self.AS` (HEOS) already exists and is unused:

```python
# __init__
self.AS = CP.AbstractState("HEOS", self.fluid)   # or "TTSE&HEOS", but only once
# compressor_model
AS = self.AS
AS.update(HmassP_INPUTS, hi, pi); s = AS.smass()
AS.update(PSmass_INPUTS, po, s);  h_is = AS.hmass()
```

Verified: with this change there is no crash on an empty cache (two clean runs), and the refrigerant step is
about 5× faster (0.10 s vs 0.58 s per step).

### Bug 3: Mass flows are 1-element arrays instead of floats (High)

**Where:** `valve_model` (line 281) and `compressor_model` (line 316) call
`self.rho_and_derivs_vec(pi, hi)` with a scalar `hi`. That returns `rho` with shape `(1,)`, so `m` and
`HP.W_comp` become shape-`(1,)` arrays.

**Consequences:**
1. **Crash with current NumPy** (tested with 2.5.3): `b_vec[0] = m_in - m_out` (line 420) raises
   `ValueError: setting an array element with a sequence` in time step 1. With the pinned 2.3.5 it is only a
   `DeprecationWarning` (deprecated since NumPy 1.25), which is why it works on your machine.
2. In the CSV, `EER`, `COP`, `W_comp` and `m_dot_ref` are stored as `[x]` strings, unlike all other scalar
   columns (also visible in the committed `results_case_*.csv`).
3. **Plotting directly after a run crashes:** with `sim_run = True` and `plots = True`,
   `plot.plot_any(... y=data['EER'] ...)` fails with
   `ValueError: ... inhomogeneous shape after 1 dimensions` because the list mixes floats (first row) and
   `(1,)` arrays. It only works via `read_data`, and there only because of the deprecated behaviour in
   `recorder.py:309` (`b[slices] = a[slices]` with a 0-d target).

**Fix:** Use the scalar helper in both functions:

```python
rho, _, _ = self.rho_and_derivs(pi, hi)   # returns floats
```

Optionally also make the recorder robust (`float(np.squeeze(v))` for size-1 arrays in `push`).

### Bug 4: Outlet-air limiter is wrong when the air is heated (Medium)

**Where:** `air_modell.py:246-250`

```python
Q_max = (h_in - h_min) * m_dot_da
Q_tot_eff = min(Q_tot, Q_max)
```

**What:** The limit "air cannot leave colder than the wall" only works for cooling. If the wall is
warmer than the air, `Q_tot < 0` and `Q_max < 0` (and more negative), so `min` picks `Q_max` and the outlet
air is set to the wall temperature regardless of the actual heat flow.

**Evidence:** Wall 10 °C, air 0 °C, Q = −0.5 W. Result: `T_out = 10.00 °C`; correct is 0.14 °C. This happens in
Case 1 during the first seconds (the evaporator briefly runs at +0.16 °C). `T_out_air_mean` rises to +0.53 °C
with a 0 °C inlet. It will also matter for any defrost or warm-start scenario.

**Fix:**

```python
Q_tot_eff = min(Q_tot, Q_max) if Q_tot >= 0.0 else max(Q_tot, Q_max)
```

### Bug 5: Face area in the Kays & London pressure drop (Medium)

**Where:** `air_modell.py:65` and `:70-75`

```python
A_fr = geom.l_fin * geom.h_fin * stacks * n_fin              # no n_seg_r
A_min_clean = l_tube() * (h_fin - d_tube_a) * n_seg_r * stacks * n_fin
```

**What / evidence:** For the example geometry, `A_fr = 0.50 m²` and `A_min = 0.79 m²`. The free-flow area is
larger than the face area, so `σ = 1.58`, which is clipped to 1.0. The entrance and exit loss terms then cancel
out, and they keep doing so until the frost blockage drops below about 0.63. The clean Δp at the fan's design
flow is only **5.1 Pa**, so the fan curve barely matters until heavy icing (blockage 0.5 → 45 Pa; 0.2 → 797 Pa).
`cfg.dp_clean = 40 Pa` is never used. `A_min` also uses `l_tube()` (pitch + thickness) where it should use
`fin_gap()`.

**Fix:**

```python
A_fr  = geom.n_fin * geom.fin_pitch_cc * geom.h_fin * n_cols * stacks
A_min = geom.n_fin * geom.fin_gap() * (geom.h_fin - geom.d_tube_a) * n_cols * stacks
```

Then either calibrate a factor so that `dp_sys(V0) == dp_clean`, or remove `dp_clean`.

### Bug 6: Inlet velocity not coupled to the fan (Medium)

**Where:** `simulator_V2.py:91-92` updates only `input_cfg.m_dot`.

**What:** Row 0 (`ix == 0`) uses `cfg_up = input_cfg`, so `alpha_tube`, `h_eff` and `Nu_edge` always see the
initial `v_a = 1.2 m/s`, even when the fan delivers less air because of frost. Rows ≥ 1 get the velocity from
`propagate_inplace`. As a result the first row (where most frost forms) does not feel the airflow reduction.

**Fix:** After solving the fan operating point:

```python
A_free = geom.h_fin*geom.n_fin*max(geom.fin_gap()-2*s_max_step, 1e-9)*geom.n_seg_r*geom.stacks
input_cfg.v_a = input_cfg.m_dot / (input_cfg.rho_amb * A_free)
```

Use the same area definition as `propagate_inplace` so the rows are consistent.

### Bug 7: Hard-coded dynamic profiles and dt window (Medium)

**Where:** `simulator_V2.py:387-391` and `:411`

```python
input_cfg.T_a   = T_a_profile(t, 0.0, -5.0, 30*60.0, 60.0)
input_cfg.w_amb = w_amb_profile(t, input_cfg.T_a, input_cfg.p_a, 0.0, 0.8, 5*60.0, 20.0)
if 5*60.0 - 10 <= t <= 5*60.0 + 10: gs.dt = max(dt_start, gs.dt*0.5)
```

**What:**
- The profiles ignore `cfg.T_a` / `cfg.RH`. With `change_temperature=True`, the air is forced to 0 °C
  until 30 min whatever the case says, and `RH` starts at 0 regardless of the configured value.
- The dt-reduction window is fixed at 300 s ± 10 s, so it no longer matches if the humidity step moves.
  It also runs in cases without any step (Case 1).
- `input_cfg.RH` and `rho_amb` are not updated when `w_amb`/`T_a` change. `rho_amb` feeds the fan and Δp model.
- The new boundary value computed from `t` is applied from step `t + dt` on (one-step lag, minor).

**Fix:** Move the profile parameters (start and end values, switch time, ramp duration) into `GridShape` or
`CaseConfig`. Default the start values to `cfg.T_a` / `cfg.RH`. Reduce dt around `t_switch` taken from that
config. Update `RH` and `rho_amb` (e.g. via `HAPropsSI("Vha", ...)`) together with `w_amb`/`T_a`.

### Bug 8: State changes inside the ODE right-hand side (Medium)

**Where:** `refrigerant_modell_V2.py:514-515` (rhs calls `compressor_model` and `valve_controller`) and
`:321` (`self.HP.W_comp = ...`)

**What:** The BDF solver calls `rhs` many times per step: Newton iterations, finite-difference Jacobian and
rejected steps. Currently each call can:
- advance the PI controller (integrator, filter, `_valve_last_t`) at trial times that may later be rejected,
  and
- overwrite `HP.W_comp`.

So the RHS is not a pure function of `(t, y)`. That hurts the Newton/Jacobian convergence and makes the
controller depend on solver internals. In addition, the `W_comp` recorded at time t comes from whatever
trial evaluation ran last in the previous step, while `Q_evap`/`Q_cond` come from the end state. When
`RPM == 0` (t = 0), `compressor_model` returns early and the placeholder `W_comp = 0.001` remains. That gives
`EER = Q/0.001` whenever Q is non-zero while the compressor is off.

**Fix:** Evaluate the controller once per macro step, before integrating, and pass the valve position into
`rhs` as a constant. Make `compressor_model` pure (return `W` instead of writing to `HP`), and compute `W_comp`
after the integration from the final state, consistent with `Q_evap`/`Q_cond`.

### Bug 9: Backup air mass flow in the examples (Medium, only without fan)

**Where:** All three `Example_Case_*.py`:

```python
m_dot = v_a*rho_amb_in*(geom.h_fin*geom.fin_gap()*geom.n_seg_r*geom.stacks)  # 0.149 kg/s
fan_V0 = v_a*(geom.h_fin*geom.n_fin*geom.fin_gap()*geom.n_seg_r*geom.stacks)  # → 1.49 kg/s
```

The `n_fin` factor is missing in `m_dot`. With `use_fan=False` the air flow would be 10× too small.
**Fix:** Add `*geom.n_fin`.

### Bug 10: SH filter never reset (Low)

`valve_controller` uses `self._SH_filt` (lines 232-235). `__init__` and `reset_integrator` set
`self._valve_SH_filt`. **Fix:** Use one name (`_valve_SH_filt`) everywhere.

### Bug 11: Connection-path variants (Low)

- `Refrigerant.__init__(..., path_variant="row_serpentine")`: this variant does not exist and raises
  `ValueError` if the default is ever used. The simulator always passes `geom.CP`, so it is currently hidden.
- `geometry.py:210/222`: `variant in ("serpentine")` is a substring test on a string, not a tuple
  membership test, so e.g. `"pent"` is accepted. The error message lists `'row_serpentine', 'col_serpentine'`,
  which are not accepted.

**Fix:** `variant == "serpentine"` / `variant == "serpentine_variant"`, a correct default, and a correct message.

### Bug 12: Hard-coded edge index (Low)

`simulator_V2.py:179`: `st.s_e[89]`. The model itself uses `j = gs.ntheta - 1`. With `ntheta < 90` this raises
`IndexError`, which the `except` turns into "error in the edge state, ending the simulation". With
`ntheta > 90` the wrong angle is checked. **Fix:** `st.s_e[gs.ntheta - 1]`.

### Bug 13: `x_ref = -1` instead of NaN (Low)

`refrigerant_modell_V2.py:701-704` expects a `ValueError` for single-phase states, but CoolProp returns `-1`
(checked: superheated R134a gives `Q = -1.0`). 4 of 40 segments in the last Case 1 snapshot have
`x_ref = -1`, and the `x_ref` field plot shows them as −1. **Fix:**
`x_out = x if 0.0 <= x <= 1.0 else float("nan")`.

### Bug 14: Refrigerant segment length (Low)

`simulator_V2.py:27`: `L = geom.l_tube()*geom.n_fin` with `l_tube() = fin_pitch_cc + fin_thickness` gives
0.052 m, while the fin block is `n_fin*fin_pitch_cc = 0.050 m` (the centre-to-centre pitch already contains one
thickness). Refrigerant volume, inner area and wall mass are 4% too large. **Fix:**
`L = geom.fin_pitch_cc*geom.n_fin`. The same applies to the other uses of `l_tube()` (e.g. bug 5).

### Bug 15: `run()` mutates its inputs (Low)

`gs.dt` (adaptive step) and `gs.t_end` (on stop) are modified in place. The `HeatPump` object, the controller
state, the BDF solver and `air._sigma_min` also persist. Running `sim.run(...)` a second time in a notebook
without rebuilding `gs`/`HP`/`Simulator` starts from the last dt and the end state of the previous run.
**Fix:** `gs = copy.deepcopy(gs)` at the start of `run()` (return it with the results if `t_end` is needed),
and call `self.refrigerant.reset_integrator()` plus a fresh `Air()`/`HP` per run.

### Bug 16: Console flooding from threads (Low)

`Frostmodell_V1.py:847` prints "Negative moisture mass flow detected" for every affected segment in every
step, from the worker threads, outside the ordered output. That gave 322 lines in the first 30 s after frost
onset in Case 2. **Fix:** Return a flag in `info["warn"]`, or count occurrences and print once per step.

### Bug 17: Snapshot contents and size (Low)

`init_fields` adds `rho_a_e` / `rho_a_ft` as dynamic attributes. `dataclasses.asdict` (used for the JSONL)
silently drops them, while `SimState.rho_a` exists but is never used. Each snapshot is about 18 MB of JSON
(`T_e`, `w_e`, `rho_e`, `rho_a_e` are 100×90 per segment), plus a deep copy in RAM. Case 1 gives 255 MB for 14
snapshots, so a 60-min run gives about 1.3 GB on disk and in memory. **Fix:** Declare `rho_a_e`, `rho_a_ft`
in `SimState`. Store the edge fields only for `ix == 0` (the only place they are computed), or at reduced
resolution, or skip them. Optionally do not keep snapshots in RAM when streaming.

### Bug 18: Edge model has no effect on results (Low / performance)

`New_edge_state_seg_at_90` runs for every `ix == 0` segment (about 100 Picard iterations, about 0.3 s each),
but `s_e`, `T_e` and `w_e` are not used for heat flow, mass flow or air. They only feed the "blocking" stop
check. This is fine if intended, but it costs a large part of the runtime. Also, both frost solvers use
under-relaxation `omega_T = 0.1`, which needs about 100 iterations per segment and step. A larger ω or a
Newton/coupled solve would cut the 5–12 s per step considerably. Threads give little speed-up here because
of the GIL. A `ProcessPoolExecutor` or vectorising across segments would help more.

### Bug 19: NaN at the property table edges (Low)

`derivatives_of_rho.py:113-120`: central differences use `rho_interp(P ± dP)`, which is NaN outside the
grid. So `drho_dP` is NaN on the whole first P row and `drho_dH` on the last H column. `rho_and_derivs*`
clip states to exactly those edges, so a state that reaches a bound gives NaN in the BDF solver instead of a
clipped value. `H_min = 200 kJ/kg` is also above the saturated-liquid enthalpy below about 3 bar
(e.g. 187 kJ/kg at 2 bar). **Fix:** Use `np.gradient(rho, P_vec, H_vec)` (one-sided at the edges) and
lower `H_min` to about 1.5e5. Building the 2 × 240 000 `PropsSI` table on every `Simulator()` could also be
cached with the existing `save_path`/`load_path`.

## 4. Small cleanups (cosmetic)

- `Simulator.__init__(..., fields=("t"))` is a string, not a tuple. It works only because `"t"` has one character. Use `("t",)`.
- `CorLammellenRohrWU.k` (`core/corrolations.py`) calls `geom.A()` / `geom.A_i()`, which do not exist (dead code).
- `Frostmodell_Hermes_2012.frost_state_at_s` raises `SystemExit` on non-convergence, which kills a notebook kernel. Raise `RuntimeError` instead.
- Adaptive dt: `fac = (it_target / n_inner) ** k` divides by zero if `n_inner == 0`.
- `do_print = ... or stop_event.is_set()` (line 239) is evaluated before any segment runs, so the `or` part is always False.
- `humid_l` / `air_temp_l` record only column `iy = 0`, but the humidity plot labels 5 segments ("Seg 1..5") for 2 series.
- `T_inlet_air_mean`, `self.AS`, `Air._K_clean`, `_last_mdot_total`, `_fan_initialized` are unused.
- `CaseConfig.isv` and `h_sub` are the same constant (2.83e6) under two names.
- `New_edge_state_seg` (all angles) is unused.
- Neither frost solver warns when `niter` is reached without convergence.

## 5. Things to double-check (maybe intended)

- **Condenser flow arrangement:** water is co-current with the refrigerant (`T_prev[1:] = T[:-1]`,
  same index direction). Plate condensers are usually counter-current. In the results, the refrigerant outlet
  temperature equals the water outlet temperature (32.65 °C), which limits subcooling.
- **Frost mass balance:** frost growth uses `m_delta / rho_fs`, density comes from the empirical Hermes
  correlation, and the vapour removed from the air is `m_fs·A`. These three are not linked, so the frost mass
  is not conserved against the air side.
- **Controller tuning:** with `Kp = 0.1 %/K` and `Ki = 0.005 %/(K·s)` the valve moves only from 20.0 % to 19.85 %
  in 10 min (Case 1). The superheat still ends at 8 K, mostly because of the fixed 20 % bias.

## Appendix: matrix check (bug 1)

```python
# after copying solve_dp_dh_evaporator() out of update_all_segments() and defining N:
dP, dh = solve_dp_dh_evaporator(P, h, h_in, m_in, m_out, Q, V, rho, rhoP, rhoh)
a, b, c, d = V*rhoP, V*rhoh, V*(h*rhoP-1), V*(h*rhoh+rho)
m = np.empty(N+1); m[0] = m_in
for k in range(N):
    m[k+1] = m[k] - (a[k]*dP + b[k]*dh[k])          # per-cell mass balance
h_up = np.concatenate([[h_in], h[:-1]])
res = c*dP + d*dh - (m[:-1]*h_up - m[1:]*h + Q)    # per-cell energy balance, must be ~0
# current code: |res| up to ~1.9e3 W;  with fill_vals sign flipped: ~1e-13
```
