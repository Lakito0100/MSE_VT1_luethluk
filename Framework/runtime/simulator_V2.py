from CoolProp.HumidAirProp import HAPropsSI

from Framework.runtime.state import SimState
from Framework.runtime.recorder import ResultRecorder
from Framework.runtime.initializer import init_fields
from Framework.models.air_modell import Air
from Framework.models.refrigerant_modell_V2 import RefGeomParams, Refrigerant
import Framework.runtime.dynamic_models as dynamic_models

import time
from datetime import datetime
import copy
import numpy as np
from CoolProp.CoolProp import PropsSI
import io
from contextlib import redirect_stdout
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class Simulator:
    """Main simulation driver for the finned-tube HX model."""
    def __init__(self, geom, cfg, HP, stream_path: str | None = "sim_grid_log", fields=("t",)):
        self._stream_path = None if stream_path is None else f"{stream_path}.jsonl"
        self._fields = fields
        self.rec = None

        A_flow = (((geom.d_tube_a/2)-geom.tube_thickness)**2) * np.pi
        L = geom.l_tube()*geom.n_fin  # tube length of one segment (n_fin fin pitches)
        A_inner = (geom.d_tube_a-2*geom.tube_thickness)*np.pi*L
        V_wall = (((geom.d_tube_a/2)**2)*np.pi - A_flow)*L

        self.gp = RefGeomParams(
            A_flow  = A_flow,          # cross-section of one tube
            dx      = L,     # length per segment
            A_inner = A_inner,    # inner area per segment
            V_wall  = V_wall,     # wall volume per segment
            rho_wall= geom.rho_solid,
            c_wall  = geom.c_solid,
            dp_ref_seg = 0.0  # or just 0.0 for now
        )

        self.HP = HP
        self._HP_initial = copy.deepcopy(HP)  # every run starts from this state
        self.air = Air()
        self.refrigerant = Refrigerant(geom,self.gp,cfg,HP,geom.CP)


    def run(self, cfg, geom, gs, model):
        """
        Run the transient simulation over the configured time horizon.

        The inputs cfg, gs and the HeatPump given to the constructor are not
        modified, so run() can be called repeatedly with identical results.
        The GridShape actually used (adapted dt, t_end shortened on an early
        stop) is available afterwards as self.gs_run.
        """
        input_cfg = copy.deepcopy(cfg)
        gs = copy.deepcopy(gs)
        self.gs_run = gs

        # Fresh model state for every run
        self.HP = copy.deepcopy(self._HP_initial)
        self.refrigerant.HP = self.HP
        self.refrigerant.reset_integrator()
        self.air = Air()
        self.rec = ResultRecorder(fields=self._fields, stream_path=self._stream_path)

        # Inlet air profiles (dynamic models): start values default to the case config
        T_a_start = cfg.T_a if gs.T_a_start is None else gs.T_a_start
        RH_start = cfg.RH if gs.RH_start is None else gs.RH_start

        def _apply_inlet_profiles(t_eval):
            if not (gs.change_temperature or gs.change_humidity):
                return
            if gs.change_temperature:
                input_cfg.T_a = dynamic_models.T_a_profile(
                    t_eval, T_a_start, gs.T_a_end, gs.t_switch_T, gs.ramp_T)
            if gs.change_humidity:
                RH_t = dynamic_models.RH_profile(
                    t_eval, RH_start, gs.RH_end, gs.t_switch_RH, gs.ramp_RH)
                w, RH, rho = dynamic_models.humid_air_state(input_cfg.T_a, input_cfg.p_a, RH=RH_t)
            else:
                # only the temperature changes: moisture content stays constant
                w, RH, rho = dynamic_models.humid_air_state(input_cfg.T_a, input_cfg.p_a, w=input_cfg.w_amb)
            input_cfg.w_amb, input_cfg.RH, input_cfg.rho_amb = w, RH, rho

        # Times around which the time step is reduced (start of each active ramp)
        t_switches = []
        if gs.change_humidity:
            t_switches.append(gs.t_switch_RH)
        if gs.change_temperature:
            t_switches.append(gs.t_switch_T)

        # Only apply at t=0 if a start value differs from the case config;
        # otherwise the inlet state stays exactly as configured
        if (gs.change_temperature and T_a_start != cfg.T_a) or (gs.change_humidity and RH_start != cfg.RH):
            _apply_inlet_profiles(0.0)

        cfg_grid, st_grid = build_segment_grids(base_cfg=input_cfg, geom=geom, gs=gs)
        n_x = len(cfg_grid)
        n_y = len(cfg_grid[0])

        # --- Parallel helpers (thread-local instances) ---
        _tls = threading.local()

        def _get_thread_models():
            # Important: thread-local instances prevent race conditions if models keep state
            if not hasattr(_tls, "model_e"):
                _tls.model_e = model.Frostmodell_Edge()
                _tls.model_ft = model.Frostmodell_Finn_and_Tube()
            return _tls.model_e, _tls.model_ft

        s_max = (geom.fin_gap()/2.0) * 0.95

        t = 0.0
        it = 0
        t0_start = time.perf_counter()
        n_inner = 10
        dt_start = gs.dt

        while t <= gs.t_end:
            it += 1
            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.2f}' +
                  " s | " + f'{t / 60:.1f} min')

            # --- per time step ---
            Q_seg_x0_list = np.zeros((n_x, n_y), dtype=float)
            t_iteration_start = time.perf_counter()

            gs.t = t

            # --- Fan operating point: once per time step ---
            if self.air._fan_enabled(input_cfg):
                # Conservative: worst-case icing (max s_ft) -> smallest free cross-section
                s_max_step = max(float(st_grid[ix][iy].s_ft) for ix in range(n_x) for iy in range(n_y))

                sigma = self.air._sigma_from_frost(geom, s_max_step)
                self.air._sigma_min = min(self.air._sigma_min, float(sigma))

                mdot_total = self.air._solve_fan_operating_point(input_cfg, geom)
                input_cfg.m_dot = float(mdot_total)

                # Inlet velocity (in the clean fin gaps) consistent with the fan flow;
                # same area definition as the outlet velocity in Air.propagate_inplace
                A_free_inlet = self.air.free_flow_area_seg(geom, 0.0) * n_y * geom.stacks
                input_cfg.v_a = input_cfg.m_dot / (input_cfg.rho_amb * A_free_inlet)

                print(f"[FAN] sigma_min={self.air._sigma_min:.3f}, mdot_total={input_cfg.m_dot:.4f} kg/s")

            max_workers = min(os.cpu_count() or 1, n_y)
            stop_event = threading.Event()  # thread-safe stop flag
            max_rh_wall_step = 0.0
            any_frost_condition_step = False

            def _fmt_segment_line(info):
                ix, iy = info["ix"], info["iy"]
                parts = [f"Seg[{ix},{iy}]"]

                # Show Edge / FT only when computed
                if info["edge"] is not None:
                    iter_e, res_w_e, res_T_e, dt_e = info["edge"]
                    parts.append(f"Edge(it={iter_e}, w={res_w_e:.3e}, T={res_T_e:.3e}, ct={dt_e:.3f} s)")
                if info["ft"] is not None:
                    iter_ft, res_w_ft, res_T_ft, dt_ft = info["ft"]
                    parts.append(f"FT(it={iter_ft}, w={res_w_ft:.3e}, T={res_T_ft:.3e}, ct={dt_ft:.3f} s)")

                # Always show air
                T_out, w_out, p_out = info["air"]
                parts.append(f"Air(T={T_out:.2f} °C, w={w_out:.3e} kg/kg, P={p_out:.3e} Pa)")

                line = " | ".join(parts)

                # Warnings/errors as extra lines, still ordered and readable
                extra = []
                extra.extend(info["warn"])
                extra.extend(info["err"])
                if extra:
                    return line + "\n" + "\n".join(extra)
                return line

            def _process_segment(ix, iy):
                cfg = cfg_grid[ix][iy]
                st  = st_grid[ix][iy]
                st.t = t

                # Upstream values are read-only
                if ix == 0:
                    cfg_up = input_cfg
                else:
                    cfg_up = cfg_grid[ix - 1][iy]

                model_e_loc, model_ft_loc = _get_thread_models()

                info = {
                    "ix": ix,
                    "iy": iy,
                    "edge": None,
                    "ft": None,
                    "air": None,
                    "warn": [],
                    "err": [],
                    "rh_wall": 0.0,
                    "frost_condition": False}

                # ---------------- Frost condition ----------------
                try:
                    RH_air_at_wall = HAPropsSI("R",
                                               "T", cfg.T_tube + 273.15,
                                               "P", cfg_up.p_a,
                                               "W", cfg_up.w_amb)
                except Exception:
                    w_sat_wall = HAPropsSI("W",
                                           "T", cfg.T_tube + 273.15,
                                           "P", cfg_up.p_a,
                                           "R", 1.0)
                    RH_air_at_wall = cfg_up.w_amb / max(w_sat_wall, 1e-12)

                RH_air_at_wall = max(0.0, min(1.0, RH_air_at_wall))

                if RH_air_at_wall >= 0.999 and cfg.T_tube <= 0.0 and gs.cal_frost:
                    cfg.frost_condition = True
                info["rh_wall"] = RH_air_at_wall
                info["frost_condition"] = cfg.frost_condition

                # ---------------- Updating the edge state ----------------
                if ix == 0 and cfg.frost_condition:
                    t0_edge = time.perf_counter()
                    iter_e = 0
                    res_T_e = float("nan")
                    res_w_e = float("nan")
                    try:
                        iter_e, res_T_e, res_w_e = model_e_loc.New_edge_state_seg_at_90(cfg_up, cfg, geom, st, gs)
                        if res_T_e > 1e-6 or res_w_e > 1e-6:
                            info["warn"].append(
                                f"\033[93mEdge solver in segment {(ix, iy)} not converged "
                                f"(it={iter_e}, res_T={res_T_e:.2e}, res_w={res_w_e:.2e}).\033[0m")
                        if st.s_e[gs.ntheta - 1] >= s_max:
                            info["warn"].append(
                                f"\033[31mThe frost in the edge segment {(ix, iy)} is blocking the air flow, ending the simulation.\033[0m"
                            )
                            stop_event.set()
                    except Exception as e:
                        info["err"].append("\033[31mThere was an error in the calculation for the new edge state, ending the simulation.\033[0m")
                        info["err"].append(f"\033[31m{e}\033[0m")
                        stop_event.set()
                    t1_edge = time.perf_counter()
                    info["edge"] = (iter_e, res_w_e, res_T_e, t1_edge - t0_edge)

                # ---------------- Updating the fin and tube state ----------------
                if cfg.frost_condition:
                    t0_ft = time.perf_counter()
                    iter_ft = 0
                    res_T_ft = float("nan")
                    res_w_ft = float("nan")
                    try:
                        iter_ft, res_T_ft, res_w_ft = model_ft_loc.New_finn_and_tube_state_seg(cfg_up, cfg, geom, st, gs)
                        if res_T_ft > 1e-6 or res_w_ft > 1e-6:
                            info["warn"].append(
                                f"\033[93mFin/tube solver in segment {(ix, iy)} not converged "
                                f"(it={iter_ft}, res_T={res_T_ft:.2e}, res_w={res_w_ft:.2e}).\033[0m")
                        if getattr(model_ft_loc, "last_negative_mass_flux", False):
                            info["warn"].append(
                                f"\033[93mNegative moisture mass flow in segment {(ix, iy)}, m_f set to 0.\033[0m")
                        if st.s_ft >= s_max:
                            info["warn"].append(
                                f"\033[31mThe frost in the segment {(ix, iy)} is blocking the air flow, ending the simulation.\033[0m"
                            )
                            stop_event.set()
                    except Exception as e:
                        info["err"].append("\033[31mThere was an error in the calculation for the new finn and tube state, ending the simulation.\033[0m")
                        info["err"].append(f"\033[31m{e}\033[0m")
                        stop_event.set()
                    t1_ft = time.perf_counter()
                    info["ft"] = (iter_ft, res_w_ft, res_T_ft, t1_ft - t0_ft)

                # ---------------- Updating the air state ----------------
                m_dot_a = input_cfg.m_dot / (n_y*geom.stacks)

                if gs.cal_air:
                    if cfg.frost_condition:
                        m_s_seg = model_ft_loc.segment_mass_flux_air_frost(cfg_up, geom, st, gs)
                        Q_sens_fs, Q_seg_x0, Q_steady = model_ft_loc.segment_heat_flux_air_frost(cfg_up, geom, st, gs)
                        T_out, w_out, p_out = self.air.propagate_inplace(
                            cfg_up, cfg, st.s_ft, st, geom, m_dot_a, Q_sens_fs, m_s_seg, gs.dt
                        )

                        q_for_list = Q_seg_x0
                    else:
                        st.T_e[:] = cfg.T_tube
                        st.T_ft[:] = cfg.T_tube
                        Q_sens_fs, Q_seg_x0, Q_steady = model_ft_loc.segment_heat_flux_air_frost(cfg_up, geom, st, gs)
                        T_out, w_out, p_out = self.air.propagate_inplace(
                            cfg_up, cfg, st.s_ft, st, geom, m_dot_a, Q_steady, 0.0, gs.dt
                        )

                        q_for_list = Q_steady

                    info["air"] = (T_out, w_out, p_out)

                return iy, q_for_list, info

            # --- Parallel over iy, sequential over ix ---
            print_every = gs.print_output_every_x_it
            do_print = (it % print_every == 0)

            for ix in range(n_x):
                infos_by_iy = [None] * n_y

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_process_segment, ix, iy) for iy in range(n_y)]
                    for fut in as_completed(futures):
                        iy, q_val, info = fut.result()
                        Q_seg_x0_list[ix, iy] = q_val
                        max_rh_wall_step = max(max_rh_wall_step, info["rh_wall"])
                        any_frost_condition_step = any_frost_condition_step or info["frost_condition"]

                        if do_print:
                            infos_by_iy[iy] = info

                # Print in order (iy=0..n_y-1), compact output

                if do_print:
                    for iy in range(n_y):
                        info = infos_by_iy[iy]
                        if info is not None:
                            print(_fmt_segment_line(info))
                elif not do_print and ix == 0:
                    print(f"Frost and Air output in {print_every - (it % print_every)} Time Steps.")

            if stop_event.is_set():
                gs.t_end = t



            # Push the data
            path_ref = geom.build_connection_path(geom.CP)
            pos_of = {seg: k for k, seg in enumerate(path_ref)}
            (x0, y0) = path_ref[0]
            (x_end, y_end) = path_ref[-1]

            # Calculation of the heat transfer coefficient
            # --- Inlet/Outlet ---
            T_outlet_air_mean = np.mean([cfg_grid[-1][iy].T_a for iy in range(n_y)])
            T_ref_out = cfg_grid[x_end][y_end].T_ref

            # Heat flows of the state at time t (end of the previous refrigerant update)
            Q_cond = self.HP.Q_cond
            Q_evap = self.HP.Q_evap

            mean_s_ft = np.mean([st_grid[ix][iy].s_ft
                                 for ix in range(n_x)
                                 for iy in range(n_y)])
            max_s_ft = np.max([st_grid[ix][iy].s_ft
                                 for ix in range(n_x)
                                 for iy in range(n_y)])
            humid_l = np.array([seg[0].w_amb for seg in cfg_grid])
            air_temp_l = np.array([seg[0].T_a for seg in cfg_grid])

            p_ref_evap = cfg_grid[x0][y0].p_ref
            p_ref_cond = self.HP.p_ref_cond

            m_dot_air = input_cfg.m_dot
            v_in_air = np.mean([cfg_grid[0][iy].v_a for iy in range(n_y)])

            p1_suction = p_ref_evap
            h1_suction = cfg_grid[x_end][y_end].h_ref

            p2_discharge = p_ref_cond
            m_comp, h2_discharge = self.refrigerant.compressor_model(pi=p1_suction, hi=h1_suction, po=p2_discharge, RPM=self.HP.RPM(t))

            # Compressor power and efficiencies at time t (same state as Q_evap / Q_cond)
            W_comp = m_comp * (h2_discharge - h1_suction)
            if W_comp > 1e-9:
                EER = Q_evap / W_comp
                COP = Q_cond / W_comp
            else:
                EER = float("nan")  # compressor off
                COP = float("nan")

            p3_cond_out = p_ref_cond
            h3_cond_out = self.HP.h_ref_cond[-1]

            p4_valve_out = p_ref_evap
            VPos = self.refrigerant.valve_pos
            if VPos is None:
                VPos = self.refrigerant.valve_controller(t=t, P_suction=p1_suction, h_suction=h1_suction)
            m_valve, h4_valve_out = self.refrigerant.valve_model(pi=p3_cond_out, hi=h3_cond_out, po=p4_valve_out, VPos=VPos)

            cycle_ph = [
                [float(p1_suction), float(h1_suction)],
                [float(p2_discharge), float(h2_discharge)],
                [float(p3_cond_out), float(h3_cond_out)],
                [float(p4_valve_out), float(h4_valve_out)],
            ]

            T = float(PropsSI("T", "P", p1_suction, "H", h1_suction, cfg.ref_str))
            T_sat = float(PropsSI("T", "P", p1_suction, "Q", 0, cfg.ref_str))
            SH = T-T_sat

            model_e_loc, model_ft_loc = _get_thread_models()
            h_eff_vals = [
                model_ft_loc.h_eff(cfg_ij, geom, st_ij)
                for cfg_row, st_row in zip(cfg_grid, st_grid)
                for cfg_ij, st_ij in zip(cfg_row, st_row)
            ]
            h_eff_mean = float(np.mean(h_eff_vals))

            T_water_outlet = self.HP.T_water[-1]

            A_evap_overall = sum(
                geom.A_one_segment_frost(st_ij.s_ft)
                for st_row in st_grid
                for st_ij in st_row
            )

            # Store simple time series in memory
            self.rec.push(t=t,
                          EER=EER,
                          COP=COP,
                          Q_cond=Q_cond,
                          Q_evap=Q_evap,
                          W_comp=W_comp,
                          h_eff_mean=h_eff_mean,
                          A_evap=A_evap_overall,
                          mean_s_ft=mean_s_ft,
                          max_s_ft=max_s_ft,
                          m_dot_air = m_dot_air,
                          v_in_air = v_in_air,
                          T_out_water = T_water_outlet,
                          T_out_air_mean=T_outlet_air_mean,
                          T_out_ref=T_ref_out,
                          p_ref_evap=p_ref_evap,
                          superheating=SH,
                          T_sat=T_sat-273.15,
                          p_ref_cond=p_ref_cond,
                          humidity=humid_l,
                          air_temp_l=air_temp_l,
                          cycle_ph=cycle_ph,
                          m_dot_ref=m_comp,
                          valve_pos=VPos,
                          dt=gs.dt)

            # Push grid snapshot
            # also store the last time step (t + dt passes t_end) and an early stop (t_end set to t)
            if it % gs.store_grid_every_x_it == 0 or t + gs.dt > gs.t_end or t >= gs.t_end:
                print('Pushing grid state')
                self.rec.push_grid_snapshot(
                    t=t,
                    cfg_grid=cfg_grid,
                    st_grid=st_grid,
                    meta={"it": it}
                )


            dt_step_n = gs.dt

            # Dynamic models: inlet air for the next time step (t + dt)
            _apply_inlet_profiles(t + dt_step_n)

            # Updating the refrigerant state

            if gs.cal_ref:
                print('Calculating the refrigerant state for the next time step in all segments...')

                n_inner = self.refrigerant.update_all_segments(
                    input_cfg,  # inlet BC
                    cfg_grid,
                    st_grid,
                    geom,
                    Q_seg_x0_list,  # this is Q_f per segment
                    time=t,
                    dt=dt_step_n,
                )

                # Adaptive time step: small steps around the ramps of the inlet profiles
                if any(abs(t - ts) <= gs.dt_window_switch for ts in t_switches):
                    gs.dt = max(dt_start, gs.dt * 0.5)
                else:
                    k = 0.2  # aggressiveness
                    fac_min, fac_max = 0.5, 1.5  # limit per outer step

                    fac = (gs.it_target / max(n_inner, 1)) ** k
                    fac = max(fac_min, min(fac_max, fac))

                    gs.dt *= fac
                    gs.dt = max(gs.dt_min, min(gs.dt_max, gs.dt))

                print(f"Updating dt for the next time step to dt={gs.dt:.2f} s")

            t_iteration_end = time.perf_counter()
            time_iteration = t_iteration_end - t_iteration_start
            time_remaining_est = time_iteration * (gs.t_end - t) / gs.dt

            print(f"Cycle time for time step {it}: {time_iteration:.1f} s\n"
                  f"\033[94mApproximate remaining time: {time_remaining_est / 60.0:.1f} min\033[0m")

            t += dt_step_n
            print(
                "--------------------------------------------------------------------------------------------------"
            )

        t1_end = time.perf_counter()
        sim_time = t1_end - t0_start
        end_wall = datetime.now()
        print(
            f"Simulation ended at {end_wall:%Y-%m-%d %H:%M:%S}\n"
            f"Time to complete: {sim_time / 60:.1f} min"
        )

        self.rec.close()
        return self.rec


def build_segment_grids(base_cfg,
                        geom,
                        gs):
    """
    Create 2D lists (grids) of CaseConfig and SimState.
    - first dimension: x direction (air flow) -> n_seg_l
    - second dimension: y direction (rows in refrigerant direction) -> n_seg_r

    Returns:
        cfg_grid[ix][iy], st_grid[ix][iy]
    """

    n_x = int(geom.n_seg_l)   # Segments in air-flow direction
    n_y = int(geom.n_seg_r)   # Segments in "row" direction

    cfg_grid = [[None for _ in range(n_y)] for _ in range(n_x)]
    st_grid  = [[None for _ in range(n_y)] for _ in range(n_x)]

    for ix in range(n_x):
        for iy in range(n_y):
            cfg_ij = copy.deepcopy(base_cfg)

            st_ij = SimState()
            init_fields(cfg_ij, st_ij, gs)

            cfg_grid[ix][iy] = cfg_ij
            st_grid[ix][iy]  = st_ij

    return cfg_grid, st_grid
