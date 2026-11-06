from CoolProp.HumidAirProp import HAPropsSI

from Framework.runtime.state import SimState
from Framework.runtime.recorder import ResultRecorder
from Framework.runtime.initializer import init_fields
from Framework.models.air_modell import Air
from Framework.models.refrigerant_modell_V2 import RefGeomParams, Refrigerant
import Framework.runtime.dynamic_models as dynamic_models
from Framework.core.config import interp_schedule

import time
import re
from datetime import datetime
import copy
import numpy as np
import threading

class Simulator:
    """Main simulation driver for the finned-tube HX model."""
    def __init__(self, geom, cfg, HP, stream_path: str | None = "sim_grid_log", fields=("t",)):
        self._stream_path = None if stream_path is None else f"{stream_path}.jsonl"
        self._fields = fields
        self.rec = None
        self.stop_reason = None
        self.gs_run = None

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
        self.stop_reason = None  # set if the simulation ends before gs.t_end

        # Fresh model state for every run
        self.HP = copy.deepcopy(self._HP_initial)
        self.refrigerant.HP = self.HP
        self.refrigerant.reset_integrator()
        self.air = Air()
        self.rec = ResultRecorder(fields=self._fields, stream_path=self._stream_path)

        # Inlet air profiles (dynamic models): start values default to the case config.
        # A schedule [(t, value), ...] replaces the hold-ramp-hold profile.
        T_a_start = cfg.T_a if gs.T_a_start is None else gs.T_a_start
        RH_start = cfg.RH if gs.RH_start is None else gs.RH_start
        use_T = gs.change_temperature or bool(gs.T_a_schedule)
        use_RH = gs.change_humidity or bool(gs.RH_schedule)

        def _T_a_at(t_eval):
            if gs.T_a_schedule:
                return interp_schedule(gs.T_a_schedule, t_eval)
            return dynamic_models.T_a_profile(t_eval, T_a_start, gs.T_a_end, gs.t_switch_T, gs.ramp_T)

        def _RH_at(t_eval):
            if gs.RH_schedule:
                return max(0.0, min(0.999999, interp_schedule(gs.RH_schedule, t_eval)))
            return dynamic_models.RH_profile(t_eval, RH_start, gs.RH_end, gs.t_switch_RH, gs.ramp_RH)

        def _apply_inlet_profiles(t_eval):
            if not (use_T or use_RH):
                return
            if use_T:
                input_cfg.T_a = _T_a_at(t_eval)
            if use_RH:
                w, RH, rho = dynamic_models.humid_air_state(input_cfg.T_a, input_cfg.p_a, RH=_RH_at(t_eval))
            else:
                # only the temperature changes: moisture content stays constant
                w, RH, rho = dynamic_models.humid_air_state(input_cfg.T_a, input_cfg.p_a, w=input_cfg.w_amb)
            input_cfg.w_amb, input_cfg.RH, input_cfg.rho_amb = w, RH, rho

        # Times around which the time step is reduced: start of each ramp,
        # or every break point of a schedule (air and heat pump)
        t_switches = []
        for flag, sched, t_sw in ((gs.change_humidity, gs.RH_schedule, gs.t_switch_RH),
                                  (gs.change_temperature, gs.T_a_schedule, gs.t_switch_T)):
            if sched:
                t_switches.extend(float(p[0]) for p in sched)
            elif flag:
                t_switches.append(t_sw)
        for sched in (getattr(self.HP, "RPM_schedule", None), getattr(self.HP, "T_in_water_schedule", None)):
            if sched:
                t_switches.extend(float(p[0]) for p in sched)

        # Only apply at t=0 if a start value differs from the case config;
        # otherwise the inlet state stays exactly as configured
        if (use_T and _T_a_at(0.0) != cfg.T_a) or (use_RH and _RH_at(0.0) != cfg.RH):
            _apply_inlet_profiles(0.0)

        cfg_grid, st_grid = build_segment_grids(base_cfg=input_cfg, geom=geom, gs=gs)
        n_x = len(cfg_grid)
        n_y = len(cfg_grid[0])

        # Frost models; all frosted segments of a row are solved together
        model_e, model_ft = model.Frostmodell_Edge(), model.Frostmodell_Finn_and_Tube()

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

            stop_event = threading.Event()  # stop flag
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

                if info["air"] is not None:
                    T_out, w_out, p_out = info["air"]
                    parts.append(f"Air(T={T_out:.2f} °C, w={w_out:.3e} kg/kg, P={p_out:.3e} Pa)")

                line = " | ".join(parts)

                # Warnings/errors as extra lines, still ordered and readable
                extra = []
                extra.extend(info["warn"])
                extra.extend(info["stop"])
                extra.extend(info["err"])
                if extra:
                    return line + "\n" + "\n".join(extra)
                return line

            def _frost_condition(cfg, cfg_up, info):
                """Frost forms where the air at the wall is saturated and the wall is below 0 °C."""
                try:
                    RH_air_at_wall = HAPropsSI("R", "T", cfg.T_tube + 273.15, "P", cfg_up.p_a, "W", cfg_up.w_amb)
                except Exception:
                    w_sat_wall = HAPropsSI("W", "T", cfg.T_tube + 273.15, "P", cfg_up.p_a, "R", 1.0)
                    RH_air_at_wall = cfg_up.w_amb / max(w_sat_wall, 1e-12)
                RH_air_at_wall = max(0.0, min(1.0, RH_air_at_wall))
                if RH_air_at_wall >= 0.999 and cfg.T_tube <= 0.0 and gs.cal_frost:
                    cfg.frost_condition = True
                info["rh_wall"], info["frost_condition"] = RH_air_at_wall, cfg.frost_condition

            def _solve_frost_row(kind, ix, iys, ups, infos):
                """Edge ("edge") or fin/tube ("ft") frost state of the frosted segments iys of row ix, all at once."""
                cfgs, sts = [cfg_grid[ix][iy] for iy in iys], [st_grid[ix][iy] for iy in iys]
                name, solver = {"edge": ("Edge", model_e.New_edge_state_row_at_90),
                                "ft": ("Fin/tube", model_ft.New_finn_and_tube_state_row)}[kind]
                t0_row = time.perf_counter()
                try:
                    results = solver([ups[iy] for iy in iys], cfgs, geom, sts, gs)
                except Exception as e:
                    state = "edge" if kind == "edge" else "finn and tube"
                    for iy in iys:
                        infos[iy]["err"] += [f"\033[31mThere was an error in the calculation for the new {state} state, ending the simulation.\033[0m",
                                             f"\033[31m{e}\033[0m"]
                    stop_event.set()
                    return
                ct = time.perf_counter() - t0_row  # time of the whole row
                for iy, st, (iters, res_T, res_w, *neg) in zip(iys, sts, results):
                    info = infos[iy]
                    info[kind] = (iters, res_w, res_T, ct)
                    if res_T > 1e-6 or res_w > 1e-6:
                        info["warn"].append(f"\033[93m{name} solver in segment {(ix, iy)} not converged "
                                            f"(it={iters}, res_T={res_T:.2e}, res_w={res_w:.2e}).\033[0m")
                    if neg and neg[0]:
                        info["warn"].append(f"\033[93mNegative moisture mass flow in segment {(ix, iy)}, m_f set to 0.\033[0m")
                    if (st.s_e[gs.ntheta - 1] if kind == "edge" else st.s_ft) >= s_max:
                        where = "edge segment" if kind == "edge" else "segment"
                        info["stop"].append(f"\033[31mThe frost in the {where} {(ix, iy)} is blocking the air flow, ending the simulation.\033[0m")
                        stop_event.set()

            def _air(cfg, st, cfg_up, info):
                """Heat flow into the wall (returned) and outlet air of one segment."""
                if not cfg.frost_condition:
                    st.T_e[:] = cfg.T_tube
                    st.T_ft[:] = cfg.T_tube
                m_s_seg, Q_sens_fs, Q_seg_x0, Q_steady = model_ft.segment_flows(cfg_up, geom, st, gs)
                if not cfg.frost_condition:
                    m_s_seg, Q_sens_fs, Q_seg_x0 = 0.0, Q_steady, Q_steady
                if gs.cal_air:
                    info["air"] = self.air.propagate_inplace(cfg_up, cfg, st.s_ft, st, geom,
                                                             input_cfg.m_dot / (n_y * geom.stacks), Q_sens_fs, m_s_seg, gs.dt)
                return Q_seg_x0

            # --- Row by row in air-flow direction (ix); the segments of a row are independent ---
            print_every = gs.print_output_every_x_it
            do_print = (it % print_every == 0)

            for ix in range(n_x):
                ups = [input_cfg if ix == 0 else cfg_grid[ix - 1][iy] for iy in range(n_y)]  # upstream air
                infos = [{"ix": ix, "iy": iy, "edge": None, "ft": None, "air": None, "warn": [],
                          "stop": [],  # conditions that end the simulation (always printed)
                          "err": [], "rh_wall": 0.0, "frost_condition": False} for iy in range(n_y)]
                for iy in range(n_y):
                    st_grid[ix][iy].t = t
                    _frost_condition(cfg_grid[ix][iy], ups[iy], infos[iy])
                frosted = [iy for iy in range(n_y) if cfg_grid[ix][iy].frost_condition]
                if frosted and ix == 0:
                    _solve_frost_row("edge", ix, frosted, ups, infos)
                if frosted:
                    _solve_frost_row("ft", ix, frosted, ups, infos)

                for iy, info in enumerate(infos):
                    Q_seg_x0_list[ix, iy] = _air(cfg_grid[ix][iy], st_grid[ix][iy], ups[iy], info)
                    max_rh_wall_step = max(max_rh_wall_step, info["rh_wall"])
                    any_frost_condition_step = any_frost_condition_step or info["frost_condition"]
                    if do_print:
                        print(_fmt_segment_line(info))
                    elif info["stop"] or info["err"]:
                        # conditions that end the simulation are reported on every step
                        print("\n".join(info["stop"] + info["err"]))
                    if (info["stop"] or info["err"]) and self.stop_reason is None:
                        msg = (info["stop"] + info["err"])[0]
                        self.stop_reason = f"t = {t:.1f} s: " + re.sub(r"\x1b\[[0-9;]*m", "", msg)
                if not do_print and ix == 0:
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

            T = float(self.refrigerant.props_PH(p1_suction, h1_suction)[0])
            T_sat = float(self.refrigerant.T_sat(p1_suction))
            SH = T-T_sat

            h_eff_vals = [
                model_ft.h_eff(cfg_ij, geom, st_ij)
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
                          dt=gs.dt,
                          # boundary conditions at time t
                          T_in_air=input_cfg.T_a,
                          RH_in=input_cfg.RH,
                          w_in=input_cfg.w_amb,
                          RPM=self.HP.RPM(t),
                          T_in_water=self.HP.T_in_water_at(t))

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

        if self.stop_reason is not None:
            print(f"\033[31mSimulation stopped early: {self.stop_reason}\033[0m")
        self.rec.stop_reason = self.stop_reason

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
