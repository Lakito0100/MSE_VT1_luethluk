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
    def __init__(self, geom, cfg, HP, stream_path: str | None = "sim_grid_log", fields=("t")):
        self.rec = ResultRecorder(fields=fields, stream_path=f"{stream_path}.jsonl")

        A_flow = (((geom.d_tube_a/2)-geom.tube_thickness)**2) * np.pi
        L = geom.l_tube()
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
        self.air = Air()
        self.refrigerant = Refrigerant(geom,self.gp,cfg,HP,geom.CP)


    def run(self, cfg, geom, gs, model):
        input_cfg = copy.deepcopy(cfg)
        input_cfg.fan_master = True
        cfg_grid, st_grid = build_segment_grids(base_cfg=cfg, geom=geom, gs=gs)
        n_x = len(cfg_grid)
        n_y = len(cfg_grid[0])

        # --- Parallel helpers (thread-local instances) ---
        _tls = threading.local()

        def _get_thread_models():
            # Wichtig: pro Thread eigene Instanzen (verhindert Race-Conditions, falls Models internen Zustand haben)
            if not hasattr(_tls, "model_e"):
                _tls.model_e = model.Frostmodell_Edge()
                _tls.model_ft = model.Frostmodell_Finn_and_Tube()
            return _tls.model_e, _tls.model_ft

        s_max = geom.fin_gap()/2.0

        t = 0.0
        it = 0
        t0_start = time.perf_counter()
        n_inner = 10

        while t <= gs.t_end:
            it += 1
            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.1f}' +
                  " s | " + f'{t / 60:.1f} min')

            # --- pro Zeitschritt ---
            Q_seg_x0_list = np.zeros((n_x, n_y), dtype=float)
            t_iteration_start = time.perf_counter()

            # gs.t einmal pro Zeitschritt setzen (nicht in Threads)
            gs.t = t

            max_workers = min(os.cpu_count() or 1, n_y)
            stop_event = threading.Event()  # thread-safe stop flag

            def _fmt_segment_line(info):
                ix, iy = info["ix"], info["iy"]
                parts = [f"Seg[{ix},{iy}]"]

                # Edge / FT nur anzeigen, wenn berechnet
                if info["edge"] is not None:
                    iter_e, res_w_e, res_T_e, dt_e = info["edge"]
                    parts.append(f"Edge(it={iter_e}, w={res_w_e:.3e}, T={res_T_e:.3e}, ct={dt_e:.3f} s)")
                if info["ft"] is not None:
                    iter_ft, res_w_ft, res_T_ft, dt_ft = info["ft"]
                    parts.append(f"FT(it={iter_ft}, w={res_w_ft:.3e}, T={res_T_ft:.3e}, ct={dt_ft:.3f} s)")

                # Air immer anzeigen
                T_out, w_out, p_out = info["air"]
                parts.append(f"Air(T={T_out:.2f} °C, w={w_out:.3e} kg/kg, P={p_out:.3e} Pa)")

                line = " | ".join(parts)

                # Warnings/Errors als Zusatzzeilen, aber weiterhin geordnet und lesbar
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

                # upstream nur lesen
                if ix == 0:
                    cfg_up = input_cfg
                else:
                    cfg_up = cfg_grid[ix - 1][iy]

                model_e_loc, model_ft_loc = _get_thread_models()

                info = {"ix": ix, "iy": iy, "edge": None, "ft": None, "air": None, "warn": [], "err": []}

                # ---------------- Frost condition ----------------
                try:
                    RH_air_at_wall = HAPropsSI("R",
                                               "T", cfg.T_tube + 273.15,
                                               "P", cfg.p_a,
                                               "W", cfg.w_amb)
                except Exception:
                    w_sat_wall = HAPropsSI("W",
                                           "T", cfg.T_tube + 273.15,
                                           "P", cfg.p_a,
                                           "R", 1.0)
                    RH_air_at_wall = cfg.w_amb / max(w_sat_wall, 1e-12)

                RH_air_at_wall = max(0.0, min(1.0, RH_air_at_wall))

                if RH_air_at_wall >= 0.999 and gs.cal_frost:
                    cfg.frost_condition = True

                # ---------------- Updating the edge state ----------------
                if ix == 0 and cfg.frost_condition:
                    t0_edge = time.perf_counter()
                    iter_e = 0
                    res_T_e = float("nan")
                    res_w_e = float("nan")
                    try:
                        iter_e, res_T_e, res_w_e = model_e_loc.New_edge_state_seg_at_90(cfg, geom, st, gs)
                        if st.s_e[89] >= s_max:
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

                # ---------------- Updating the finn and tube state ----------------
                if cfg.frost_condition:
                    t0_ft = time.perf_counter()
                    iter_ft = 0
                    res_T_ft = float("nan")
                    res_w_ft = float("nan")
                    try:
                        iter_ft, res_T_ft, res_w_ft = model_ft_loc.New_finn_and_tube_state_seg(cfg, geom, st, gs)
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
                        Q_seg_fs, Q_seg_x0, Q_steady = model_ft_loc.segment_heat_flux_air_frost(cfg_up, geom, st, gs)
                        T_out, w_out, p_out = self.air.propagate_inplace(
                            cfg_up, cfg, st.s_ft, st, geom, m_dot_a, Q_seg_fs, m_s_seg, gs.dt
                        )

                        q_for_list = Q_seg_x0
                    else:
                        st.T_e[:] = cfg.T_tube
                        st.T_ft[:] = cfg.T_tube
                        Q_seg_fs, Q_seg_x0, Q_steady = model_ft_loc.segment_heat_flux_air_frost(cfg_up, geom, st, gs)
                        T_out, w_out, p_out = self.air.propagate_inplace(
                            cfg_up, cfg, st.s_ft, st, geom, m_dot_a, Q_steady, 0.0, gs.dt
                        )

                        q_for_list = Q_steady

                    info["air"] = (T_out, w_out, p_out)

                return iy, q_for_list, info

            # --- Parallel über iy, sequenziell über ix ---
            for ix in range(n_x):
                infos_by_iy = [None] * n_y

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_process_segment, ix, iy) for iy in range(n_y)]
                    for fut in as_completed(futures):
                        iy, q_val, info = fut.result()
                        Q_seg_x0_list[ix, iy] = q_val
                        infos_by_iy[iy] = info

                # Geordnet ausgeben (iy=0..n_y-1), aber kompakt
                for iy in range(n_y):
                    info = infos_by_iy[iy]
                    if info is not None:
                        print(_fmt_segment_line(info))

            if stop_event.is_set():
                gs.t_end = t



            # Pushing the data ---------------------------------------------------------------------------------------------
            path_ref = geom.build_connection_path(geom.CP)
            pos_of = {seg: k for k, seg in enumerate(path_ref)}
            (x0, y0) = path_ref[0]
            (x_end, y_end) = path_ref[-1]

            # Calculation of the heat transfer coefficient
            # --- Inlet/Outlet ---
            T_inlet_air_mean = np.mean([cfg_grid[0][iy].T_a for iy in range(n_y)])
            T_outlet_air_mean = np.mean([cfg_grid[-1][iy].T_a for iy in range(n_y)])
            #
            ## --- Massflow air one row ---
            #m_dot_dry_y = cfg.m_dot / n_y
            #
            ## --- Refrigerant In/Out ---
            #T_ref_in = cfg_grid[x0][y0].T_ref
            T_ref_out = cfg_grid[x_end][y_end].T_ref

            #Calculating the COP
            W_comp = self.HP.W_comp
            Q_cond = self.HP.Q_cond
            Q_evap = self.HP.Q_evap

            EER = Q_evap/W_comp
            COP = Q_cond/W_comp

            mean_s_ft = np.mean([st_grid[ix][iy].s_ft
                                 for ix in range(n_x)
                                 for iy in range(n_y)])
            humid_l = np.array([seg[0].w_amb for seg in cfg_grid])

            p_ref_evap = cfg_grid[x0][y0].p_ref
            p_ref_cond = self.HP.p_ref_cond

            m_dot_air = input_cfg.m_dot
            v_in_air = np.mean([cfg_grid[0][iy].v_a for iy in range(n_y)])

            p1_suction = p_ref_evap
            h1_suction = cfg_grid[x_end][y_end].h_ref

            p2_discharge = p_ref_cond
            m_comp, h2_discharge = self.refrigerant.compressor_model(pi=p1_suction, hi=h1_suction, po=p2_discharge, RPM=self.HP.RPM(t))

            p3_cond_out = p_ref_cond
            h3_cond_out = self.HP.h_ref_cond[-1]

            p4_valve_out = p_ref_evap
            VPos = self.refrigerant.valve_controller(t)
            m_valve, h4_valve_out = self.refrigerant.valve_model(pi=p3_cond_out, hi=h3_cond_out, po=p4_valve_out, VPos=VPos)

            cycle_ph = [
                [float(p1_suction), float(h1_suction)],
                [float(p2_discharge), float(h2_discharge)],
                [float(p3_cond_out), float(h3_cond_out)],
                [float(p4_valve_out), float(h4_valve_out)],
            ]

            # einfache Zeitsignale im Speicher halten
            self.rec.push(t=t,
                          EER=EER,
                          COP=COP,
                          mean_s_ft=mean_s_ft,
                          m_dot_air = m_dot_air,
                          v_in_air = v_in_air,
                          T_out_air_mean=T_outlet_air_mean,
                          T_out_ref=T_ref_out,
                          p_ref_evap=p_ref_evap,
                          p_ref_cond=p_ref_cond,
                          humidity=humid_l,
                          cycle_ph=cycle_ph)

            # Push grid snapshot
            if it % gs.store_grid_every_x_it == 0 or t >= gs.t_end:
                print('Pushing grid state')
                self.rec.push_grid_snapshot(
                    t=t,
                    cfg_grid=cfg_grid,
                    st_grid=st_grid,
                    meta={"it": it}
                )


            # Dynamic models

            input_cfg.T_a = dynamic_models.T_a_profile(t, 20.0, 2.0, 200.0, 120.0)
            #input_cfg.w_amb = dynamic_models.w_amb_profile(t,input_cfg.T_a,input_cfg.p_a,0.0,0.85,120.0,10.0)

        # Updating the refrigerant state -------------------------------------------------------------------------------

            dt_step_n = gs.dt

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

                # Adaptiv time step:
                # parameters
                it_target = 10
                k = 0.5  # aggressiveness
                fac_min, fac_max = 0.5, 2.0  # limit per outer step
                dt_min, dt_max = 0.02, 5.0  # absolute bounds

                fac = (it_target / n_inner) ** k
                fac = max(fac_min, min(fac_max, fac))

                gs.dt *= fac
                gs.dt = max(dt_min, min(dt_max, gs.dt))

                print(f"Updating dt for the next time step to dt={gs.dt:.2f} s")

            t_iteration_end = time.perf_counter()
            time_iteration = t_iteration_end - t_iteration_start
            time_remaining_est = time_iteration * (gs.t_end - t) / gs.dt

            print(f"Cycle time for time step {it}: {time_iteration:.1f} s\n"
                  f"\033[94mTime remaining: {time_remaining_est / 60.0:.1f} min\033[0m")

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
    Erzeugt 2D-Listen (Grids) von CaseConfig und SimState:
    - erste Dimension: x-Richtung (Luftfluss) -> n_seg_l
    - zweite Dimension: y-Richtung (Reihen in Kältemittel-Richtung) -> n_seg_r

    Rückgabe:
        cfg_grid[ix][iy], st_grid[ix][iy]
    """

    n_x = int(geom.n_seg_l)   # Segmente in Luftflussrichtung
    n_y = int(geom.n_seg_r)   # Segmente in "Reihen"-Richtung

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