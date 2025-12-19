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

        self.air = Air()
        self.refrigerant = Refrigerant(geom,self.gp,cfg,HP,geom.CP)


    def run(self, cfg, geom, gs, model):
        input_cfg = copy.deepcopy(cfg)
        cfg_grid, st_grid = build_segment_grids(base_cfg=cfg, geom=geom, gs=gs)
        n_x = len(cfg_grid)
        n_y = len(cfg_grid[0])

        model_e = model.Frostmodell_Edge()
        model_ft = model.Frostmodell_Finn_and_Tube()

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
                    st_up  = st
                else:
                    cfg_up = cfg_grid[ix - 1][iy]
                    st_up  = st_grid[ix - 1][iy]

                model_e_loc, model_ft_loc = _get_thread_models()

                info = {"ix": ix, "iy": iy, "edge": None, "ft": None, "air": None, "warn": [], "err": []}

                # ---------------- Frost condition (unverändert) ----------------
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
                if RH_air_at_wall >= 0.99 and gs.cal_frost:
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
                m_dot_a = input_cfg.m_dot / n_y

                if gs.cal_air:
                    if cfg.frost_condition:
                        if ix == 0:
                            T_out, w_out, p_out = self.air.propagate_inplace(
                                input_cfg, cfg, st.s_e[89], st, geom, m_dot_a, 0.0, 0.0, gs.dt
                            )
                        else:
                            m_s_seg = model_ft_loc.segment_mass_flux_air_frost(cfg_grid[ix - 1][iy], geom, st_up, gs)
                            Q_seg_fs, Q_seg_x0, Q_steady = model_ft_loc.segment_heat_flux_air_frost(cfg_grid[ix - 1][iy], geom, st_up, gs)
                            T_out, w_out, p_out = self.air.propagate_inplace(
                                cfg_grid[ix - 1][iy], cfg, st_up.s_ft, st_up, geom, m_dot_a, Q_seg_fs, m_s_seg, gs.dt
                            )

                        _, Q_seg_x0_n, _ = model_ft_loc.segment_heat_flux_air_frost(cfg, geom, st, gs)
                        q_for_list = Q_seg_x0_n
                    else:
                        if ix == 0:
                            T_out, w_out, p_out = self.air.propagate_inplace(
                                input_cfg, cfg, st.s_e[89], st, geom, m_dot_a, 0.0, 0.0, gs.dt
                            )
                        else:
                            Q_seg_fs, Q_seg_x0, Q_steady = model_ft_loc.segment_heat_flux_air_frost(cfg_grid[ix - 1][iy], geom, st_up, gs)
                            T_out, w_out, p_out = self.air.propagate_inplace(
                                cfg_grid[ix - 1][iy], cfg, st_up.s_ft, st_up, geom, m_dot_a, Q_steady, 0.0, gs.dt
                            )

                        _, _, Q_steady_n = model_ft_loc.segment_heat_flux_air_frost(cfg, geom, st, gs)
                        q_for_list = Q_steady_n

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

            # --- Massflow air one row ---
            m_dot_dry_y = cfg.m_dot / n_y

            # --- Refrigerant In/Out ---
            T_ref_in = cfg_grid[x0][y0].T_ref
            T_ref_out = cfg_grid[x_end][y_end].T_ref

            Q_air = np.full((n_x, n_y), np.nan, dtype=float)
            Q_ref = np.full((n_x, n_y), np.nan, dtype=float)
            dTlm = np.full((n_x, n_y), np.nan, dtype=float)
            U_from_air = np.full((n_x, n_y), np.nan, dtype=float)
            U_from_ref = np.full((n_x, n_y), np.nan, dtype=float)

            for ix in range(n_x):
                for iy in range(n_y):
                    if ix == 0:
                        T_air_in = input_cfg.T_a
                        w_air_in = input_cfg.w_amb
                    else:
                        T_air_in = cfg_grid[ix - 1][iy].T_a
                        w_air_in = cfg_grid[ix - 1][iy].w_amb

                    T_air_out = cfg_grid[ix][iy].T_a
                    w_air_out = cfg_grid[ix][iy].w_amb

                    h_air_in = HAPropsSI("H", "T", T_air_in + 273.15, "P", cfg_grid[ix][iy].p_a, "W", w_air_in)
                    h_air_out = HAPropsSI("H", "T", T_air_out + 273.15, "P", cfg_grid[ix][iy].p_a, "W", w_air_out)

                    Q_seg_air = m_dot_dry_y * (h_air_in - h_air_out)  # [W] positiv: Luft -> HX
                    Q_air[ix, iy] = Q_seg_air

                    k = pos_of.get((ix, iy), None)

                    if k == 0:
                        # Falls du eine Ref-Inlet-Temperatur im cfg hast, hier einsetzen:
                        T_ref_in = cfg_grid[ix][iy].T_ref
                    else:
                        (px, py) = path_ref[k - 1]
                        T_ref_in = cfg_grid[px][py].T_ref

                    T_ref_out = cfg_grid[ix][iy].T_ref
                    T_wall = cfg_grid[ix][iy].T_tube

                    area_ref = self.gp.A_inner
                    Q_seg_ref = self.refrigerant.h_int_corr() * area_ref * (T_wall - T_ref_out)
                    Q_ref[ix, iy] = Q_seg_ref

                    # --- lokales LMTD ---
                    dT1 = T_air_in - T_ref_out
                    dT2 = T_air_out - T_ref_in
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dT_lm = (dT1 - dT2) / np.log(dT1 / dT2)
                    dTlm[ix, iy] = dT_lm

                    area_air = geom.A_one_segment()

                    with np.errstate(divide="ignore", invalid="ignore"):
                        U_from_air[ix,iy] = Q_seg_air/(area_air*dT_lm)
                        U_from_ref[ix, iy] = Q_seg_ref / (area_ref * dT_lm)



            mean_s_ft = np.mean([st_grid[ix][iy].s_ft
                                 for ix in range(n_x)
                                 for iy in range(n_y)])
            humid_l = np.array([seg[0].w_amb for seg in cfg_grid])

            p_ref = cfg_grid[x0][y0].p_ref

            # einfache Zeitsignale im Speicher halten
            self.rec.push(t=t,
                          U_from_air=U_from_air,
                          U_from_ref=U_from_ref,
                          mean_s_ft=mean_s_ft,
                          T_out_air_mean=T_outlet_air_mean,
                          T_out_ref=T_ref_out,
                          p_ref = p_ref,
                          humidity=humid_l)

            # Push grid snapshot
            if it % gs.store_grid_every_x_it == 0 or t >= gs.t_end:
                print('Pushing grid state')
                self.rec.push_grid_snapshot(
                    t=t,
                    cfg_grid=cfg_grid,
                    st_grid=st_grid,
                    meta={"it": it}
                )

        # Updating the refrigerant state -------------------------------------------------------------------------------

            if gs.cal_ref:
                print('Calculating the refrigerant state for the next time step in all segments...')

                self.refrigerant.update_all_segments(
                    input_cfg,  # inlet BC
                    cfg_grid,
                    st_grid,
                    geom,
                    Q_seg_x0_list,  # this is Q_f per segment
                    time=t,
                    dt=gs.dt,
                )

            t_iteration_end = time.perf_counter()
            time_iteration = t_iteration_end - t_iteration_start
            time_remaining_est = time_iteration*(gs.t_end - t)/gs.dt

            print(f"Cycle time for time step {it}: {time_iteration:.1f} s\n"
                  f"\033[94mTime remaining: {time_remaining_est/60.0:.1f} min\033[0m")

            t += gs.dt
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