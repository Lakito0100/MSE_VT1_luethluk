from CoolProp.HumidAirProp import HAPropsSI

from Framework.runtime.state import SimState
from Framework.runtime.recorder import ResultRecorder
from Framework.runtime.initializer import init_fields
from Framework.models.air_modell import Air
from Framework.models.refrigerant_modell import RefGeomParams, Refrigerant
import Framework.runtime.dynamic_models as dynamic_models

import time
from datetime import datetime
import copy
import numpy as np
import io
from contextlib import redirect_stdout


class Simulator:
    def __init__(self, geom, cfg, fields=("t")):
        self.rec = ResultRecorder(fields=fields, stream_path="sim_grid_log.jsonl")

        A_flow = (((geom.d_tube_a/2)-geom.tube_thickness)**2) * np.pi
        L = geom.l_tube()
        A_inner = (geom.d_tube_a-2*geom.tube_thickness)*np.pi*L
        V_wall = (((geom.d_tube_a/2)**2)*np.pi - A_flow)*L

        gp = RefGeomParams(
            A_flow  = A_flow,          # cross-section of one tube
            dx      = L,     # length per segment
            A_inner = A_inner,    # inner area per segment
            V_wall  = V_wall,     # wall volume per segment
            rho_wall= geom.rho_solid,
            c_wall  = geom.c_solid,
            dp_ref_seg = 0.0  # or just 0.0 for now
        )

        self.air = Air()
        self.refrigerant = Refrigerant(geom,gp,cfg,geom.CP)



    def steady_state_air_refrigerant(self, gs, geom, input_cfg, cfg_grid, st_grid, n_x, n_y, model_e, model_ft):
        t = 0.0
        st_it = 0
        st_cond_1 = 1000.0
        st_cond_2 = 1000.0
        T_old = 1000.0
        T_outlet_air_mean_old = 1000.0

        while (st_cond_1 > 1e-2 or st_cond_2 > 1e-2) and st_it <= 1000:
            st_it += 1
            Q_seg_x0_list_steady = np.zeros((n_x, n_y), dtype=float)

            for ix in range(n_x):
                for iy in range(n_y):
                    cfg = cfg_grid[ix][iy]
                    st = st_grid[ix][iy]

                    cfg_up = cfg_grid[ix - 1][iy]
                    st_up = st_grid[ix - 1][iy]

                    # Updating the air state ---------------------------------------------------------------------------------------
                    st.T_ft[...] = cfg.T_tube
                    st.T_e[:, :] = cfg.T_tube



                    m_dot_a = input_cfg.m_dot / n_y

                    if ix == 0:
                        T_out, w_out, p_out = self.air.propagate_inplace(input_cfg, cfg, st.s_e[89], st, geom,
                                                                         m_dot_a, 0.0, 0.0,gs.dt)
                    else:
                        m_s_seg = model_ft.segment_mass_flux_air_frost(cfg_grid[ix - 1][iy], geom, st_up, gs)
                        Q_seg_fs, Q_seg_x0, Q_steady = model_ft.segment_heat_flux_air_frost(cfg_grid[ix - 1][iy], geom, st_up, gs)
                        T_out, w_out, p_out = self.air.propagate_inplace(cfg_grid[ix - 1][iy], cfg, st_up.s_ft, st_up, geom,
                                                                         m_dot_a, Q_steady, m_s_seg,gs.dt)

                    Q_seg_fs_n, Q_seg_x0_n, Q_steady_n = model_ft.segment_heat_flux_air_frost(cfg, geom, st, gs)
                    Q_seg_x0_list_steady[ix, iy] = Q_steady_n    # For the steady state the heat flow is set equal

            # Updating the refrigerant state -------------------------------------------------------------------------------

            self.refrigerant.update_all_segments(
                input_cfg,  # inlet BC
                cfg_grid,
                st_grid,
                geom,
                Q_seg_x0_list_steady,  # this is Q_f per segment
                time=t,
                dt=gs.dt,
            )

            T_new = np.mean([cfg_grid[ix][iy].T_tube
                                 for ix in range(n_x)
                                 for iy in range(n_y)])
            T_outlet_air_mean_new = np.mean([cfg_grid[-1][iy].T_a
                                         for iy in range(n_y)])
            st_cond_1 = abs(T_new - T_old)/abs(T_old)
            st_cond_2 = abs(T_outlet_air_mean_new - T_outlet_air_mean_old)/abs(T_outlet_air_mean_old)
            T_old = T_new
            T_outlet_air_mean_old = T_outlet_air_mean_new

            print(f"Mean Tube T = {T_old} °C and Mean outlet air T = {T_outlet_air_mean_old} °C")

        return st_it, st_cond_1, st_cond_2


    def run(self, cfg, geom, gs, model):
        input_cfg = copy.deepcopy(cfg)
        cfg_grid, st_grid = build_segment_grids(base_cfg=cfg, geom=geom, gs=gs)
        n_x = len(cfg_grid)
        n_y = len(cfg_grid[0])

        model_e = model.Frostmodell_Edge()
        model_ft = model.Frostmodell_Finn_and_Tube()

        s_max = geom.fin_gap()/2.0

        t = 0.0
        it = 0
        t0_start = time.perf_counter()

        t_start_steady = time.perf_counter()
        print("Calculating the initial condition...")
        f = io.StringIO()
        with redirect_stdout(f):
            st_it, st_cond_1, st_cond_2 = self.steady_state_air_refrigerant(gs,geom,input_cfg,cfg_grid,st_grid,n_x,n_y,model_e,model_ft)
        t_end_steady = time.perf_counter()
        steady_time = t_end_steady - t_start_steady
        print(f"Initial conditions calculated with after {st_it} iterations \n"
              f"With residuals of {st_cond_1:.3e} and {st_cond_2:.3e} cycle time: {steady_time:.3f} s")

        while t <= gs.t_end:
            it += 1
            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.1f}' +
                  " s | " + f'{t / 60:.1f} min')

            Q_seg_x0_list = np.zeros((n_x, n_y), dtype=float)

            t_iteration_start = time.perf_counter()

            for ix in range(n_x):
                for iy in range(n_y):
                    cfg = cfg_grid[ix][iy]
                    st = st_grid[ix][iy]

                    cfg_up = cfg_grid[ix - 1][iy]
                    st_up = st_grid[ix - 1][iy]

                    gs.t = t
                    st.t = t
                    print(f'Segment [{ix},{iy}]')

                    try:
                        RH_air_at_wall = HAPropsSI("R",
                                                   "T", cfg.T_tube+273.15,
                                                   "P", cfg.p_a,
                                                   "W", cfg.w_amb)
                    except:
                        w_sat_wall = HAPropsSI("W",
                                               "T", cfg.T_tube + 273.15,
                                               "P", cfg.p_a,
                                               "R", 1.0)
                        RH_air_at_wall = cfg.w_amb / max(w_sat_wall, 1e-12)

                    RH_air_at_wall = max(0.0, min(1.0, RH_air_at_wall))
                    if RH_air_at_wall >= 0.98:
                        cfg.frost_condition = True

        # Updating the edge state --------------------------------------------------------------------------------------

                    if ix == 0 and cfg.frost_condition:
                        t0_edge = time.perf_counter()
                        try:
                            iter_e, res_T_e, res_w_e = model_e.New_edge_state_seg_at_90(cfg, geom, st, gs)
                            if st.s_e[89] >= s_max:
                                print(f"\033[31mThe frost in the edge segment {(ix,iy)} is blocking the air flow, ending the simulation.\033[0m")
                                gs.t_end = t
                        except Exception as e:
                            print("\033[31mThere was an error in the calculation for the new edge state, ending the simulation.\033[0m")
                            print(f'\033[31m{e}\033[0m')
                            gs.t_end = t
                        t1_edge = time.perf_counter()
                        edge_time = t1_edge - t0_edge
                        print("Edge Domain Inner Iterations: " + str(iter_e) +
                              " \t \t \t w: " + f'{res_w_e:.3e}' +
                              " \t T: " + f'{res_T_e:.3e}' +
                              " \t cycle time: " + f'{edge_time:.3f} s')

        # Updating the finn and tube state -----------------------------------------------------------------------------

                    if cfg.frost_condition:
                        t0_ft = time.perf_counter()
                        try:
                            iter_ft, res_T_ft, res_w_ft = model_ft.New_finn_and_tube_state_seg(cfg, geom, st, gs)
                            if st.s_ft >= s_max:
                                print(f"\033[31mThe frost in the segment {(ix, iy)} is blocking the air flow, ending the simulation.\033[0m")
                                gs.t_end = t
                        except Exception as e:
                            print("\033[31mThere was an error in the calculation for the new finn and tube state, ending the simulation.\033[0m")
                            print(f'\033[31m{e}\033[0m')
                            gs.t_end = t
                        t1_ft = time.perf_counter()
                        ft_time = t1_ft - t0_ft
                        print("Finn & Tube Domain Inner Iterations: " + str(iter_ft) +
                              " \t w: " + f'{res_w_ft:.3e}' +
                              " \t T: " + f'{res_T_ft:.3e}' +
                              " \t cycle time: " + f'{ft_time:.3f} s'
                              )

        # Updating the air state ---------------------------------------------------------------------------------------

                    print(f'Updating the air state for this segment [{ix},{iy}]...')

                    m_dot_a = input_cfg.m_dot / n_y

                    if cfg.frost_condition == True:
                        if ix == 0:
                            T_out, w_out, p_out = self.air.propagate_inplace(input_cfg,cfg,st.s_e[89],st,geom,
                                                                        m_dot_a,0.0,0.0,gs.dt)
                        else:
                            m_s_seg = model_ft.segment_mass_flux_air_frost(cfg_grid[ix-1][iy], geom, st_up, gs)
                            Q_seg_fs, Q_seg_x0, Q_steady = model_ft.segment_heat_flux_air_frost(cfg_grid[ix-1][iy], geom, st_up, gs)
                            T_out, w_out, p_out = self.air.propagate_inplace(cfg_grid[ix-1][iy], cfg,st_up.s_ft,st_up,geom,
                                                                        m_dot_a, Q_seg_fs, m_s_seg,gs.dt)

                        Q_seg_fs_n, Q_seg_x0_n, Q_steady_n = model_ft.segment_heat_flux_air_frost(cfg, geom, st, gs)
                        Q_seg_x0_list[ix, iy] = Q_seg_x0_n

                    else:
                        if ix == 0:
                            T_out, w_out, p_out = self.air.propagate_inplace(input_cfg,cfg,st.s_e[89],st,geom,
                                                                        m_dot_a,0.0,0.0,gs.dt)
                        else:
                            Q_seg_fs, Q_seg_x0, Q_steady = model_ft.segment_heat_flux_air_frost(cfg_grid[ix-1][iy], geom, st_up, gs)
                            T_out, w_out, p_out = self.air.propagate_inplace(cfg_grid[ix-1][iy], cfg,st_up.s_ft,st_up,geom,
                                                                        m_dot_a, Q_steady, 0.0,gs.dt)

                        Q_seg_fs_n, Q_seg_x0_n, Q_steady_n = model_ft.segment_heat_flux_air_frost(cfg, geom, st, gs)
                        Q_seg_x0_list[ix, iy] = Q_steady_n

                    print("Air Domain Results: " +
                          " \t new T: " + f'{T_out:.2f} °C' +
                          " \t new w: " + f'{w_out:.3e} kg/kg' +
                          " \t new P: " + f'{p_out:.3e} Pa'
                          )





        # Pushing the data ---------------------------------------------------------------------------------------------
            # Beispiel für ein paar globale Grössen:
            path_ref = geom.build_connection_path(geom.CP)
            (x_end, y_end) = path_ref[-1]

            T_outlet_air_mean = np.mean([cfg_grid[-1][iy].T_a
                                 for iy in range(n_y)])
            T_outlet_ref = cfg_grid[x_end][y_end].T_ref
            mean_s_ft = np.mean([st_grid[ix][iy].s_ft
                                 for ix in range(n_x)
                                 for iy in range(n_y)])
            humid_l = np.array([seg[0].w_amb for seg in cfg_grid])

            # einfache Zeitsignale im Speicher halten
            self.rec.push(t=t,
                          mean_s_ft=mean_s_ft,
                          T_out_air_mean=T_outlet_air_mean,
                          T_out_ref=T_outlet_ref,
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