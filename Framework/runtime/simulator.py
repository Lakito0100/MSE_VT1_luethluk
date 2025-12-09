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


class Simulator:
    def __init__(self, geom, fields=("t")):
        self.rec = ResultRecorder(fields=fields, stream_path="sim_grid_log.jsonl")

        A_flow = (((geom.d_tube_a/2)-geom.tube_thickness)**2) * np.pi
        L = geom.l_tube()
        A_inner = (geom.d_tube_a-2*geom.tube_thickness)*np.pi*L
        V_wall = (((geom.d_tube_a/2)**2) - A_flow)*L

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
        self.refrigerant = Refrigerant(gp)


    def run(self, cfg, geom, gs, model):
        input_cfg = copy.deepcopy(cfg)
        cfg_grid, st_grid = build_segment_grids(base_cfg=cfg, geom=geom, gs=gs)
        n_x = len(cfg_grid)
        n_y = len(cfg_grid[0])

        model_e = model.Frostmodell_Edge()
        model_ft = model.Frostmodell_Finn_and_Tube()


        t = 0.0
        it = 1
        t0_start = time.perf_counter()

        while t <= gs.t_end:
            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.1f}' +
                  " s | " + f'{t / 60:.1f} min')

            Q_seg_x0_list = np.zeros((n_x, n_y), dtype=float)

            for ix in range(n_x):
                for iy in range(n_y):
                    cfg = cfg_grid[ix][iy]
                    st = st_grid[ix][iy]

                    gs.t = t
                    st.t = t
                    print(f'Segment [{ix},{iy}]')

        # Updating the edge state --------------------------------------------------------------------------------------

                    if ix == 0:
                        t0_edge = time.perf_counter()
                        try:
                            iter_e, res_T_e, res_w_e = model_e.New_edge_state_seg_at_90(cfg, geom, st, gs)
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

                    t0_ft = time.perf_counter()
                    try:
                        iter_ft, res_T_ft, res_w_ft = model_ft.New_finn_and_tube_state_seg(cfg, geom, st, gs)
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

                    m_s_seg = model_ft.segment_mass_flux_air_frost(cfg,geom,st,gs)
                    Q_seg_fs, Q_seg_x0 = model_ft.segment_heat_flux_air_frost(cfg,geom,st,gs)

                    m_dot_a = input_cfg.m_dot / n_y

                    if ix == 0:
                        T_out, w_out, p_out = self.air.propagate_inplace(input_cfg,cfg_grid[ix][iy],st.s_e[89],st,geom,
                                                                    m_dot_a,Q_seg_fs,m_s_seg)
                    else:
                        T_out, w_out, p_out = self.air.propagate_inplace(cfg_grid[ix-1][iy], cfg_grid[ix][iy],st.s_ft,st,geom,
                                                                    m_dot_a, Q_seg_fs, m_s_seg)

                    print('Done')

                    Q_seg_x0_list[ix,iy] = Q_seg_x0

        # Updating the refrigerant state -------------------------------------------------------------------------------

            print('Updating the refrigerant state for all segments...')

            self.refrigerant.update_all_segments(
                input_cfg,  # inlet BC
                cfg_grid,
                st_grid,
                geom,
                Q_seg_x0_list,  # this is Q_f per segment
                time=t,
                dt=gs.dt,
            )

            print('Done')

        # Pushing the data ---------------------------------------------------------------------------------------------

            # Beispiel für ein paar globale Grössen:
            T_outlet_air_mean = np.mean([cfg_grid[ix][-1].T_a
                                 for ix in range(n_x)])
            T_outlet_ref = cfg_grid[0][0].T_ref
            mean_s_ft = np.mean([st_grid[ix][iy].s_ft
                                 for ix in range(n_x)
                                 for iy in range(n_y)])

            # einfache Zeitsignale im Speicher halten
            self.rec.push(t=t,
                          mean_s_ft=mean_s_ft,
                          T_out_air_mean=T_outlet_air_mean,
                          T_out_ref=T_outlet_ref)

            # Push grid snapshot
            if it % gs.store_grid_every_x_it == 0 or t >= gs.t_end:
                print('Pushing grid state')
                self.rec.push_grid_snapshot(
                    t=t,
                    cfg_grid=cfg_grid,
                    st_grid=st_grid,
                    meta={"it": it}
                )

            it += 1
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

            # falls du schon zu Beginn Unterschiede willst (z.B. anderer p_ref
            # für jede Reihe, andere T_tube etc.), kannst du das hier tun:
            # if iy > 0: cfg_ij.T_tube += 0.5  # nur als Beispiel

            st_ij = SimState()
            init_fields(cfg_ij, st_ij, gs)

            cfg_grid[ix][iy] = cfg_ij
            st_grid[ix][iy]  = st_ij

    return cfg_grid, st_grid