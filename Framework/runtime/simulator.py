from Framework.runtime.state import SimState
from Framework.runtime.recorder import ResultRecorder
from Framework.runtime.initializer import init_fields
import time
from datetime import datetime


class Simulator:
    def __init__(self, fields=("t","x_frost")):
        self.rec = ResultRecorder(fields=fields)


    def run(self, cfg, geom, gs, model):
        st = SimState()
        init_fields(cfg, st, gs)
        model_e = model.Frostmodell_Edge()
        model_ft = model.Frostmodell_Finn_and_Tube()

        t = 0.0
        it = 1
        t0_start = time.perf_counter()

        while t <= gs.t_end:
            gs.t = t

            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.1f}' +
                  " s | " + f'{t / 60:.1f} min')

            #cfg.v_a = dynamic_models.velocity(t)
            #print(f"Geschwindigkeit angepasst auf: {cfg.v_a:.2f}")

            # Updating the edge state ----------------------------------------------------------------------------------

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

            # Updating the finn and tube state -------------------------------------------------------------------------

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

            self.rec.push_from_state(st)
            t += gs.dt
            it += 1
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

        return self.rec