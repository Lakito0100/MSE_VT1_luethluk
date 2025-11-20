from Framework_V3.runtime.state import SimState
from Framework_V3.runtime.recorder import ResultRecorder
from Framework_V3.runtime.initializer import init_fields
import time
from Framework_V3.runtime import dynamic_models

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

        while t <= cfg.t_end:
            st.t = t

            #cfg.v_a = dynamic_models.velocity(t)
            #print(f"Geschwindigkeit angepasst auf: {cfg.v_a:.2f}")

            t0_edge = time.perf_counter()
            iter_e, res_T_e, res_w_e = model_e.New_edge_state_seg_at_90(cfg, geom, st, gs)
            t1_edge = time.perf_counter()
            edge_time = t1_edge - t0_edge

            t0_ft = time.perf_counter()
            iter_ft, res_T_ft, res_w_ft = model_ft.New_finn_and_tube_state_seg(cfg, geom, st, gs)
            t1_ft = time.perf_counter()
            ft_time = t1_ft - t0_ft

            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.1f}' +
                  " s | " + f'{t/60:.1f}' +
                  " min \nEdge Domain Inner Iterations: " + str(iter_e) +
                  " \t \t \t w: " + f'{res_w_e:.3e}' +
                  " \t T: " + f'{res_T_e:.3e}' +
                  " \t cycle time: " + f'{edge_time:.3f} s' +

                  "\nFinn & Tube Domain Inner Iterations: " + str(iter_ft) +
                  " \t w: " + f'{res_w_ft:.3e}' +
                  " \t T: " + f'{res_T_ft:.3e}' +
                  " \t cycle time: " + f'{ft_time:.3f} s' +
                  "\n--------------------------------------------------------------------------------------------------"
                  )

            self.rec.push_from_state(st)
            t += cfg.dt
            it += 1

        return self.rec