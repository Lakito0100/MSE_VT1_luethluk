from Framework_V2.runtime.state import SimState
from Framework_V2.runtime.recorder import ResultRecorder
from Framework_V2.runtime.initializer import init_fields

class Simulator:
    def __init__(self, fields=("t","x_frost")):
        self.rec = ResultRecorder(fields=fields)


    def run(self, cfg, geom, gs, model):
        st = SimState()
        init_fields(cfg, st, gs)
        model = model.Frostmodell_Edge()

        t = 0.0
        it = 1

        while t <= cfg.t_end:
            st.t = t
            iter, res_T, res_w = model.New_edge_state_seg_diverg_form(cfg, geom, st, gs)
            print("Time Step: " + str(it) +
                  "\t Time: " + f'{t:.1f}' +
                  " s | " + f'{t/60:.1f}' +
                  " min \t Inner Iterations: " + str(iter) +
                  " \t w: " + f'{res_w:.3e}' +
                  " \t T: " + f'{res_T:.3e}')

            self.rec.push_from_state(st)
            t += cfg.dt
            it += 1

        return self.rec