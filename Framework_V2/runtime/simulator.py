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

        t = 0

        while t <= cfg.t_end:
            st.t = t
            iter, res_T, res_w = model.New_edge_state_seg(cfg, geom, st, gs)
            print("Time Step: " + str(t) + " s \t Inner Iterations: " + str(iter) + " \t w: " + str(res_w) + " \t T: " + str(res_T))

            self.rec.push_from_state(st)
            t += cfg.dt

        return self.rec