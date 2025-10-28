from Framework_V2.runtime.state import SimState
from Framework_V2.runtime.recorder import ResultRecorder
from Framework_V2.runtime.initializer import init_fields

class Simulator:
    def __init__(self, fields=("t","x_frost")):
        self.rec = ResultRecorder(fields=fields)


    def run(self, cfg, geom, gs, model):
        st = SimState()
        init_fields(cfg, st, gs)
        edge_model = model.Frostmodell_Edge()

        t = 0

        while t <= cfg.t_end:
            st.t = t
            iter, res_T, res_w = edge_model.New_edge_state_seg(cfg, geom, st, gs)

            self.rec.push_from_state(st)
            t += cfg.dt

        return self.rec