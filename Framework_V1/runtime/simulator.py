from .state import SimState
from .recorder import ResultRecorder

class Simulator:
    def __init__(self, fields=("t","x_frost")):
        self.rec = ResultRecorder(fields=fields)

    def run(self, cfg, model):
        st = SimState()
        t = 0.001

        while t <= cfg.t_end:
            st.t = t
            fx = model.testmodell_fx_at_t(con=cfg, t=t)
            st.x_frost = fx

            self.rec.push_from_state(st)
            t += cfg.dt

        return self.rec