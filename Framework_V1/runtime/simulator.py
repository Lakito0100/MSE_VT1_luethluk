from .state import SimState


class ResultRecorder:
    def __init__(self): self.data = {k: [] for k in ['t', 'fx']}
    def push(self, t, fx):
        self.data['t'].append(t)
        self.data['fx'].append(fx)

class Simulator:
    #def __init__(self):

    def run(self, cfg, model):
        st = SimState()
        rec = ResultRecorder()
        t = 0.001

        while t <= cfg.t_end:
            fx = model.testmodell_fx_at_t(con=cfg, t=t)
            rec.push(t, fx)
            t += cfg.dt

        return rec