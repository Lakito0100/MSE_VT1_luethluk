from __future__ import annotations
import csv
from typing import Iterable, Mapping, Any

# --- ResultRecorder: minimaler Patch ---
import numpy as np
import copy

class ResultRecorder:
    def __init__(self, fields=("t", "s_e")):
        self.data = {k: [] for k in fields}
        self._n = 0

    def add_field(self, name: str) -> None:
        if name not in self.data:
            self.data[name] = [None] * self._n

    def push(self, **values):
        # neue Felder on-the-fly akzeptieren
        for k in values.keys():
            if k not in self.data:
                self.add_field(k)

        # WICHTIG: Mutables kopieren (v.a. NumPy-Arrays)
        def _snapshot(v):
            if isinstance(v, np.ndarray):
                return v.copy()
            # optional: tiefe Kopie für Listen/Dicts
            if isinstance(v, (list, dict)):
                return copy.deepcopy(v)
            return v

        for k in self.data.keys():
            self.data[k].append(_snapshot(values.get(k, None)))

        self._n += 1

    def push_from_state(self, state_obj, **extras):
        row = {}
        for k, v in vars(state_obj).items():
            if not k.startswith("_"):
                row[k] = v
        row.update(extras)
        self.push(**row)


    # ----------------- Exporte -----------------
    def to_csv(self, path: str) -> None:
        cols = list(self.data.keys())
        rows = zip(*[self.data[c] for c in cols])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
