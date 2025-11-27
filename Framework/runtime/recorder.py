from __future__ import annotations
import json
import pandas as pd
import numpy as np
import copy
from dataclasses import is_dataclass, asdict


class ResultRecorder:
    def __init__(self, fields=("t",), stream_path: str | None = None):
        """
        fields       : Felder, die du in self.data als Zeitsignale speichern willst
        stream_path  : Pfad zu einer Textdatei. Wenn gesetzt, werden Grid-Snapshots
                       als JSON-Zeilen während der Simulation hineingeschrieben.
        """
        self.data = {k: [] for k in fields}
        self.grid_snapshots = []
        self._n = 0

        self.stream_path = stream_path
        self._stream_file = None
        if stream_path is not None:
            # Textdatei im "write"-Modus, Zeilenweise JSON
            self._stream_file = open(stream_path, "w", encoding="utf-8")


    def add_field(self, name: str) -> None:
        if name not in self.data:
            self.data[name] = [None] * self._n

    def _snapshot(self, v):
        """Mutables sauber kopieren (np.array, list, dict)."""
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, (list, dict)):
            return copy.deepcopy(v)
        return v

    def _to_serializable(self, obj):
        """
        Rekursiv alles in JSON-kompatible Strukturen überführen:
        - Dataclasses -> dict
        - np.ndarray  -> list
        - list/tuple  -> Liste
        - dict        -> dict (Werte rekursiv)
        - sonst       -> direkt zurückgeben
        """
        if is_dataclass(obj):
            d = asdict(obj)
            return {k: self._to_serializable(v) for k, v in d.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        return obj


    def push(self, **values):
        for k in values.keys():
            if k not in self.data:
                self.add_field(k)

        for k in self.data.keys():
            self.data[k].append(self._snapshot(values.get(k, None)))

        self._n += 1

    def push_from_state(self, state_obj, **extras):
        row = {}
        for k, v in vars(state_obj).items():
            if not k.startswith("_"):
                row[k] = v
        row.update(extras)
        self.push(**row)


    def _to_serializable(self, obj):
        """Deine Version – hier nur zur Vollständigkeit."""
        if is_dataclass(obj):
            d = asdict(obj)
            return {k: self._to_serializable(v) for k, v in d.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        return obj

    def push_grid_snapshot(self, t, cfg_grid, st_grid, meta=None):
        """
        Speichert einen Snapshot des gesamten Segment-Grids zum Zeitpunkt t.
        Wichtig: cfg_grid und st_grid werden *deepcopied*, damit spätere
        Änderungen den Snapshot nicht überschreiben.
        """
        cfg_copy = copy.deepcopy(cfg_grid)
        st_copy  = copy.deepcopy(st_grid)

        snap = {
            "t": float(t),
            "cfg_grid": cfg_copy,
            "st_grid": st_copy,
        }
        if meta is not None:
            snap["meta"] = copy.deepcopy(meta)

        # 1) im RAM speichern
        self.grid_snapshots.append(snap)

        # 2) optional in Stream-Datei schreiben
        if self._stream_file is not None:
            serializable = self._to_serializable(snap)
            self._stream_file.write(json.dumps(serializable) + "\n")
            self._stream_file.flush()


    @staticmethod
    def to_csv(path: str, data):
        # data: dict mit Arrays, z.B.:
        # t: (nt,), s_e: (nt, nθ), T_e: (nt, nr, nθ), ...
        if "t" not in data:
            raise ValueError("to_csv erwartet ein Feld 't' in data.")

        nt = len(data["t"])
        df = pd.DataFrame({"t": data["t"]})

        def col_from_timeslices(arr):
            # arr: (nt, ...) -> Liste mit JSON-Strings je Zeit
            return [json.dumps(np.asarray(arr[i]).tolist()) for i in range(nt)]

        for k, v in data.items():
            if k == "t":
                continue
            df[k] = col_from_timeslices(v)

        df.to_csv(path, index=False)

    def close(self):
        if self._stream_file is not None:
            self._stream_file.close()
            self._stream_file = None
