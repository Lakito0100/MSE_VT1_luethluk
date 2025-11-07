from __future__ import annotations
import csv
from typing import Iterable, Mapping, Any

class ResultRecorder:

    def __init__(self, fields: Iterable[str] = ("t", "x_frost")):
        self.data: dict[str, list[Any]] = {k: [] for k in fields}
        self._n = 0  # Anzahl Zeilen (Timesteps)

    def add_field(self, name: str) -> None:
        if name not in self.data:
            # rückwirkend mit None auffüllen
            self.data[name] = [None] * self._n

    def push(self, **values: Any) -> None:
        # neue Felder on-the-fly akzeptieren
        for k in values.keys():
            if k not in self.data:
                self.add_field(k)

        # Werte schreiben (für nicht gelieferte Felder None eintragen)
        for k in self.data.keys():
            self.data[k].append(values.get(k, None))

        self._n += 1

    def push_from_state(self, state_obj: Any, **extras: Any) -> None:
        row: dict[str, Any] = {}
        # alle public-Attribute des States lesen (ohne __dunder__)
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
