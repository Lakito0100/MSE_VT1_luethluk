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

        df.to_csv(f"{path}.csv", index=False)

    @classmethod
    def from_jsonl(cls, path: str, fields=("t",), rebuild_objects: bool = True) -> "ResultRecorder":
        """
        Lädt Grid-Snapshots aus einer JSONL-Datei und gibt einen Recorder zurück,
        so dass du anschließend `results.grid_snapshots` wie gewohnt verwenden kannst.

        rebuild_objects=True:
            - cfg_grid-Zellen werden wieder zu CaseConfig-Objekten
            - st_grid-Zellen werden wieder zu SimState-Objekten (inkl. np.ndarray)
        """
        rec = cls(fields=fields, stream_path=None)
        rec.grid_snapshots = cls.read_jsonl(f"{path}.jsonl", rebuild_objects=rebuild_objects)
        return rec

    @staticmethod
    def read_jsonl(path: str, rebuild_objects: bool = True) -> list[dict]:
        """
        Liest die JSONL-Datei (eine JSON-Struktur pro Zeile) und gibt eine Liste von Snapshots zurück.
        """
        snaps: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON-Parse-Fehler in {path} Zeile {ln}: {e}") from e

                if rebuild_objects:
                    snap = ResultRecorder._rebuild_snapshot_objects(snap)

                snaps.append(snap)
        return snaps

    @staticmethod
    def _rebuild_snapshot_objects(snap: dict) -> dict:
        """
        Baut aus den serialisierten Dict-/Listen-Strukturen wieder CaseConfig- und SimState-Objekte.
        """
        import numpy as _np
        from dataclasses import fields as _dc_fields

        # lokale Imports, damit keine Import-Zyklen entstehen
        from Framework.core.config import CaseConfig
        from Framework.runtime.state import SimState

        case_field_names = {f.name for f in _dc_fields(CaseConfig)}
        sim_field_names  = {f.name for f in _dc_fields(SimState)}

        sim_array_fields = {
            "s_e", "T_e", "rho_e", "rho_a", "w_e",
            "T_ft", "rho_ft", "w_ft"
        }

        # t auf float (falls als int/str gespeichert)
        if "t" in snap and snap["t"] is not None:
            snap["t"] = float(snap["t"])

        # cfg_grid: 2D Liste -> 2D Liste von CaseConfig
        cfg_grid_raw = snap.get("cfg_grid", None)
        if isinstance(cfg_grid_raw, list):
            cfg_grid = []
            for row in cfg_grid_raw:
                new_row = []
                for cell in (row or []):
                    if isinstance(cell, dict):
                        filtered = {k: v for k, v in cell.items() if k in case_field_names}
                        new_row.append(CaseConfig(**filtered))
                    else:
                        new_row.append(cell)
                cfg_grid.append(new_row)
            snap["cfg_grid"] = cfg_grid

        # st_grid: 2D Liste -> 2D Liste von SimState (mit np.arrays)
        st_grid_raw = snap.get("st_grid", None)
        if isinstance(st_grid_raw, list):
            st_grid = []
            for row in st_grid_raw:
                new_row = []
                for cell in (row or []):
                    if isinstance(cell, dict):
                        filtered = {k: v for k, v in cell.items() if k in sim_field_names}

                        # Scalars sauber casten
                        if "t" in filtered and filtered["t"] is not None:
                            filtered["t"] = float(filtered["t"])
                        if "s_ft" in filtered and filtered["s_ft"] is not None:
                            filtered["s_ft"] = float(filtered["s_ft"])

                        # Arrays zurück zu numpy
                        for k in sim_array_fields:
                            if k in filtered and filtered[k] is not None:
                                filtered[k] = _np.asarray(filtered[k], dtype=float)

                        new_row.append(SimState(**filtered))
                    else:
                        new_row.append(cell)
                st_grid.append(new_row)
            snap["st_grid"] = st_grid

        return snap

    @staticmethod
    def read_results_csv_json(path: str) -> dict:
        # Falls du leere Strings behalten willst statt NaN:
        # df = pd.read_csv(path, keep_default_na=False)
        df = pd.read_csv(f"{path}.csv")

        out = {}

        # Zeitachse
        if "t" in df.columns:
            out["t"] = pd.to_numeric(df["t"], errors="coerce").to_numpy()

        # Hilfsparser: einzelne Zelle -> np.array(...)
        def parse_cell(v):
            # numerisch -> als 0D-Array zurück
            if isinstance(v, (int, float)) and not pd.isna(v):
                return np.array(v, dtype=float)
            if pd.isna(v):
                return np.array(np.nan)
            s = str(v).strip()
            if s == "" or s.lower() in ("nan", "none"):
                return np.array(np.nan)
            # hier sollte es ein JSON-String sein
            try:
                obj = json.loads(s)
            except Exception:
                # Fallback: Zahlen via Regex extrahieren (falls jemand anderes exportiert hat)
                nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
                if nums:
                    return np.array([float(x) for x in nums], dtype=float)
                raise
            return np.array(obj, dtype=float)

        for col in df.columns:
            if col == "t":
                continue

            cells = [parse_cell(v) for v in df[col].tolist()]

            # Beispiel-Array finden (erste sinnvolle Form)
            exemplar = next((a for a in cells if isinstance(a, np.ndarray) and a.size > 0 and not np.isnan(a).all()),
                            None)

            # Falls gar nichts Sinnvolles gefunden wurde: als 1D NaN-Vektor zurück
            if exemplar is None:
                out[col] = np.full((len(cells),), np.nan)
                continue

            ndim = exemplar.ndim
            shp = exemplar.shape

            # Alle Zellen auf gleiche Form bringen (NaN auffüllen, falls nötig)
            normed = []
            for a in cells:
                if a.ndim == 0:
                    # Skalar (z. B. Zeitreihe) ODER NaN
                    if ndim == 0:
                        normed.append(np.array(a, dtype=float))
                    else:
                        # in Arrayform auffüllen
                        if np.isnan(a):
                            normed.append(np.full(shp, np.nan))
                        else:
                            # Skalar -> überall gleicher Wert (selten sinnvoll, aber robust)
                            normed.append(np.full(shp, float(a)))
                elif a.ndim == ndim and a.shape == shp:
                    normed.append(a.astype(float, copy=False))
                else:
                    # unpassende Form -> auf Ziel-Shape mit NaN polstern/kürzen
                    b = np.full(shp, np.nan, dtype=float)
                    # sichere Kopie über den überlappenden Bereich
                    slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(shp, a.shape))
                    b[slices] = a[slices]
                    normed.append(b)

            # Stapeln gemäß Dimension
            if ndim == 0:
                # Skalar pro Zeit -> (nt,)
                out[col] = np.array([float(x) if x.ndim == 0 else np.nan for x in normed], dtype=float)
            elif ndim == 1:
                # Vektor pro Zeit -> (nt, nθ)
                out[col] = np.vstack(normed)
            elif ndim == 2:
                # Matrix pro Zeit -> (nt, nr, nθ)
                out[col] = np.stack(normed, axis=0)
            else:
                # Höhere Dimensionen ggf. objektbasiert zurückgeben
                out[col] = np.array(normed, dtype=object)

        return out

    def close(self):
        if self._stream_file is not None:
            self._stream_file.close()
            self._stream_file = None
