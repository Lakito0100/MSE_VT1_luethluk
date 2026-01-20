from __future__ import annotations
import json
import pandas as pd
import numpy as np
import copy
import re
from dataclasses import is_dataclass, asdict


class ResultRecorder:
    """Record time series and grid snapshots during simulation."""
    def __init__(self, fields=("t",), stream_path: str | None = None):
        """
        fields       : fields to store as time series in self.data
        stream_path  : path to a text file. When set, grid snapshots are written
                       as JSON lines during the simulation.
        """
        self.data = {k: [] for k in fields}
        self.grid_snapshots = []
        self._n = 0

        self.stream_path = stream_path
        self._stream_file = None
        if stream_path is not None:
            # Text file in "write" mode, JSON per line
            self._stream_file = open(stream_path, "w", encoding="utf-8")


    def add_field(self, name: str) -> None:
        """Register a new time-series field."""
        if name not in self.data:
            self.data[name] = [None] * self._n

    def _snapshot(self, v):
        """Safely copy mutables (np.array, list, dict)."""
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, (list, dict)):
            return copy.deepcopy(v)
        return v

    def _to_serializable(self, obj):
        """
        Recursively convert to JSON-compatible structures:
        - dataclasses -> dict
        - np.ndarray  -> list
        - list/tuple  -> list
        - dict        -> dict (values recursively converted)
        - otherwise   -> return as-is
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
        """Append a new time-step row to the time series."""
        for k in values.keys():
            if k not in self.data:
                self.add_field(k)

        for k in self.data.keys():
            self.data[k].append(self._snapshot(values.get(k, None)))

        self._n += 1

    def push_from_state(self, state_obj, **extras):
        """Push values from a state object plus any extra fields."""
        row = {}
        for k, v in vars(state_obj).items():
            if not k.startswith("_"):
                row[k] = v
        row.update(extras)
        self.push(**row)

    def push_grid_snapshot(self, t, cfg_grid, st_grid, meta=None):
        """
        Save a snapshot of the full segment grid at time t.
        Note: cfg_grid and st_grid are deep-copied so later changes do not
        overwrite the snapshot.
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

        # 1) Store in RAM
        self.grid_snapshots.append(snap)

        # 2) Optionally write to stream file
        if self._stream_file is not None:
            serializable = self._to_serializable(snap)
            self._stream_file.write(json.dumps(serializable) + "\n")
            self._stream_file.flush()


    @staticmethod
    def to_csv(path: str, data):
        """Write time-series data (arrays) to a CSV with JSON-encoded cells."""
        # data: dict with arrays, e.g.:
        # t: (nt,), s_e: (nt, nθ), T_e: (nt, nr, nθ), ...
        if "t" not in data:
            raise ValueError("to_csv expects a field 't' in data.")

        nt = len(data["t"])
        df = pd.DataFrame({"t": data["t"]})

        def col_from_timeslices(arr):
            # arr: (nt, ...) -> list of JSON strings per time step
            return [json.dumps(np.asarray(arr[i]).tolist()) for i in range(nt)]

        for k, v in data.items():
            if k == "t":
                continue
            df[k] = col_from_timeslices(v)

        df.to_csv(f"{path}.csv", index=False)

    @classmethod
    def from_jsonl(cls, path: str, fields=("t",), rebuild_objects: bool = True) -> "ResultRecorder":
        """
        Load grid snapshots from a JSONL file and return a recorder so
        `results.grid_snapshots` can be used as usual.

        rebuild_objects=True:
            - cfg_grid cells become CaseConfig objects again
            - st_grid cells become SimState objects again (including np.ndarray)
        """
        rec = cls(fields=fields, stream_path=None)
        rec.grid_snapshots = cls.read_jsonl(f"{path}.jsonl", rebuild_objects=rebuild_objects)
        return rec

    @staticmethod
    def read_jsonl(path: str, rebuild_objects: bool = True) -> list[dict]:
        """
        Read a JSONL file (one JSON object per line) and return snapshot list.
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
                    raise ValueError(f"JSON parse error in {path} line {ln}: {e}") from e

                if rebuild_objects:
                    snap = ResultRecorder._rebuild_snapshot_objects(snap)

                snaps.append(snap)
        return snaps

    @staticmethod
    def _rebuild_snapshot_objects(snap: dict) -> dict:
        """
        Rebuild CaseConfig and SimState objects from serialized dict/list structures.
        """
        import numpy as _np
        from dataclasses import fields as _dc_fields

        # Local imports to avoid import cycles
        from Framework.core.config import CaseConfig
        from Framework.runtime.state import SimState

        case_field_names = {f.name for f in _dc_fields(CaseConfig)}
        sim_field_names  = {f.name for f in _dc_fields(SimState)}

        sim_array_fields = {
            "s_e", "T_e", "rho_e", "rho_a", "w_e",
            "T_ft", "rho_ft", "w_ft"
        }

        # Cast t to float (if stored as int/str)
        if "t" in snap and snap["t"] is not None:
            snap["t"] = float(snap["t"])

        # cfg_grid: 2D list -> 2D list of CaseConfig
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

        # st_grid: 2D list -> 2D list of SimState (with np.arrays)
        st_grid_raw = snap.get("st_grid", None)
        if isinstance(st_grid_raw, list):
            st_grid = []
            for row in st_grid_raw:
                new_row = []
                for cell in (row or []):
                    if isinstance(cell, dict):
                        filtered = {k: v for k, v in cell.items() if k in sim_field_names}

                        # Cleanly cast scalars
                        if "t" in filtered and filtered["t"] is not None:
                            filtered["t"] = float(filtered["t"])
                        if "s_ft" in filtered and filtered["s_ft"] is not None:
                            filtered["s_ft"] = float(filtered["s_ft"])

                        # Arrays back to numpy
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
        """Read CSV time-series data with JSON-encoded cells back into arrays."""
        # If you want to keep empty strings instead of NaN:
        # df = pd.read_csv(path, keep_default_na=False)
        df = pd.read_csv(f"{path}.csv")

        out = {}

        # Time axis
        if "t" in df.columns:
            out["t"] = pd.to_numeric(df["t"], errors="coerce").to_numpy()

        # Helper parser: a single cell -> np.array(...)
        def parse_cell(v):
            # numeric -> return as 0D array
            if isinstance(v, (int, float)) and not pd.isna(v):
                return np.array(v, dtype=float)
            if pd.isna(v):
                return np.array(np.nan)
            s = str(v).strip()
            if s == "" or s.lower() in ("nan", "none"):
                return np.array(np.nan)
            # this should be a JSON string
            try:
                obj = json.loads(s)
            except Exception:
                # Fallback: extract numbers via regex (if exported by another tool)
                nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
                if nums:
                    return np.array([float(x) for x in nums], dtype=float)
                raise
            return np.array(obj, dtype=float)

        for col in df.columns:
            if col == "t":
                continue

            cells = [parse_cell(v) for v in df[col].tolist()]

            # Find an example array (first sensible shape)
            exemplar = next((a for a in cells if isinstance(a, np.ndarray) and a.size > 0 and not np.isnan(a).all()),
                            None)

            # If nothing sensible was found: return 1D NaN vector
            if exemplar is None:
                out[col] = np.full((len(cells),), np.nan)
                continue

            ndim = exemplar.ndim
            shp = exemplar.shape

            # Normalize all cells to the same shape (pad with NaN if needed)
            normed = []
            for a in cells:
                if a.ndim == 0:
                    # Scalar (e.g., time series) OR NaN
                    if ndim == 0:
                        normed.append(np.array(a, dtype=float))
                    else:
                        # Pad to array shape
                        if np.isnan(a):
                            normed.append(np.full(shp, np.nan))
                        else:
                            # Scalar -> same value everywhere (rare, but robust)
                            normed.append(np.full(shp, float(a)))
                elif a.ndim == ndim and a.shape == shp:
                    normed.append(a.astype(float, copy=False))
                else:
                    # Mismatched shape -> pad/trim to target shape with NaN
                    b = np.full(shp, np.nan, dtype=float)
                    # Safe copy over the overlapping region
                    slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(shp, a.shape))
                    b[slices] = a[slices]
                    normed.append(b)

            # Stack according to dimension
            if ndim == 0:
                # Scalar per time -> (nt,)
                out[col] = np.array([float(x) if x.ndim == 0 else np.nan for x in normed], dtype=float)
            elif ndim == 1:
                # Vector per time -> (nt, nθ)
                out[col] = np.vstack(normed)
            elif ndim == 2:
                # Matrix per time -> (nt, nr, nθ)
                out[col] = np.stack(normed, axis=0)
            else:
                # Higher dimensions: return as object array if needed
                out[col] = np.array(normed, dtype=object)

        return out

    def close(self):
        """Close the optional stream file if open."""
        if self._stream_file is not None:
            self._stream_file.close()
            self._stream_file = None
