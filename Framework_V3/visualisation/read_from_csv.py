import json, re
import numpy as np
import pandas as pd

def read_results_csv_json(path: str) -> dict:
    # Falls du leere Strings behalten willst statt NaN:
    # df = pd.read_csv(path, keep_default_na=False)
    df = pd.read_csv(path)

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
        exemplar = next((a for a in cells if isinstance(a, np.ndarray) and a.size > 0 and not np.isnan(a).all()), None)

        # Falls gar nichts Sinnvolles gefunden wurde: als 1D NaN-Vektor zurück
        if exemplar is None:
            out[col] = np.full((len(cells),), np.nan)
            continue

        ndim = exemplar.ndim
        shp  = exemplar.shape

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
