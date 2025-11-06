import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

def read_xy_from_csv(csv_path, x_col, y_col, x_idx=None, y_idx=None, dropna=True, sort_by_x=True):
    """
    Liest zwei Spalten aus einer CSV und gibt skalare x- und y-Werte als 1D-Arrays zurück.
    Spalten dürfen Zahlen, Vektoren oder (N,M)-Arrays als Text enthalten (z. B. '[-10, -9.9, ...]'
    oder mehrzeiliges '[[...],[...],...]'). Mit x_idx / y_idx wählst du den Eintrag aus.

    Parameters
    ----------
    csv_path : str | Path
        Pfad zur CSV-Datei.
    x_col, y_col : str
        Spaltennamen ('t', 's_e', 'T_e', 'rho_e', 'rho_a', 'w_e', ...).
    x_idx, y_idx : int | tuple[int,int] | None
        Index für Vektor (int) oder 2D-Array (tuple). None bedeutet:
        - Wenn der Zellenwert ein Skalar ist -> direkt nutzen
        - Wenn die Zelle ein Array mit Größe >1 ist -> Fehler (bitte Index angeben)

    dropna : bool
        Fehlende Werte entfernen.
    sort_by_x : bool
        Nach x aufsteigend sortieren.

    Returns
    -------
    x : np.ndarray, y : np.ndarray
        1D-Arrays gleicher Länge, zeilenweise aus der CSV extrahiert.
    """
    # CSV robust einlesen (handhabt auch mehrzeilige, gequotete Felder)
    df = pd.read_csv(csv_path, sep=None, engine="python")

    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht gefunden. Verfügbar: {list(df.columns)}")

    def _parse_cell(cell):
        # Zahl?
        if isinstance(cell, (int, float, np.number)) or cell is None:
            return cell
        # Versuch: Text -> Python-Objekt
        if isinstance(cell, str):
            s = cell.strip()
            # Leere Strings als NaN behandeln
            if s == "":
                return np.nan
            try:
                obj = ast.literal_eval(s)
            except Exception:
                # Vielleicht ist es eine einfache Zahl als Text
                try:
                    return float(s)
                except Exception:
                    # Unparsbar -> NaN
                    return np.nan
            # In Array konvertieren, falls Liste/Liste-von-Listen
            if isinstance(obj, (list, tuple)):
                try:
                    return np.array(obj)
                except Exception:
                    return obj
            # Falls bereits Zahl
            if isinstance(obj, (int, float, np.number)):
                return float(obj)
            return obj
        return cell  # unbekannter Typ

    def _extract_scalar(val, idx, colname):
        # None / NaN
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        # Array?
        if isinstance(val, np.ndarray):
            if idx is None:
                if val.size == 1:
                    return float(val.reshape(-1)[0])
                else:
                    raise ValueError(
                        f"Spalte '{colname}' enthält Arrays mit Größe {val.shape}, "
                        f"aber es wurde kein Index übergeben (idx für '{colname}' setzen!)."
                    )
            # Indizieren (int für 1D, tuple für 2D)
            try:
                picked = val[idx]
            except Exception as e:
                raise IndexError(
                    f"Ungültiger Index {idx} für Wert aus Spalte '{colname}' mit Shape {val.shape}"
                ) from e
            # Ergebnis muss skalar sein
            if isinstance(picked, np.ndarray):
                if picked.size != 1:
                    raise ValueError(
                        f"Index {idx} in '{colname}' liefert kein Skalar (Shape {picked.shape}). "
                        f"Bitte Index anpassen."
                    )
                picked = float(picked.reshape(-1)[0])
            return float(picked)
        # Skalar?
        if isinstance(val, (int, float, np.number)):
            if idx is not None:
                # User hat Index angegeben, aber Zelle ist skalar -> ignorieren wir mit Hinweis?
                # Wir nehmen einfach den Skalar.
                pass
            return float(val)
        # Fallback: nochmal versuchen, in float umzuwandeln
        try:
            return float(val)
        except Exception:
            return np.nan

    # Spalten parsen
    x_parsed = df[x_col].map(_parse_cell)
    y_parsed = df[y_col].map(_parse_cell)

    # Skalar extrahieren (ggf. mit Index)
    x_vals = x_parsed.map(lambda v: _extract_scalar(v, x_idx, x_col)).to_numpy()
    y_vals = y_parsed.map(lambda v: _extract_scalar(v, y_idx, y_col)).to_numpy()

    # DataFrame für optionales Aufräumen
    out = pd.DataFrame({x_col: x_vals, y_col: y_vals})

    if dropna:
        out = out.dropna(subset=[x_col, y_col])

    if sort_by_x:
        out = out.sort_values(by=x_col)

    return out[x_col].to_numpy(), out[y_col].to_numpy()



def plot_xy(
    x,
    y,
    *,
    r_idx=None,
    theta_idx=None,
    y_axis=None,
    title=None,
    xlabel=None,
    ylabel=None,
    save_path=None,
    show=True,
    marker='o',
    line_kwargs=None
):
    """
    Plot x vs y where:
      - x is 1D
      - y can be 1D, 2D (e.g., [r, theta] or [time, r]...), or 3D (e.g., [time, r, theta]).
    For 3D, one axis of y must match len(x) (usually time); the other two are (r, theta).
    Fix the spatial position with r_idx and theta_idx.

    Parameters
    ----------
    x : array-like (1D)
    y : array-like (1D/2D/3D)
    r_idx : int, optional
        Radius index to fix when y has an r dimension.
    theta_idx : int, optional
        Angle index to fix when y has a theta dimension.
    y_axis : int, optional
        Axis of y that varies with x (length == len(x)). If None, inferred.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    n = len(x)

    # Simple 1D -> 1D case
    if y.ndim == 1:
        if len(y) != n:
            raise ValueError(f"len(y)={len(y)} != len(x)={n}")
        y_line = y

    else:
        # Find axis in y matching len(x)
        if y_axis is None:
            candidates = [ax for ax, L in enumerate(y.shape) if L == n]
            if not candidates:
                raise ValueError(
                    f"No axis in y matches len(x)={n}; y.shape={y.shape}. "
                    "Specify `y_axis` or pass a matching array."
                )
            y_axis = candidates[0]
        if y.shape[y_axis] != n:
            raise ValueError(
                f"Chosen y_axis={y_axis} length {y.shape[y_axis]} != len(x)={n}"
            )

        # Remaining axes are the "spatial" ones we need to fix
        other_axes = [ax for ax in range(y.ndim) if ax != y_axis]

        if y.ndim == 2:
            # Only one other axis; treat it like either r OR theta.
            # If both r and theta are conceptually present, this array
            # represents just one of them; you fix it with r_idx or theta_idx.
            if r_idx is None and theta_idx is None:
                # default to 0
                fixed_idx = 0
            else:
                # Prefer whichever is provided
                fixed_idx = r_idx if r_idx is not None else theta_idx
            ax_fixed = other_axes[0]
            if not (0 <= fixed_idx < y.shape[ax_fixed]):
                raise IndexError(
                    f"Index {fixed_idx} out of bounds for axis {ax_fixed} with size {y.shape[ax_fixed]}"
                )
            slicer = [slice(None)] * y.ndim
            slicer[ax_fixed] = fixed_idx
            y_line = np.asarray(y[tuple(slicer)])
            y_line = np.squeeze(y_line)
            if y_line.ndim != 1:
                raise RuntimeError(f"Expected 1D after slicing; got shape {y_line.shape}")

        elif y.ndim == 3:
            # Two spatial axes to fix: interpret them as (r, theta) in any order
            if r_idx is None:
                r_idx = 0
            if theta_idx is None:
                theta_idx = 0

            # Build slicer: free along y_axis, fix the other two
            slicer = []
            fixed_map = {}
            # Assign the two non-y_axis dims deterministicly:
            # other_axes[0] -> r, other_axes[1] -> theta
            r_axis, theta_axis = other_axes

            # bounds
            if not (0 <= r_idx < y.shape[r_axis]):
                raise IndexError(
                    f"r_idx {r_idx} out of bounds for r_axis {r_axis} with size {y.shape[r_axis]}"
                )
            if not (0 <= theta_idx < y.shape[theta_axis]):
                raise IndexError(
                    f"theta_idx {theta_idx} out of bounds for theta_axis {theta_axis} with size {y.shape[theta_axis]}"
                )

            for ax_i in range(y.ndim):
                if ax_i == y_axis:
                    slicer.append(slice(None))
                elif ax_i == r_axis:
                    slicer.append(r_idx)
                elif ax_i == theta_axis:
                    slicer.append(theta_idx)
                else:
                    raise RuntimeError("Unexpected axis layout.")

            y_line = np.asarray(y[tuple(slicer)])
            y_line = np.squeeze(y_line)
            if y_line.ndim != 1:
                raise RuntimeError(f"Expected 1D after slicing; got shape {y_line.shape}")

        else:
            raise ValueError(
                f"y with ndim={y.ndim} not supported. Use 1D, 2D, or 3D (time, r, theta)."
            )

        if len(y_line) != n:
            raise RuntimeError(
                f"After slicing, len(y_line)={len(y_line)} != len(x)={n}"
            )

    # Plot
    fig, ax = plt.subplots()
    plot_kwargs = dict(marker=marker)
    if line_kwargs:
        plot_kwargs.update(line_kwargs)
    ax.plot(x, y_line, **plot_kwargs)

    ax.set_xlabel(xlabel if xlabel else "x")
    ax.set_ylabel(ylabel if ylabel else "y")
    if title:
        ax.set_title(title)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def plot_spatial_slice(
    y,
    *,
    vary="r",              # "r" or "theta"
    t_idx=0,               # time index (ignored if y is 2D)
    r_idx=None,            # fixed r when varying theta
    theta_idx=None,        # fixed theta when varying r
    r_coords=None,         # optional 1D array of r positions (for x-axis)
    theta_coords=None,     # optional 1D array of theta positions (for x-axis)
    title=None,
    xlabel=None,
    ylabel=None,
    save_path=None,
    show=True,
    marker='o',
    line_kwargs=None
):
    """
    Plot a spatial profile through r or theta at one specified time step.

    Parameters
    ----------
    y : array-like
        Field with shape (nt, nr, ntheta) or (nr, ntheta).
    vary : {"r","theta"}
        Which spatial direction to plot along.
    t_idx : int
        Time index to slice at (ignored if y is 2D).
    r_idx : int or None
        If vary == "theta", fix radius with r_idx.
    theta_idx : int or None
        If vary == "r", fix angle with theta_idx.
    r_coords, theta_coords : array-like or None
        Coordinate vectors for pretty x-axis (otherwise indices are used).
    """
    y = np.asarray(y)

    # Infer layout
    if y.ndim == 3:
        # Assume (t, r, theta)
        nt, nr, ntheta = y.shape
        if not (0 <= t_idx < nt):
            raise IndexError(f"t_idx {t_idx} out of bounds for time axis with size {nt}")
        y_t = y[t_idx]
    elif y.ndim == 2:
        nr, ntheta = y.shape
        y_t = y  # time already collapsed
    else:
        raise ValueError(f"Expected y with ndim 2 or 3, got shape {y.shape}")

    # Defaults for missing indices
    if vary == "r":
        if theta_idx is None:
            theta_idx = 0
        if not (0 <= theta_idx < ntheta):
            raise IndexError(f"theta_idx {theta_idx} out of bounds for theta axis size {ntheta}")
        y_line = y_t[:, theta_idx]            # length nr
        x_vals = np.asarray(r_coords) if r_coords is not None else np.arange(nr)
        xlab_default = "r"
    elif vary == "theta":
        if r_idx is None:
            r_idx = 0
        if not (0 <= r_idx < nr):
            raise IndexError(f"r_idx {r_idx} out of bounds for r axis size {nr}")
        y_line = y_t[r_idx, :]                # length ntheta
        x_vals = np.asarray(theta_coords) if theta_coords is not None else np.arange(ntheta)
        xlab_default = "θ"
    else:
        raise ValueError("vary must be 'r' or 'theta'")

    if x_vals.shape[0] != y_line.shape[0]:
        raise ValueError(
            f"x length {x_vals.shape[0]} does not match y length {y_line.shape[0]} "
            f"for vary='{vary}'."
        )

    # Plot
    fig, ax = plt.subplots()
    kw = dict(marker=marker)
    if line_kwargs:
        kw.update(line_kwargs)
    ax.plot(x_vals, y_line, **kw)

    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlab_default if r_coords is None and theta_coords is None
                      else (xlab_default + " coordinate"))

    ax.set_ylabel(ylabel if ylabel else "value")

    if title:
        ax.set_title(title)
    else:
        if y.ndim == 3:
            ax.set_title(f"Spatial slice at t_idx={t_idx} (vary={vary})")
        else:
            ax.set_title(f"Spatial slice (vary={vary})")

    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def plot_finned_tube_side(he):
    L   = he.l_rohr()
    D   = he.d_tube_a
    N   = int(he.n_fin)
    Lf  = he.l_fin
    tf  = he.fin_pitch

    pitch = L/(N-1) if N > 1 else L
    Hfin = D + 2*Lf  # gesamte Ausladung (Breite) der Lamelle

    fig, ax = plt.subplots()

    # Rohr (jetzt vertikal: Breite=D, Höhe=L)
    tube = Rectangle((-D/2, 0), D, L, fill=False, linewidth=2)
    ax.add_patch(tube)

    # Lamellen (horizontale Rechtecke, entlang y verteilt)
    for i in range(N):
        y = i * pitch
        fin = Rectangle((-Hfin/2, y - tf/2), Hfin, tf, fill=False, linewidth=1)
        ax.add_patch(fin)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1*(Hfin/2), 1.1*(Hfin/2))  # Breite
    ax.set_ylim(-0.05*L, 1.05*L)      # Länge
    ax.set_xlabel("Breite [m]")
    ax.set_ylabel("Länge [m]")
    ax.set_title("Lamellenverdampfer")
    ax.grid(True, alpha=0.3)
    plt.show()