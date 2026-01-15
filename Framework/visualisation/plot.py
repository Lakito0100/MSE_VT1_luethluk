import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from CoolProp.CoolProp import PropsSI

def extract_segment_timeseries_from_snapshots(grid_snapshots, field, ix, iy):
    """
    Holt aus einer Liste von Grid-Snapshots (ResultRecorder.grid_snapshots)
    die Zeitreihe eines Feldes für ein bestimmtes Segment [ix, iy].

    Parameter
    ---------
    grid_snapshots : list[dict]
        Liste von Snapshots, wie sie der Simulator in ResultRecorder.grid_snapshots ablegt.
        Jeder Eintrag hat mindestens Keys "t" und "st_grid".
    field : str
        Name des Attributs im SimState (z.B. "s_e", "s_ft", "T_ft", ...)
    ix, iy : int
        Segment-Koordinaten (x-Richtung: Luftfluss, y-Richtung: Reihen).

    Rückgabe
    --------
    t : np.ndarray, shape (nt,)
        Zeitvektor
    y : np.ndarray
        Zeitreihe des gewünschten Feldes, z.B.:
        - (nt,)          für Skalare
        - (nt, ntheta)   für s_e
        - (nt, nr, ntheta) für T_e usw.
    """
    t_list = []
    y_list = []

    for snap in grid_snapshots:
        # Zeit aus Snapshot
        t_list.append(snap["t"])

        # SimState dieses Segments
        st_ij = snap["st_grid"][ix][iy]

        # Feld aus dem SimState holen
        try:
            val = getattr(st_ij, field)
        except AttributeError as e:
            raise AttributeError(
                f"SimState hat kein Attribut '{field}'. "
                f"Verfügbare: {list(vars(st_ij).keys())}"
            ) from e

        # in Array wandeln (damit Shapes vergleichbar sind)
        y_list.append(np.asarray(val))

    t = np.array(t_list)

    # Versuche, entlang der Zeitachse zu stacken
    try:
        y = np.stack(y_list, axis=0)
    except ValueError:
        # Fallback, falls die Shapes zwischenzeitlich variieren
        # (z.B. wachsender Frost mit anderem Grid)
        # Dann lieber als Objekt-Array zurückgeben
        y = np.array(y_list, dtype=object)

    return t, y


def extract_segment_scalar_from_snapshots(grid_snapshots, field, ix, iy):
    """
    Wie oben, aber für reine Skalarwerte (z.B. s_ft wenn 0D) – Komfortfunktion.
    """
    t, y = extract_segment_timeseries_from_snapshots(grid_snapshots, field, ix, iy)
    # falls y shape (nt, 1) oder so hat, flachziehen:
    return t, np.asarray(y).reshape(len(t))

def plot_any(
    kind,
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
    line_kwargs=None,
    label=None,          # für 1 Kurve
    labels=None,         # für mehrere Kurven (z.B. 4 Spalten)
    ylimitter=None
):

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")

    # Y definieren
    match kind:
        case "time vs any":
            y_line = y

        case "time vs 1D(r)":
            if r_idx is None:
                raise ValueError("r_idx muss angegeben werde.")
            y_line = []
            for t in y:
                y_line.append(t[r_idx])

        case "time vs 1D(theta)":
            if theta_idx is None:
                raise ValueError("theta_idx muss angegeben werde.")
            y_line = []
            for t in y:
                y_line.append(t[theta_idx])

        case "time vs 2D(r,theta)":
            if r_idx is None:
                raise ValueError("r_idx muss angegeben werde.")
            if theta_idx is None:
                raise ValueError("theta_idx muss angegeben werde.")
            y_line = []
            for t in y:
                y_line.append(t[r_idx, theta_idx])

        case _:
            raise ValueError(f"Unknown kind: {kind}")

    y_line = np.asarray(y_line)

    # Plot
    fig, ax = plt.subplots()
    plot_kwargs = dict(marker=marker)
    if line_kwargs:
        plot_kwargs.update(line_kwargs)

    # Wenn y_line 1D ist → eine Kurve
    # Wenn y_line 2D ist → mehrere Kurven (z.B. 4 Spalten)
    if y_line.ndim == 1:
        if label is not None:
            plot_kwargs["label"] = label
        ax.plot(x, y_line, **plot_kwargs)
    elif y_line.ndim == 2:
        # Matplotlib macht automatisch mehrere Kurven
        # (jede Spalte von y_line ist eine Kurve)
        ax.plot(x, y_line, **plot_kwargs)
        # Labels später per ax.legend(labels) setzen
    else:
        raise ValueError(f"y_line must be 1D or 2D, got shape {y_line.shape}")

    if ylimitter is not None:
        ax.set_ylim(ylimitter[0], ylimitter[1])

    ax.set_xlabel(xlabel if xlabel else "x")
    ax.set_ylabel(ylabel if ylabel else "y")
    if title:
        ax.set_title(title)
    ax.grid(True)

    # Legend-Logik
    if y_line.ndim == 1:
        if label is not None:
            ax.legend()
    else:  # mehrere Kurven
        if labels is not None:
            ax.legend(labels)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
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
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def plot_finned_tube_side(he):
    """
    Zeichnet Seitenansicht: vertikales Rohr (Rechteck) mit horizontalen Finnen (Rechtecke).
    Erwartete Attribute in `he`:
      - n_fin (Anzahl Finnen)
      - l_fin (Lamellenausladung je Seite; Gesamtausladung = d_tube_a + 2*l_fin)
      - d_fin (Lamellendicke)
      - fin_pitch (Center-to-Center-Abstand der Finnen)
      - d_tube_a (Außendurchmesser Rohr)
      - tube_thickness (Wanddicke Rohr)
    """
    # Geometrie auslesen
    N   = int(he.n_fin)
    Lf  = float(he.l_fin)         # Ausladung je Seite
    t_f = float(he.fin_thickness)         # Finnen-DICKE
    p   = float(he.fin_pitch_cc)     # Finnen-PITCH (center-to-center)
    D   = float(he.d_tube_a)
    t_t = float(he.tube_thickness)

    if N <= 0:
        raise ValueError("n_fin muss > 0 sein.")

    # Gesamtlänge in y-Richtung: von der ersten Finne (unten) bis zur letzten (oben)
    # Annahme: Pitch ist Center-to-Center, d.h. erste Finnenmitte bei y = t_f/2
    # => L = t_f + (N-1)*p, und Finnen decken genau [0, L] ab.
    L = t_f + (N - 1) * p

    # Gesamtausladung (Breite) der Lamelle in x-Richtung
    Hfin = D + 2 * Lf

    fig, ax = plt.subplots()

    # Rohr (Seitenansicht als Rechteck): Breite = D, Höhe = L
    # Zentriere das Rohr bei x=0, reiche über y in [0, L]
    tube = Rectangle((-D/2, 0.0), D, L, fill=False, linewidth=2)
    ax.add_patch(tube)

    # Innenrohr andeuten (falls sinnvolle Wanddicke)
    Di = D - 2.0 * t_t
    if Di > 0:
        inner = Rectangle((-Di/2, 0.0), Di, L, fill=False, linewidth=1, linestyle='--', alpha=0.7)
        ax.add_patch(inner)

    # Finnen zeichnen (horizontale Rechtecke), zentriert bei x=0
    # Finnenmitten bei y = t_f/2 + i*p; Höhe = t_f, Breite = Hfin
    for i in range(N):
        y_center = t_f / 2.0 + i * p
        fin = Rectangle((-Hfin/2.0, y_center - t_f/2.0), Hfin, t_f, fill=False, linewidth=1)
        ax.add_patch(fin)

    # Achsen & Layout
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1 * (Hfin / 2.0), 1.1 * (Hfin / 2.0))  # Breite
    ax.set_ylim(-0.05 * L, 1.05 * L)                      # Länge
    ax.set_xlabel("Breite [m]")
    ax.set_ylabel("Länge [m]")
    ax.set_title("Lamellenverdampfer – Seitenansicht")
    ax.grid(True, alpha=0.3)
    plt.show()

def plot_frost_polar_slice(
    y, *,                         # y: (t, θ) oder (t, r, θ) – darf list/obj sein
    vary="theta",
    t_idx=None,
    at_time=None,                 # Sekunden; nimmt nächsten Zeitstempel
    t=None,                       # Zeitvektor in s (für at_time)
    r_idx=None,
    theta_vals=None,
    theta_max=np.pi/2,
    unit="mm",
    title=None,
    ylabel=None,
    marker=None,
    linestyle=None,
    ax=None,
    legend=True
):
    assert vary == "theta", "Diese Funktion plottet aktuell s_e(θ); setze vary='theta'."

    # --- NEU: y robust in ein 2D/3D-Array überführen ---
    arr = np.asarray(y, dtype=object)
    if arr.ndim == 1:
        # Liste von 1D-Sequenzen -> stacken zu (time, theta)
        arr = np.vstack([np.asarray(row, dtype=float) for row in arr])
    else:
        arr = np.asarray(arr, dtype=float)
    y = arr
    assert y.ndim in (2, 3), f"y hat unexpected shape {y.shape}; erwarte (t,θ) oder (t,r,θ)."

    # θ-Achse
    ntheta = y.shape[-1]
    if theta_vals is None:
        theta = np.linspace(0.0, theta_max, ntheta)
    else:
        theta = np.asarray(theta_vals)
        assert len(theta) == ntheta, "theta_vals passt nicht zu y.shape[-1]."

    # Zeitindizes/at_time
    if t_idx is not None:
        idxs = np.atleast_1d(t_idx).astype(int)
    elif at_time is not None:
        assert t is not None, "Für at_time muss t (Zeitvektor) übergeben werden."
        t = np.asarray(t).ravel()
        targets = np.atleast_1d(at_time).astype(float)
        idxs = np.array([np.abs(t - tau).argmin() for tau in targets], dtype=int)
    else:
        idxs = np.array([0], dtype=int)

    # Einheit
    factor = 1000.0 if unit.lower() == "mm" else 1.0
    label_unit = "mm" if unit.lower() == "mm" else "m"

    owns_fig = ax is None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure

    # >>> Nur 0..90° anzeigen
    rad_max = np.deg2rad(90)
    ax.set_thetalim(0, rad_max)  # Grenzen in Radiant
    ax.set_thetagrids([0, 30, 60, 90])  # Ticks in Grad

    # Plotten
    for i in idxs:
        if i < 0 or i >= y.shape[0]:
            raise IndexError(f"t_idx {i} liegt außerhalb [0,{y.shape[0]-1}].")
        if y.ndim == 2:
            yplot = y[i, :]
        else:
            assert r_idx is not None, "Für 3D y bitte r_idx angeben."
            yplot = y[i, r_idx, :]

        lbl = f"t={t[i]:g} s" if (t is not None and len(np.shape(t))==1 and i < len(t)) else f"t_idx={i}"
        ax.plot(theta, yplot * factor,
                label=lbl,
                marker=marker if marker is not None else None,
                linestyle=linestyle if linestyle is not None else None)

    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel if ylabel else f"s_e [{label_unit}]")
    if legend:
        ax.legend(loc="upper left")#, bbox_to_anchor=(1.3, 1.1))

    if owns_fig:
        plt.show()
    return fig, ax

def extract_segment_field_grid(
    grid_snapshots,
    field: str,
    *,
    source: str = "st",
    t_idx: int | None = None,
    at_time: float | None = None
):
    if not grid_snapshots:
        raise ValueError("grid_snapshots ist leer.")

    # Zeitindex bestimmen
    if t_idx is None:
        if at_time is None:
            t_idx = 0
        else:
            times = np.array([snap["t"] for snap in grid_snapshots], dtype=float)
            t_idx = int(np.abs(times - at_time).argmin())
    else:
        # Python-Style negative Indizes erlauben
        if t_idx < 0:
            t_idx = len(grid_snapshots) + t_idx

    if t_idx < 0 or t_idx >= len(grid_snapshots):
        raise IndexError(f"t_idx {t_idx} außerhalb des gültigen Bereichs [0, {len(grid_snapshots)-1}]")

    snap = grid_snapshots[t_idx]
    t_sel = float(snap["t"])

    if source == "st":
        grid = snap["st_grid"]
    elif source == "cfg":
        grid = snap["cfg_grid"]
    else:
        raise ValueError("source muss 'st' oder 'cfg' sein.")

    n_x = len(grid)
    n_y = len(grid[0])

    Z = np.zeros((n_x, n_y), dtype=float)

    for ix in range(n_x):
        for iy in range(n_y):
            obj = grid[ix][iy]
            val = getattr(obj, field)
            Z[ix, iy] = float(val)

    return t_sel, Z

def plot_segment_field_grid(
    grid_snapshots,
    field: str,
    *,
    source: str = "st",
    t_idx: int | None = None,
    at_time: float | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    show: bool = True,
    colorbar: bool = True
):
    """
    Plottet ein Feld über alle Segmente als 2D-Map (ix vs. iy).

    ix: Segmente in Luftflussrichtung (0..n_seg_l-1)
    iy: Segmente in Kältemittel-Richtung (0..n_seg_r-1)
    """
    t_sel, Z = extract_segment_field_grid(
        grid_snapshots,
        field,
        source=source,
        t_idx=t_idx,
        at_time=at_time,
    )

    n_x, n_y = Z.shape
    fig, ax = plt.subplots()

    im = ax.imshow(
        Z.T,                # transpose, damit x horizontal, y vertikal
        origin="lower",
        aspect="auto",
        cmap=cmap
    )

    ax.set_xlabel("ix (Luftflussrichtung)")
    ax.set_ylabel("iy (Reihen)")

    if title is None:
        ax.set_title(f"{field} bei t = {t_sel:.1f} s")
    else:
        ax.set_title(title + f" (t = {t_sel:.1f} s)")

    ax.set_xticks(np.arange(n_x))
    ax.set_yticks(np.arange(n_y))

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(field)

    if show:
        plt.show()

    return fig, ax, Z

def _sat_dome_ph(ref: str, n: int = 400):
    T_tr = PropsSI("Ttriple", ref)
    T_cr = PropsSI("Tcrit", ref)
    T = np.linspace(T_tr + 1.0, T_cr - 1.0, n)

    p  = np.array([PropsSI("P", "T", Ti, "Q", 0, ref) for Ti in T], dtype=float)
    hL = np.array([PropsSI("H", "T", Ti, "Q", 0, ref) for Ti in T], dtype=float)
    hV = np.array([PropsSI("H", "T", Ti, "Q", 1, ref) for Ti in T], dtype=float)
    return hL, hV, p

def _grid_style(ax, grid: str):
    grid = (grid or "dashed").lower()
    if grid in ("none", "off", "false", "0"):
        ax.grid(False)
    elif grid in ("dashed", "dash"):
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    elif grid in ("light", "thin"):
        ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.2)
    else:
        raise ValueError("grid must be 'none', 'dashed', or 'light'.")

def _auto_limits_logph_from_cycle(cycle_ph_sel, *, x_pad_rel=0.5, x_pad_abs=15.0, y_pad_factor=1.5):
    """
    cycle_ph_sel: (nsel, 4, 2) mit p[Pa], h[J/kg]
    Returns: (xlim_kJkg, ylim_bar)
    """
    ph = np.asarray(cycle_ph_sel, dtype=float)
    p_bar = ph[:, :, 0] / 1e5
    h_kj  = ph[:, :, 1] / 1000.0

    p_bar = p_bar[np.isfinite(p_bar)]
    h_kj  = h_kj[np.isfinite(h_kj)]

    if p_bar.size == 0 or h_kj.size == 0:
        return None, None

    hmin, hmax = float(np.min(h_kj)), float(np.max(h_kj))
    pmin, pmax = float(np.min(p_bar[p_bar > 0])) if np.any(p_bar > 0) else 0.1, float(np.max(p_bar))

    dh = hmax - hmin
    pad_x = max(x_pad_abs, x_pad_rel * dh) if dh > 1e-9 else x_pad_abs
    xlim = (hmin - pad_x, hmax + pad_x)

    # log-y: multiplikativer Rand ist stabiler
    ylo = max(pmin / y_pad_factor, 1e-6)
    yhi = pmax * y_pad_factor
    ylim = (ylo, yhi)

    return xlim, ylim

def _compute_isotherms_ph(ref: str, pmin: float, pmax: float, Ts_C, nP: int = 90):
    lines = []
    for T_C in Ts_C:
        T_K = float(T_C) + 273.15
        try:
            p_sat = float(PropsSI("P", "T", T_K, "Q", 0, ref))
        except Exception:
            continue

        # Dampfseite (p <= p_sat)
        h_vap = p_vap = None
        p_hi_vap = min(p_sat * 0.999, pmax)
        if pmin < p_hi_vap:
            pv = np.geomspace(max(pmin, 1.0), p_hi_vap, nP)
            hv = []
            for p in pv:
                try:
                    hv.append(float(PropsSI("H", "T", T_K, "P", float(p), ref)))
                except Exception:
                    hv.append(np.nan)
            hv = np.asarray(hv, dtype=float)
            m = np.isfinite(hv)
            if np.any(m):
                h_vap, p_vap = hv[m], pv[m]

        # Flüssigkeitsseite (p >= p_sat)
        h_liq = p_liq = None
        p_lo_liq = max(p_sat * 1.001, pmin)
        if p_lo_liq < pmax:
            pl = np.geomspace(p_lo_liq, pmax, nP)
            hl = []
            for p in pl:
                try:
                    hl.append(float(PropsSI("H", "T", T_K, "P", float(p), ref)))
                except Exception:
                    hl.append(np.nan)
            hl = np.asarray(hl, dtype=float)
            m = np.isfinite(hl)
            if np.any(m):
                h_liq, p_liq = hl[m], pl[m]

        if (h_vap is not None) or (h_liq is not None):
            lines.append((float(T_C), h_vap, p_vap, h_liq, p_liq))
    return lines

def plot_logph_cycles(
    ref: str,
    t,
    cycle_ph,
    *,
    t_idx=None,
    at_time=None,
    t_start=None,
    t_end=None,
    every_s=None,                 # z.B. 60.0
    plot_dome=True,

    # Isothermen + Grid
    isotherms: bool = True,
    iso_Ts_C=None,                # z.B. [-10, 0, 10, 20, 30, 40]
    n_iso: int = 10,               # falls iso_Ts_C None
    iso_style: str = "--",
    iso_lw: float = 1.0,
    iso_alpha: float = 0.45,
    iso_labels: bool = True,
    grid: str = "dashed",         # "none" | "dashed" | "light"

    # ---- NEU: Auto-Skalierung ohne Änderungen am Aufruf ----
    figsize=None,                 # None => automatisch (größerer Default)
    xlim=None,                    # (xmin, xmax) in kJ/kg; None => automatisch
    ylim=None,                    # (ymin, ymax) in bar;   None => automatisch

    title=None,
    save_path=None,
    show=True
):
    """
    t:        (nt,)
    cycle_ph: (nt,4,2) oder Liste von 4x2 pro Zeit:
              cycle_ph[i] = [[p1,h1],[p2,h2],[p3,h3],[p4,h4]] mit p[Pa], h[J/kg]
    """

    t = np.asarray(t, dtype=float).ravel()

    arr = np.asarray(cycle_ph, dtype=object)
    if arr.ndim == 1:
        arr = np.stack([np.asarray(row, dtype=float) for row in arr], axis=0)
    else:
        arr = np.asarray(arr, dtype=float)
    cycle_ph = arr

    if cycle_ph.ndim != 3 or cycle_ph.shape[1:] != (4, 2):
        raise ValueError(f"cycle_ph hat shape {cycle_ph.shape}, erwartet (nt,4,2).")

    # --- Zeit-Auswahl ---
    if t_idx is not None:
        idxs = np.atleast_1d(t_idx).astype(int)
        idxs = np.array([i if i >= 0 else len(t) + i for i in idxs], dtype=int)
    elif at_time is not None:
        targets = np.atleast_1d(at_time).astype(float)
        idxs = np.array([np.abs(t - tau).argmin() for tau in targets], dtype=int)
    elif (t_start is not None) or (t_end is not None):
        if t_start is None: t_start = float(t[0])
        if t_end   is None: t_end   = float(t[-1])
        if every_s is None:
            idxs = np.where((t >= t_start) & (t <= t_end))[0]
        else:
            targets = np.arange(float(t_start), float(t_end) + 1e-12, float(every_s))
            idxs = np.array([np.abs(t - tau).argmin() for tau in targets], dtype=int)
            idxs = np.unique(idxs)
    else:
        idxs = np.array([0], dtype=int)

    # --- Auto-Limits (wenn nicht explizit gesetzt) ---
    if xlim is None or ylim is None:
        auto_xlim, auto_ylim = _auto_limits_logph_from_cycle(cycle_ph[idxs])
        if xlim is None:
            xlim = auto_xlim
        if ylim is None:
            ylim = auto_ylim

    # Isothermen brauchen pmin/pmax in Pa: nutze bevorzugt die (auto/gegebenen) y-Limits
    if ylim is not None:
        pmin_pa = float(ylim[0]) * 1e5
        pmax_pa = float(ylim[1]) * 1e5
    else:
        # fallback
        p_sel = cycle_ph[idxs, :, 0].ravel()
        p_sel = p_sel[np.isfinite(p_sel)]
        pmin_pa = max(float(np.min(p_sel)) * 0.7, 1.0) if p_sel.size else 1e4
        pmax_pa = float(np.max(p_sel)) * 1.3 if p_sel.size else 1e7

    # --- Figure ---
    if figsize is None:
        figsize = (11, 7)  # größerer Default, ohne dass du beim Aufruf etwas ändern musst
    fig, ax = plt.subplots(figsize=figsize)

    # Sättigungsglocke
    if plot_dome:
        hL, hV, p = _sat_dome_ph(ref)
        ax.plot(hL/1000.0, p/1e5)
        ax.plot(hV/1000.0, p/1e5)

    # Isothermen
    if isotherms:
        if iso_Ts_C is None:
            # Tmin/Tmax aus ausgewählten Punkten (über P,H -> T)
            Ts = []
            for i in idxs:
                for k in range(4):
                    pPa = float(cycle_ph[i, k, 0])
                    hJ  = float(cycle_ph[i, k, 1])
                    if not (np.isfinite(pPa) and np.isfinite(hJ)):
                        continue
                    try:
                        Ts.append(float(PropsSI("T", "P", pPa, "H", hJ, ref) - 273.15))
                    except Exception:
                        pass
            if len(Ts) >= 2:
                Tmin, Tmax = min(Ts) - 5.0, max(Ts) + 5.0
                Ts_C = np.linspace(Tmin, Tmax, int(n_iso))
                Ts_C = [5.0 * round(x/5.0) for x in Ts_C]
                Ts_C = sorted(set(Ts_C))
            else:
                Ts_C = [-30, -20, -10, 0, 10, 20, 30, 40]
        else:
            Ts_C = list(iso_Ts_C)

        iso_lines = _compute_isotherms_ph(ref, pmin_pa, pmax_pa, Ts_C, nP=90)
        for (T_C, h_vap, p_vap, h_liq, p_liq) in iso_lines:
            if h_vap is not None:
                ax.plot(h_vap/1000.0, p_vap/1e5, linestyle=iso_style, linewidth=iso_lw, alpha=iso_alpha)
                if iso_labels:
                    ax.annotate(f"{T_C:.0f}°C", (h_vap[-1]/1000.0, p_vap[-1]/1e5),
                                textcoords="offset points", xytext=(4, 2))
            if h_liq is not None:
                ax.plot(h_liq/1000.0, p_liq/1e5, linestyle=iso_style, linewidth=iso_lw, alpha=iso_alpha)

    # Kreisprozess(e)
    for i in idxs:
        ph = cycle_ph[i]
        pPa = ph[:, 0]
        hJ  = ph[:, 1]

        p_plot = np.r_[pPa, pPa[0]] / 1e5
        h_plot = np.r_[hJ,  hJ[0]]  / 1000.0

        ax.plot(h_plot, p_plot, marker="o", label=f"t={t[i]:g} s")
        for k in range(4):
            ax.annotate(str(k+1), (hJ[k]/1000.0, pPa[k]/1e5),
                        textcoords="offset points", xytext=(5, 5))

    ax.set_yscale("log")
    ax.set_xlabel("h [kJ/kg]")
    ax.set_ylabel("p [bar]")

    # Auto/Manual Limits anwenden
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _grid_style(ax, grid)

    ax.legend()
    ax.set_title(title if title else f"log(p)-h Diagramm ({ref})")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def _psat_buck_Pa(T_C):
    """
    Sättigungsdampfdruck nach Buck (1981), Stückweise über Wasser/Eis.
    T_C in °C, Rückgabe in Pa.
    """
    T = np.asarray(T_C, dtype=float)
    # Buck: e_s in kPa
    e_kPa = np.where(
        T >= 0.0,
        0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T))),   # über Wasser
        0.61115 * np.exp((23.036 - T / 333.7) * (T / (279.82 + T)))    # über Eis
    )
    return 1000.0 * e_kPa  # Pa


def _humidity_ratio_from_pv(pv, P):
    """
    Feuchteverhältnis w [kg/kg_trockene_Luft] aus Partialdruck pv und Gesamtdruck P [Pa].
    """
    pv = np.asarray(pv, dtype=float)
    P = float(P)
    pv = np.clip(pv, 0.0, 0.999 * P)
    return 0.62198 * pv / (P - pv)


def _moist_air_enthalpy_kJkg_da(T_C, w):
    """
    Lineare Standardnäherung (wie in deinem Text): h [kJ/kg_da]
    T_C in °C, w in kg/kg_da.
    """
    T_C = np.asarray(T_C, dtype=float)
    w   = np.asarray(w, dtype=float)
    c_p_da = 1.006  # kJ/(kg K)
    c_p_v  = 1.86   # kJ/(kg K)
    h_g0   = 2501.0 # kJ/kg
    return c_p_da * T_C + w * (h_g0 + c_p_v * T_C)


def plot_mollier_hx_time(
    t,
    humidity,
    air_temp_l,
    *,
    P=101325.0,
    seg_idx=-1,                    # int | list[int] | "mean"
    plot_background=True,
    T_bg_min=-25.0,
    T_bg_max=35.0,
    iso_Ts_C=None,                 # z.B. [-20,-10,0,10,20,30]
    rh_lines=(0.2, 0.4, 0.6, 0.8, 1.0),
    title=None,
    save_path=None,
    show=True,
    ax=None,
    marker=None,
    line_kwargs=None,
    s_scatter=14
):
    """
    Mollier-(h-x)-Plot (h über x=w) für feuchte Luft, Zeit farbcodiert.

    Erwartete Einheiten (wie bei dir im Recorder):
      - t: Sekunden
      - humidity: w [kg/kg_trockene_Luft] als (nt,) oder (nt,nseg)
      - air_temp_l: T [°C] als (nt,) oder (nt,nseg)

    seg_idx:
      - int: Segmentindex (z.B. -1 = Outlet, 0 = Inlet)
      - list[int]: mehrere Segmente gleichzeitig
      - "mean": Mittel über Segmente (Achse 1)
    """

    t = np.asarray(t, dtype=float).ravel()
    w_raw = np.asarray(humidity, dtype=float)
    T_raw = np.asarray(air_temp_l, dtype=float)

    if t.ndim != 1:
        raise ValueError("t muss 1D sein.")
    if w_raw.shape[0] != t.shape[0] or T_raw.shape[0] != t.shape[0]:
        raise ValueError(f"Zeitachse passt nicht: t={t.shape}, w={w_raw.shape}, T={T_raw.shape}")

    # Temperatur-Einheit robust (falls doch in K gespeichert)
    if np.nanmean(T_raw) > 150.0:
        T_raw = T_raw - 273.15

    # Segmentauswahl normalisieren
    if isinstance(seg_idx, str) and seg_idx.lower() == "mean":
        if w_raw.ndim != 2 or T_raw.ndim != 2:
            raise ValueError("seg_idx='mean' setzt (nt,nseg) voraus.")
        w_list = [np.nanmean(w_raw, axis=1)]
        T_list = [np.nanmean(T_raw, axis=1)]
        labels = ["mean"]
    else:
        idxs = np.atleast_1d(seg_idx).astype(int) if not isinstance(seg_idx, (list, tuple, np.ndarray)) else np.asarray(seg_idx, dtype=int)
        if w_raw.ndim == 1:
            if idxs.size != 1:
                raise ValueError("Bei 1D humidity/air_temp_l ist nur ein seg_idx sinnvoll.")
            w_list = [w_raw]
            T_list = [T_raw]
            labels = [f"seg={int(idxs[0])}"]
        elif w_raw.ndim == 2:
            w_list, T_list, labels = [], [], []
            nseg = w_raw.shape[1]
            for i in idxs:
                ii = i if i >= 0 else nseg + i
                if ii < 0 or ii >= nseg:
                    raise IndexError(f"seg_idx {i} außerhalb [0,{nseg-1}]")
                w_list.append(w_raw[:, ii])
                T_list.append(T_raw[:, ii])
                labels.append(f"seg={i}")
        else:
            raise ValueError(f"humidity/air_temp_l müssen 1D oder 2D sein, got {w_raw.ndim}D.")

    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    # --- Hintergrund (Sättigungslinie, RH-Linien, Isothermen) ---
    if plot_background:
        T_bg = np.linspace(float(T_bg_min), float(T_bg_max), 260)
        psat = _psat_buck_Pa(T_bg)
        w_sat = _humidity_ratio_from_pv(psat, P)
        h_sat = _moist_air_enthalpy_kJkg_da(T_bg, w_sat)

        # Sättigungslinie (100% RH)
        ax.plot(w_sat * 1000.0, h_sat, linewidth=1.5, label="RH=100%")

        # RH-Linien
        for rh in rh_lines:
            rh = float(rh)
            if not (0.0 < rh <= 1.0):
                continue
            pv = rh * psat
            w_rh = _humidity_ratio_from_pv(pv, P)
            h_rh = _moist_air_enthalpy_kJkg_da(T_bg, w_rh)
            ax.plot(w_rh * 1000.0, h_rh, linewidth=0.9, alpha=0.5)

        # Isothermen (als h(w) bei festem T)
        if iso_Ts_C is None:
            iso_Ts_C = [-20, -10, 0, 10, 20, 30]
        for T0 in iso_Ts_C:
            ps = float(_psat_buck_Pa(T0))
            w0 = float(_humidity_ratio_from_pv(ps, P))
            w_line = np.linspace(0.0, max(w0, 1e-6), 120)
            h_line = _moist_air_enthalpy_kJkg_da(float(T0), w_line)
            ax.plot(w_line * 1000.0, h_line, linestyle="--", linewidth=0.8, alpha=0.5)
            ax.annotate(f"{T0:.0f}°C", (w_line[0]*1000.0, h_line[0]), textcoords="offset points", xytext=(-4, 2))

    # --- Pfade über Zeit (farbcodiert) ---
    plot_kwargs = {}
    if line_kwargs:
        plot_kwargs.update(line_kwargs)

    last_sc = None
    for w_i, T_i, lab in zip(w_list, T_list, labels):
        w_i = np.asarray(w_i, dtype=float)
        T_i = np.asarray(T_i, dtype=float)
        h_i = _moist_air_enthalpy_kJkg_da(T_i, w_i)

        x = w_i * 1000.0  # g/kg_da
        y = h_i           # kJ/kg_da

        # Linie (Zustandspfad)
        ax.plot(x, y, label=lab, **plot_kwargs)

        # Zeit-codierte Punkte
        last_sc = ax.scatter(x, y, c=t, s=s_scatter, marker=marker if marker else "o")
        # Start/End markieren
        if len(t) >= 2:
            ax.annotate("start", (x[0], y[0]), textcoords="offset points", xytext=(6, 6))
            ax.annotate("end",   (x[-1], y[-1]), textcoords="offset points", xytext=(6, 6))

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=ax)
        cbar.set_label("t [s]")

    ax.set_xlabel("x = w [g/kg$_{da}$]")
    ax.set_ylabel("h [kJ/kg$_{da}$]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title if title else "Mollier (h-x): Luftzustand über Zeit")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if show and owns_fig:
        plt.show()

    return fig, ax
