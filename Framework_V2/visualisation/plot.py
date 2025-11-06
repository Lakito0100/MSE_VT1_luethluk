import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path
import numpy as np
import ast, re

def _arr(cell):
    """Parse array-like CSV cell:
       - Python-list style:        [1, 2, 3], [[1,2],[3,4]]
       - NumPy-style (spaces):     [1 2 3],   [[1 2] [3 4]]
    Returns a numpy array (float).
    """
    # numbers already
    if isinstance(cell, (int, float, np.floating)):
        return np.array([float(cell)], dtype=float)

    if not isinstance(cell, str):
        return np.array([], dtype=float)

    s = cell.strip()
    if not s:
        return np.array([], dtype=float)

    # 1) try Python-literal directly
    try:
        return np.array(ast.literal_eval(s), dtype=float)
    except Exception:
        pass

    # 2) NumPy-space style
    #    normalize to rows by splitting on '][' or ']\s+\[' boundaries
    inner = s.strip("[]").strip()
    if not inner:
        return np.array([], dtype=float)

    # split rows at ][ (with optional spaces)
    row_strs = re.split(r"\]\s*\[", inner)
    rows = []
    for r in row_strs:
        r_clean = r.strip("[]").strip()
        if not r_clean:
            rows.append([])
            continue
        # split numbers by whitespace
        nums = r_clean.split()
        rows.append([float(x) for x in nums])

    # if single row -> 1D
    if len(rows) == 1:
        return np.array(rows[0], dtype=float)
    # else -> 2D (ragged rows will raise; that’s fine)
    return np.array(rows, dtype=float)


# --- 1) plot value at (r, theta) vs time ---
def plot_vs_time(csv_file: str, var: str, r_idx: int = -1, theta_idx: int | None = None,
                 t_col: str | None = None, save: bool = True):
    df = pd.read_csv(csv_file)

    # pick time column (or use index)
    if t_col is None or t_col not in df.columns:
        df = df.reset_index().rename(columns={"index": "t"})
        t_col = "t"

    # build y(t) by slicing each timestep's array
    y = []
    for cell in df[var]:
        a = _arr(cell).squeeze()
        if a.ndim == 1:                      # shape: [r]
            y.append(float(a[r_idx]))
        else:                                # shape: [r, theta] (or transposed)
            if theta_idx is None: theta_idx = 0
            try:
                y.append(float(a[r_idx, theta_idx]))
            except Exception:
                y.append(float(a[theta_idx]))

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(df[t_col].values, y)
    plt.xlabel(t_col); plt.ylabel(f"{var} @ r={r_idx}" + (f", θ={theta_idx}" if theta_idx is not None else ""))
    plt.title(f"{Path(csv_file).stem}: {var} vs time")
    plt.grid(True)
    for tick in plt.gca().get_xticklabels():
        tick.set_rotation(20); tick.set_ha("right")

    out = None
    if save:
        out = Path(csv_file).with_suffix("")
        out = Path(f"{out}_{var}_vs_time_r{r_idx}" + (f"_th{theta_idx}" if theta_idx is not None else "") + ".png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    return str(out) if save else None

# --- 2) plot profile vs r at a chosen time row ---
def plot_vs_r(csv_file: str, var: str, row: int = -1, theta_idx: int | None = None,
              save: bool = True):
    df = pd.read_csv(csv_file)
    a = _arr(df[var].iloc[row]).squeeze()

    if a.ndim == 0:
        raise ValueError(f"'{var}' at row {row} is scalar; cannot plot vs r.")
    if a.ndim == 1:
        y = a                          # already [r]
    else:
        if theta_idx is None: theta_idx = 0
        try:
            y = a[:, theta_idx]        # [r, theta]
        except Exception:
            y = a.T[:, theta_idx]      # fallback if stored transposed

    x = np.arange(len(y))              # r index (no physical r given in CSV)
    plt.figure(figsize=(8,4))
    plt.plot(x, y)
    plt.xlabel("r index"); plt.ylabel(var)
    plt.title(f"{Path(csv_file).stem}: {var} vs r (time={row}" + (f", θ={theta_idx}" if theta_idx is not None else "") + ")")
    plt.grid(True)

    out = None
    if save:
        out = Path(csv_file).with_suffix("")
        out = Path(f"{out}_{var}_vs_r_row{row}" + (f"_th{theta_idx}" if theta_idx is not None else "") + ".png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    return str(out) if save else None


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