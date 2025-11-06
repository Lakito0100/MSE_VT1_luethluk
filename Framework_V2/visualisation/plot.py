import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from pathlib import Path

def plot_results(csv_file: str, x: str = None, y: list | None = None,
                 title: str | None = None, save: bool = True,
                 dpi: int = 200, show: bool = False) -> str | None:
    """
    Plot results from a CSV.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    x : str, optional
        Column to use as x-axis. If None, attempts to guess a time-like column.
        If none is found, uses the row index.
    y : list[str], optional
        Columns to plot on the y-axis. If None, plots all numeric columns except x.
    title : str, optional
        Plot title.
    save : bool, optional
        If True, saves a PNG next to the CSV with suffix '_plot.png'.
    dpi : int
        Figure resolution when saving.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    str | None
        Path to the saved PNG if save=True, otherwise None.
    """
    df = pd.read_csv(csv_file)

    # Choose x axis
    if x is None:
        x = _guess_x_column(list(df.columns))
        if x is None:
            df = df.reset_index().rename(columns={"index": "index"})
            x = "index"

    # Choose y columns
    if y is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y_cols = [c for c in numeric_cols if c != x]
    else:
        y_cols = y

    if len(y_cols) == 0:
        raise ValueError("No numeric columns to plot. Provide 'y' or ensure the CSV has numeric data.")

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for col in y_cols:
        ax.plot(df[x], df[col], label=col)

    ax.set_xlabel(x)
    ax.set_ylabel("Value")
    ax.set_title(title or f"Results from {Path(csv_file).name}")
    ax.grid(True)
    ax.legend(loc="best")

    out_path = None
    if save:
        out_path = Path(csv_file).with_suffix("")  # strip .csv
        out_path = Path(str(out_path) + "_plot.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(out_path) if save else None

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