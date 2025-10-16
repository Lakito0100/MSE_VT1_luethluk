import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_frostdicke(results, time_unit="s", show=True, savepath=None):
    # results kann entweder ein ResultRecorder (mit .data) oder ein dict sein
    data = results.data
    t = data["t"]
    fx = data["x_frost"]             # Frostdicke in m
    fx_mm = [x*60 for x in fx]  # Frostdicke in mm

    if time_unit == "min":
        t = [ti / 60.0 for ti in t]
        xlabel = "Zeit [min]"
    else:
        xlabel = "Zeit [s]"

    plt.plot(t, fx_mm, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("Frostdicke s [mm]")
    plt.title("Frostdicke über der Zeit")
    plt.grid(True, alpha=0.3)

    plt.show()
    return plt

def plot_finned_tube_side(he):
    L   = he.l_rohr()
    D   = he.d_rohr_a
    N   = int(he.n_rippen)
    Lf  = he.l_rippen
    tf  = he.rippen_dicke

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