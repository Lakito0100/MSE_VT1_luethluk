import matplotlib.pyplot as plt

def plot_frostdicke(results, time_unit="s", show=True, savepath=None):
    # results kann entweder ein ResultRecorder (mit .data) oder ein dict sein
    data = results.data
    t = data["t"]
    fx = data["fx"]             # Frostdicke in m
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
