import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator
import pickle

def drho_dP_dH(
    self,
    fluid: str = "R134a",
    # Tabellierungsbereich (P,H)
    P_min: float = 1e5,
    P_max: float = 20e5,
    dP: float = 1e4,
    H_min: float = 2e5,
    H_max: float = 5e5,
    dH: float = 1e3,
    # Wenn gesetzt: Ableitungen nur an diesen Punkten ausgeben
    query_P=None,   # float oder array-like [Pa]
    query_H=None,   # float oder array-like [J/kg]
    # zentral ist i.d.R. stabiler als vorwärts
    scheme: str = "central",  # "forward" oder "central"
    # optional: Interpolatoren speichern/laden
    save_path: str | None = None,
    load_path: str | None = None,
):
    """
    Liefert partielle Ableitungen der Dichte ρ nach Druck P und Enthalpie H:
        dρ/dP [kg/m³/Pa], dρ/dH [kg/m³/(J/kg)].

    Falls query_P/query_H gesetzt:
        Rückgabe sind Ableitungen an diesen Punkten (gleiche Form wie query).
    Sonst:
        Rückgabe sind Ableitungsfelder auf dem Tabellengitter plus Interpolatoren.

    Inputs:
      - fluid: CoolProp Fluidstring (z.B. "R134a")
      - P_min,P_max,dP: Druckbereich [Pa] und Schritt
      - H_min,H_max,dH: Enthalpiebereich [J/kg] und Schritt
      - query_P, query_H: Auswertepunkt(e) für Ableitungen
      - scheme: "central" oder "forward"
      - save_path/load_path: Pickle für Interpolatoren (optional)
    """

    # ------------------------------------------------------------
    # 0) Interpolatoren ggf. laden
    # ------------------------------------------------------------
    if load_path is not None:
        with open(load_path, "rb") as f:
            interps = pickle.load(f)
        rho_interp = interps["rho"]
        # optional: wenn du auch direkt Ableitungsinterpolatoren speicherst
        drho_dP_interp = interps.get("drho_dP", None)
        drho_dH_interp = interps.get("drho_dH", None)

        # Wenn Ableitungsinterpolatoren vorhanden sind, direkt auswerten:
        if (query_P is not None) and (query_H is not None) and \
           (drho_dP_interp is not None) and (drho_dH_interp is not None):
            qP = np.asarray(query_P, dtype=float)
            qH = np.asarray(query_H, dtype=float)
            pts = np.stack([qP, qH], axis=-1)
            return drho_dP_interp(pts), drho_dH_interp(pts)

        # sonst: wir bauen die Ableitungen unten neu (mit rho_interp)
        # -> dafür brauchen wir das Gitter; wenn du "load only" willst,
        #    speichere drho_dP/drho_dH mit ab.
        #    (Weiter unten werden P_vec/H_vec sowieso erzeugt.)
    # ------------------------------------------------------------
    # 1) Definitionsbereich (Tabellen-Gitter)
    # ------------------------------------------------------------
    P_vec = np.arange(P_min, P_max + dP, dP, dtype=float)
    H_vec = np.arange(H_min, H_max + dH, dH, dtype=float)

    P_grid, H_grid = np.meshgrid(P_vec, H_vec, indexing="ij")
    shape = P_grid.shape

    # ------------------------------------------------------------
    # 2) Thermophysikalische Eigenschaften (CoolProp) -> rho(P,H)
    # ------------------------------------------------------------
    if load_path is None:
        P_flat = P_grid.ravel()
        H_flat = H_grid.ravel()

        rho_flat = PropsSI("D", "P", P_flat, "H", H_flat, fluid)
        rho = rho_flat.reshape(shape)

        T_flat = PropsSI("T", "P", P_flat, "H", H_flat, fluid)
        T = T_flat.reshape(shape)

        rho_interp = RegularGridInterpolator(
            (P_vec, H_vec), rho, bounds_error=False, fill_value=np.nan
        )

        T_interp = RegularGridInterpolator(
            (P_vec, H_vec), T, bounds_error=False, fill_value=np.nan
        )
    else:
        # rho_interp wurde geladen
        pass

    # ------------------------------------------------------------
    # 3) Numerische Ableitungen auf dem Gitter (für Interpolator)
    # ------------------------------------------------------------
    # Punkte fürs Interpolieren müssen (...,2) sein:
    pts0 = np.stack([P_grid, H_grid], axis=-1)
    rho0 = rho_interp(pts0)

    if scheme.lower() == "forward":
        rho_P_fwd = rho_interp(np.stack([P_grid + dP, H_grid], axis=-1))
        rho_H_fwd = rho_interp(np.stack([P_grid, H_grid + dH], axis=-1))

        d_rho_dP = (rho_P_fwd - rho0) / dP
        d_rho_dH = (rho_H_fwd - rho0) / dH

    elif scheme.lower() == "central":
        rho_P_plus  = rho_interp(np.stack([P_grid + dP, H_grid], axis=-1))
        rho_P_minus = rho_interp(np.stack([P_grid - dP, H_grid], axis=-1))
        rho_H_plus  = rho_interp(np.stack([P_grid, H_grid + dH], axis=-1))
        rho_H_minus = rho_interp(np.stack([P_grid, H_grid - dH], axis=-1))

        d_rho_dP = (rho_P_plus - rho_P_minus) / (2.0 * dP)
        d_rho_dH = (rho_H_plus - rho_H_minus) / (2.0 * dH)
    else:
        raise ValueError("scheme muss 'forward' oder 'central' sein.")

    # Interpolatoren für die Ableitungen (praktisch für Solver)
    drho_dP_interp = RegularGridInterpolator(
        (P_vec, H_vec), d_rho_dP, bounds_error=False, fill_value=np.nan
    )
    drho_dH_interp = RegularGridInterpolator(
        (P_vec, H_vec), d_rho_dH, bounds_error=False, fill_value=np.nan
    )

    # ------------------------------------------------------------
    # 4) Optional speichern (inkl. Ableitungs-Interpolatoren)
    # ------------------------------------------------------------
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(
                {"rho": rho_interp, "T": T_interp,  "drho_dP": drho_dP_interp, "drho_dH": drho_dH_interp},
                f
            )

    # ------------------------------------------------------------
    # 5) Output: entweder an query-Punkten oder ganze Felder
    # ------------------------------------------------------------
    if (query_P is not None) and (query_H is not None):
        qP = np.asarray(query_P, dtype=float)
        qH = np.asarray(query_H, dtype=float)
        pts = np.stack([qP, qH], axis=-1)
        return drho_dP_interp(pts), drho_dH_interp(pts)

    return {
        "P_vec": P_vec,
        "H_vec": H_vec,
        "d_rho_dP_grid": d_rho_dP,
        "d_rho_dH_grid": d_rho_dH,
        "rho_interp": rho_interp,
        "T_interp": T_interp,
        "drho_dP_interp": drho_dP_interp,
        "drho_dH_interp": drho_dH_interp,
    }
