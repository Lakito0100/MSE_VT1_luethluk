import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator
import pickle

def drho_dP_dH(
    self,
    fluid: str = "R134a",
    # Tabulation range (P,H)
    P_min: float = 1e5,
    P_max: float = 20e5,
    dP: float = 1e4,
    H_min: float = 2e5,
    H_max: float = 5e5,
    dH: float = 1e3,
    # If set: return derivatives only at these points
    query_P=None,   # float or array-like [Pa]
    query_H=None,   # float or array-like [J/kg]
    # Central is typically more stable than forward
    scheme: str = "central",  # "forward" or "central"
    # Optional: save/load interpolators
    save_path: str | None = None,
    load_path: str | None = None,
):
    """
    Return partial derivatives of density ρ with respect to pressure P and enthalpy H:
        dρ/dP [kg/m³/Pa], dρ/dH [kg/m³/(J/kg)].

    If query_P/query_H are set:
        return derivatives at those points (same shape as query).
    Otherwise:
        return derivative fields on the tabulation grid plus interpolators.

    Inputs:
      - fluid: CoolProp fluid string (e.g. "R134a")
      - P_min,P_max,dP: pressure range [Pa] and step
      - H_min,H_max,dH: enthalpy range [J/kg] and step
      - query_P, query_H: evaluation point(s) for derivatives
      - scheme: "central" or "forward"
      - save_path/load_path: pickle for interpolators (optional)
    """

    # ------------------------------------------------------------
    # 0) Load interpolators if requested
    # ------------------------------------------------------------
    if load_path is not None:
        with open(load_path, "rb") as f:
            interps = pickle.load(f)
        rho_interp = interps["rho"]
        # Optional: if derivative interpolators are stored directly
        drho_dP_interp = interps.get("drho_dP", None)
        drho_dH_interp = interps.get("drho_dH", None)

        # If derivative interpolators exist, evaluate directly:
        if (query_P is not None) and (query_H is not None) and \
           (drho_dP_interp is not None) and (drho_dH_interp is not None):
            qP = np.asarray(query_P, dtype=float)
            qH = np.asarray(query_H, dtype=float)
            pts = np.stack([qP, qH], axis=-1)
            return drho_dP_interp(pts), drho_dH_interp(pts)

        # Otherwise build derivatives below (using rho_interp).
        # -> We still need the grid; if you want "load only",
        #    save drho_dP/drho_dH as well.
        #    (P_vec/H_vec are created below anyway.)
    # ------------------------------------------------------------
    # 1) Definition range (tabulation grid)
    # ------------------------------------------------------------
    P_vec = np.arange(P_min, P_max + dP, dP, dtype=float)
    H_vec = np.arange(H_min, H_max + dH, dH, dtype=float)

    P_grid, H_grid = np.meshgrid(P_vec, H_vec, indexing="ij")
    shape = P_grid.shape

    # ------------------------------------------------------------
    # 2) Thermophysical properties (CoolProp) -> rho(P,H)
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
        # rho_interp was loaded
        pass

    # ------------------------------------------------------------
    # 3) Numerical derivatives on the grid (for interpolators)
    # ------------------------------------------------------------
    # Points for interpolation must be (...,2):
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
        raise ValueError("scheme must be 'forward' or 'central'.")

    # Interpolators for derivatives (useful for solvers)
    drho_dP_interp = RegularGridInterpolator(
        (P_vec, H_vec), d_rho_dP, bounds_error=False, fill_value=np.nan
    )
    drho_dH_interp = RegularGridInterpolator(
        (P_vec, H_vec), d_rho_dH, bounds_error=False, fill_value=np.nan
    )

    # ------------------------------------------------------------
    # 4) Optional save (including derivative interpolators)
    # ------------------------------------------------------------
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(
                {"rho": rho_interp, "T": T_interp,  "drho_dP": drho_dP_interp, "drho_dH": drho_dH_interp},
                f
            )

    # ------------------------------------------------------------
    # 5) Output: either at query points or full fields
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
