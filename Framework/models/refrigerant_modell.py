from CoolProp.CoolProp import PropsSI


class Refrigerant:
    """
    Aktualisiert den Kältemittelzustand entlang eines vorgegebenen Pfades
    durch das Segment-Grid.

    Parameter
    ---------
    cfg_grid : 2D-Liste [ix][iy] -> CaseConfig
        Enthält für jedes Segment u.a.:
            T_ref [°C], p_ref [Pa], h_ref [J/kg], V_dot_ref [m³/s], x_ref [-], T_tube [°C]
    st_grid  : 2D-Liste [ix][iy] -> SimState
        Enthält den Frostzustand pro Segment. Erwartet wird:
            st_grid[ix][iy].Q_seg_wall [W]  (Wärmestrom in das Rohr/Kältemittel)
    geom     : FinnTubedHX
        Aktuell nur als Platzhalter mitgeführt (falls du später z.B. Längen o.ä. brauchst).
    fluid_ref : str
        Kältemittelfluidname für CoolProp, z.B. "R134a", "R410A", "R744".
    connection_path : list[tuple[int, int]]
        Liste von (ix, iy)-Koordinaten in physikalischer Flussreihenfolge.
        Beispiel: [(0,0), (0,1), (0,2), (1,2), (1,1), (1,0)]
    dp_ref_seg : float, optional
        Konstanter Druckverlust pro Segment [Pa]. Default: 0.0

    Verwendung
    ----------
    model = RefrigerantModel(cfg_grid, st_grid, "R134a", connection_path)
    model.update_all_segments()
    """

    def update_all_segments(self,
                 cfg_inlet,
                 cfg_grid,
                 st_grid,
                 Q_seg_list: float,
                 fluid: str,
                 connection_path: list = [(4,4),(4,3),(4,2),(4,1),(4,0),
                                          (3,0),(3,1),(3,2),(3,3),(3,4),
                                          (2,4),(2,3),(2,2),(2,1),(2,0),
                                          (1,0),(1,1),(1,2),(1,3),(1,4),
                                          (0,4),(0,3),(0,2),(0,1),(0,0)],
                 dp_ref_seg: float = 0.0):
        """
        Geht den vorgegebenen Pfad segmentweise durch und aktualisiert
        den Kältemittelzustand in den jeweiligen CaseConfig-Objekten.
        """

        # Eintrittsdichte und Massenstrom (mass flow bleibt entlang des Pfades konstant)
        rho_in = PropsSI("D", "P", cfg_inlet.p_ref, "H", cfg_inlet.h_ref, fluid)
        m_dot_ref = rho_in * cfg_inlet.V_dot_ref  # [kg/s]

        if m_dot_ref <= 0.0:
            raise ValueError("RefrigerantModel: m_dot_ref <= 0 aus Eintrittszustand abgeleitet.")

        # Segmentweise entlang des Pfades marschieren
        for seg_idx, (ix, iy) in enumerate(connection_path):
            cfg = cfg_grid[ix][iy]
            st = st_grid[ix][iy]

            # Wärmestrom in das Rohr/Kältemittel in diesem Segment
            # (muss vorher im Frostmodell gesetzt werden, z.B.: st.Q_seg_wall = Q_seg_x0)
            Q_seg = Q_seg_list[ix][iy]

            h_in = cfg.h_ref
            p_in = cfg.p_ref

            # Steady-State Energiebilanz:
            #   m_dot * (h_out - h_in) = Q_seg  ->  h_out = h_in + Q_seg / m_dot
            h_out = h_in + Q_seg / m_dot_ref
            p_out = p_in - dp_ref_seg

            # Thermodynamischer Zustand aus (p_out, h_out) bestimmen
            T_out_K = PropsSI("T", "P", p_out, "H", h_out, fluid)
            T_out = T_out_K - 273.15

            rho_out = PropsSI("D", "P", p_out, "H", h_out, fluid)
            V_dot_out = m_dot_ref / rho_out  # [m³/s]

            # Dampfqualität, falls 2-phasig – sonst NaN
            try:
                x_out = PropsSI("Q", "P", p_out, "H", h_out, fluid)
            except ValueError:
                x_out = float("nan")

            # Config im Segment aktualisieren
            cfg.h_ref = h_out
            cfg.p_ref = p_out
            cfg.T_ref = T_out
            cfg.V_dot_ref = V_dot_out
            cfg.x_ref = x_out

            # Einfache Annahme: Rohrtemperatur = Kältemitteltemperatur
            cfg.T_tube = T_out
