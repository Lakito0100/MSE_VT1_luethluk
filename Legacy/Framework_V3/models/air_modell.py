from CoolProp.HumidAirProp import HAPropsSI
from Legacy.Framework_V3.models.Frostmodell_V1 import Frostmodell_Finn_and_Tube

class Air:

    def __init__(self):
        self.ft_model = Frostmodell_Finn_and_Tube()

    def propagate_inplace(cls,
                          cfg_in,
                          cfg_out,
                          st_bevor,
                          st_seg,
                          geom,
                          m_dot_a: float,
                          Q_seg: float,
                          m_s_seg: float,
                          dp_seg: float = 0.0):
        """
        Aktualisiert cfg_out in-place auf Basis von cfg_in und den Segmentflüssen.

        cfg_in  : Luftzustand am Eintritt des aktuellen Segments
        cfg_out : Luft-Config des 'nächsten' Segments (wird überschrieben)
        m_dot_a : Luftmassenstrom [kg/s]
        Q_seg   : Wärmestrom Luft -> Frost [W]
        m_s_seg : Wasserdampfstrom Luft -> Frost [kg/s]
        dp_seg  : Druckverlust [Pa], p_out = p_in - dp_seg

        Rückgabe:
            T_out, w_out, p_out
        """
        T_in = cfg_in.T_a
        w_in = cfg_in.w_amb
        p_in = cfg_in.p_a
        v_in = cfg_in.v_a
        rho_in = cfg_in.rho_a

        # Energie- und Stoffbilanz
        dT = -Q_seg / (m_dot_a * cfg_in.c_p_a)
        dw = -m_s_seg / m_dot_a

        T_out = T_in + dT
        w_out = w_in + dw
        p_out = p_in - dp_seg

        RH_out = HAPropsSI("R", "T", T_out+273.15, "P", p_out, "W", w_out)
        rho_out = 1.0 / HAPropsSI("Vha", "T", T_out+273.15, "P", p_out, "W", w_out)

        # Update Geschwindigkeit
        A_in = geom.h_fin * (geom.fin_pitch-2*st_bevor.s_ft)
        A_out = geom.h_fin * (geom.fin_pitch - 2 * st_seg.s_ft)
        mass_flow = A_in * v_in * rho_in
        v_out = mass_flow / (A_out * rho_out)

        # cfg_out in-place überschreiben (nur Luft-Daten)
        cfg_out.T_a = T_out
        cfg_out.v_a = v_out  # Geschwindigkeit gleich angenommen
        cfg_out.p_a = p_out
        cfg_out.RH = RH_out
        cfg_out.w_amb = w_out
        cfg_out.rho_amb = rho_out

        return T_out, w_out, p_out


