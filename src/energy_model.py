import numpy as np
import matplotlib.pyplot as plt
from vehicle_params import VehicleParams

def simulate_energy(cycle: dict, params: VehicleParams, plot: bool = False) -> dict:
    """
    根据工况与车辆参数计算能耗。
    """
    try:
        t, v, theta = cycle["t"], cycle["v"], cycle["theta"]
    except KeyError:
        raise KeyError("[ERROR] cycle data MISSING t/v/theta")

    dt = np.diff(t, prepend=t[0])
    a = np.gradient(v, t)

    # Forces
    F_inert = params.mass_kg * a
    F_grade = params.mass_kg * params.g * np.sin(theta)
    F_roll  = params.Cr * params.mass_kg * params.g * np.cos(theta)
    F_aero  = 0.5 * params.rho_air * params.CdA_m2 * v**2
    F_trac  = F_inert + F_grade + F_roll + F_aero

    # Power
    P_wheel = F_trac * v
    P_bat = np.where(P_wheel > 0,
                     P_wheel / params.eta_m,
                     params.eta_regen * P_wheel)
    P_bat = np.maximum(P_bat, -params.Pchg_max)

    E_Wh = np.trapezoid(P_bat, t) / 3600
    dist_km = np.trapezoid(v, t) / 1000
    Wh_per_km = E_Wh / max(dist_km, 1e-6)

    if plot:
        plt.figure(figsize=(7,5))
        plt.subplot(2,1,1)
        plt.plot(t, v*3.6)
        plt.ylabel("Speed (km/h)")
        plt.subplot(2,1,2)
        plt.plot(t, P_bat/1000)
        plt.ylabel("Battery Power (kW)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return {
        "E_Wh": E_Wh,
        "distance_km": dist_km,
        "Wh_per_km": Wh_per_km
    }
