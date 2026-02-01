import numpy as np
import matplotlib.pyplot as plt
from vehicle_params import VehicleParams


def simulate_energy(cycle: dict, params: VehicleParams, plot: bool = False) -> dict:
    """
    Calculate energy usage including:
    - battery power
    - regenerative braking energy
    - peak regen power
    - SoC trace and final SoC
    - plot with highlighted regen region
    """
    try:
        t, v, theta = cycle["t"], cycle["v"], cycle["theta"]
    except KeyError:
        raise KeyError("[ERROR] cycle data MISSING t/v/theta")

    dt = np.diff(t, prepend=t[0])
    a = np.gradient(v, t)

    total_mass = params.mass_kg + params.rider_mass_kg

    # ==========================
    # 1. Forces
    # ==========================
    F_inert = total_mass * a
    F_grade = total_mass * params.g * np.sin(theta)
    F_roll = params.Cr * total_mass * params.g * np.cos(theta)
    F_aero = 0.5 * params.rho_air * params.CdA_m2 * v**2
    F_trac = F_inert + F_grade + F_roll + F_aero

    # ==========================
    # 2. Wheel power
    # ==========================
    P_wheel = F_trac * v

    # ==========================
    # 3. Battery power (includes regen)
    # ==========================
    P_bat = np.where(
        P_wheel > 0,
        P_wheel / params.eta_m,
        params.eta_regen * P_wheel
    )

    # limit regen power
    P_bat = np.maximum(P_bat, -params.Pchg_max)

    # limit discharge power
    P_bat = np.minimum(P_bat, params.Pdis_max)

    # ==========================
    # 4. Integrate battery energy
    # ==========================
    E_Wh = np.trapezoid(P_bat, t) / 3600.0
    dist_km = np.trapezoid(v, t) / 1000.0
    Wh_per_km = E_Wh / max(dist_km, 1e-9)

    def signed_energy(power_trace: np.ndarray) -> float:
        """Integrate a power trace in Wh (signed)."""
        return np.trapezoid(power_trace, t) / 3600.0

    def positive_energy(power_trace: np.ndarray) -> float:
        """Compute propulsion-only (positive) energy for a power trace in Wh."""
        return signed_energy(np.where(power_trace > 0, power_trace, 0.0))

    grade_power = F_grade * v
    grade_assist_Wh = signed_energy(np.where(grade_power < 0, -grade_power, 0.0))
    grade_work_Wh = signed_energy(grade_power)
    grade_undulation_Wh = signed_energy(np.abs(grade_power))

    component_energy_Wh = {
        "inertial": positive_energy(F_inert * v),
        "grade": positive_energy(grade_power),
        "undulation": grade_undulation_Wh,
        "rolling": positive_energy(F_roll * v),
        "aerodynamic": positive_energy(F_aero * v),
    }

    # ==========================
    # 5. Regenerative braking energy
    # ==========================
    P_regen = np.where(P_bat < 0, -P_bat, 0.0)  # convert negative to positive
    E_regen_Wh = np.trapezoid(P_regen, t) / 3600.0

    # regen fraction = recovered brake energy / wheel braking energy
    P_wheel_brake = np.where(P_wheel < 0, -P_wheel, 0.0)
    braking_energy_raw = np.trapezoid(P_wheel_brake, t) / 3600.0
    regen_fraction = E_regen_Wh / max(braking_energy_raw, 1e-9)

    # ==========================
    # 6. Peak regenerative braking power
    # ==========================
    regen_peak_power_W = np.max(P_regen)  # already positive

    # ==========================
    # 7. Battery SoC simulation
    # ==========================
    SoC = np.zeros_like(t)
    SoC[0] = params.soc0

    for i in range(1, len(t)):
        # dE = P_bat * dt   → convert J to Wh → SoC change
        dWh = P_bat[i] * dt[i] / 3600.0
        SoC[i] = np.clip(SoC[i-1] - dWh / params.Cap_Wh, 0.0, 1.0)

    final_soc = SoC[-1]

    # ==========================
    # 8. Plot
    # ==========================
    if plot:
        plt.figure(figsize=(8, 7))

        # Speed
        plt.subplot(3, 1, 1)
        plt.plot(t, v*3.6)
        plt.ylabel("Speed (km/h)")

        # Wheel Power
        plt.subplot(3, 1, 2)
        plt.plot(t, P_wheel/1000, label="Wheel Power")
        plt.ylabel("P_wheel (kW)")

        # Battery power with regen highlighted
        plt.subplot(3, 1, 3)
        plt.plot(t, P_bat/1000, label="Battery Power", color='blue')

        # Highlight regen region (P_bat < 0)
        regen_mask = P_bat < 0
        plt.fill_between(t, P_bat/1000, 0,
                         where=regen_mask,
                         color='red', alpha=0.4,
                         label="Regenerative Region")

        plt.ylabel("P_bat (kW)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==========================
    # 9. Return all metrics
    # ==========================
    return {
        "E_Wh": E_Wh,
        "distance_km": dist_km,
        "Wh_per_km": Wh_per_km,
        "E_regen_Wh": E_regen_Wh,
        "regen_fraction": regen_fraction,
        "regen_peak_power_W": regen_peak_power_W,
        "SoC_trace": SoC.tolist(),
        "final_SoC": final_soc,
        "component_energy_Wh": component_energy_Wh,
        "grade_work_Wh": grade_work_Wh,
        "grade_assist_Wh": grade_assist_Wh,
        "undulation_Wh": grade_undulation_Wh,
        "total_mass_kg": total_mass,
    }
