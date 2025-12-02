from cycle_loader import load_drive_cycle
from vehicle_params import load_vehicle_params
from energy_model import simulate_energy


def main():
    try:
        cycle = load_drive_cycle("cycle.yaml")
        params = load_vehicle_params("scooter_params.yaml")
        result = simulate_energy(cycle, params, plot=True)

        print(f"Simulation complete: {cycle['name']} — {cycle['description']}")
        print(f"Total mass (vehicle + rider): {result['total_mass_kg']:.1f} kg")
        print(f"Energy consumption: {result['E_Wh']:.1f} Wh | Distance: {result['distance_km']:.2f} km | "
              f"Specific energy consumption: {result['Wh_per_km']:.1f} Wh/km | Energy recovered: {result['E_regen_Wh']:.1f} Wh "
              f"({result['regen_fraction']*100:.1f}%) | Final SoC: {result['final_SoC']*100:.1f}%")

        comp = result["component_energy_Wh"]
        print("Energy contribution breakdown (propulsion only):")
        print(f"  • Rider mass & slope (grade): {comp['grade']:.1f} Wh")
        print(f"  • Rider mass & acceleration (inertial): {comp['inertial']:.1f} Wh")
        print(f"  • Rolling resistance: {comp['rolling']:.1f} Wh")
        print(f"  • Aerodynamic drag: {comp['aerodynamic']:.1f} Wh")
        print("Gravity effect (positive = climbing work, negative = downhill assist):")
        print(f"  • Net gravitational work: {result['grade_work_Wh']:.1f} Wh")
        print(f"  • Downhill assistance captured: {result['grade_assist_Wh']:.1f} Wh")

    except Exception as e:
        print(f"[Fatal error] Simulation failed: {e}")


if __name__ == "__main__":
    main()
