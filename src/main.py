from cycle_loader import load_drive_cycle
from vehicle_params import load_vehicle_params
from energy_model import simulate_energy


def main():
    try:
        cycle = load_drive_cycle("cambridge.yaml")
        params = load_vehicle_params("scooter_params.yaml")
        result = simulate_energy(cycle, params, plot=True)

        print(f"Simulation complete: {cycle['name']} — {cycle['description']}")
        print(f"Energy consumption: {result['E_Wh']:.1f} Wh | Distance: {result['distance_km']:.2f} km | "
              f"Specific energy consumption: {result['Wh_per_km']:.1f} Wh/km | Energy recovered: {result['E_regen_Wh']:.1f} Wh "
              f"({result['regen_fraction']*100:.1f}%) | Final SoC: {result['final_SoC']*100:.1f}%")

    except Exception as e:
        print(f"[Fatal error] Simulation failed: {e}")


if __name__ == "__main__":
    main()
