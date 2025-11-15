from cycle_loader import load_drive_cycle
from vehicle_params import load_vehicle_params
from energy_model import simulate_energy

def main():
    try:
        cycle = load_drive_cycle("cycle.yaml")
        params = load_vehicle_params("scooter_params.yaml")
        result = simulate_energy(cycle, params, plot=True)

        print(f"✅ 模拟完成: {cycle['name']} — {cycle['description']}")
        print(f"能耗: {result['E_Wh']:.1f} Wh | 距离: {result['distance_km']:.2f} km | "
              f"单位能耗: {result['Wh_per_km']:.1f} Wh/km")

    except Exception as e:
        print(f"[致命错误] 仿真失败: {e}")

if __name__ == "__main__":
    main()
