import csv
import json
from pathlib import Path

import yaml

from config import DEFAULT_LOCATION
from cycle_loader import load_drive_cycle
from energy_model import simulate_energy
from utils import get_data_dir
from vehicle_params import load_vehicle_params


BASELINE_CYCLE_FILENAME = "London_cycle_baseline.yaml"
OPTIMIZED_CYCLE_FILENAME = "London_cycle_optimized.yaml"
PARAMS_FILENAME = "scooter_params.yaml"
PLOT_ENERGY_TRACES = True
PLOT_COMPARISON = True
PLOT_ROUTES = True
COMPARISON_JSON = "comparison_summary.json"
COMPARISON_CSV = "comparison_summary.csv"


def summarize_result(label: str, cycle: dict, result: dict) -> dict:
    trip_time_s = float(cycle["t"][-1]) if len(cycle["t"]) else 0.0
    energy_recovered_Wh = float(result["E_regen_Wh"])
    net_energy_Wh = float(result["E_Wh"])
    total_consumed_Wh = net_energy_Wh + energy_recovered_Wh

    return {
        "label": label,
        "total_energy_consumed_Wh": total_consumed_Wh,
        "energy_recovered_Wh": energy_recovered_Wh,
        "net_energy_Wh": net_energy_Wh,
        "undulation_Wh": float(result["undulation_Wh"]),
        "wh_per_km": float(result["Wh_per_km"]),
        "trip_time_s": trip_time_s,
        "distance_km": float(result["distance_km"]),
    }


def percent_improvement(baseline: float, optimized: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - optimized) / baseline * 100.0


def save_comparison(outputs: list[dict]) -> tuple[Path, Path]:
    data_dir = get_data_dir()
    json_path = data_dir / COMPARISON_JSON
    csv_path = data_dir / COMPARISON_CSV

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=outputs[0].keys())
        writer.writeheader()
        writer.writerows(outputs)

    return json_path, csv_path


def print_summary(summary: dict) -> None:
    print(f"Simulation complete: {summary['label']}")
    print(
        "Energy consumed: "
        f"{summary['total_energy_consumed_Wh']:.1f} Wh | "
        f"Energy recovered: {summary['energy_recovered_Wh']:.1f} Wh | "
        f"Net energy: {summary['net_energy_Wh']:.1f} Wh"
    )
    print(
        f"Wh/km: {summary['wh_per_km']:.1f} | "
        f"Trip time: {summary['trip_time_s'] / 60:.1f} min | "
        f"Distance: {summary['distance_km']:.2f} km"
    )
    print(f"Energy loading (undulation): {summary['undulation_Wh']:.1f} Wh")


def plot_comparison_summary(summaries: list[dict]) -> None:
    import matplotlib.pyplot as plt

    labels = [summary["label"] for summary in summaries]
    net_energy = [summary["net_energy_Wh"] for summary in summaries]
    trip_time = [summary["trip_time_s"] / 60 for summary in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, net_energy, color=["#4C72B0", "#55A868"])
    axes[0].set_title("Net Energy (Wh)")
    axes[0].set_ylabel("Wh")

    axes[1].bar(labels, trip_time, color=["#4C72B0", "#55A868"])
    axes[1].set_title("Trip Time (min)")
    axes[1].set_ylabel("Minutes")

    plt.tight_layout()
    plt.show()


def load_route_coords(filename: str) -> tuple[list[float], list[float]]:
    data_dir = get_data_dir()
    file_path = data_dir / filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"[route_plot] Cannot read YAML: {e}")

    route_coords = data.get("route_coords", {})
    lat = route_coords.get("lat", [])
    lon = route_coords.get("lon", [])
    if not lat or not lon:
        raise RuntimeError(
            f"[route_plot] Missing route coordinates in {file_path}."
        )
    if len(lat) != len(lon):
        raise RuntimeError(
            f"[route_plot] Mismatched lat/lon lengths in {file_path}."
        )
    return lat, lon


def plot_route_comparison(
    baseline_filename: str,
    optimized_filename: str,
) -> None:
    import matplotlib.pyplot as plt

    baseline_lat, baseline_lon = load_route_coords(baseline_filename)
    optimized_lat, optimized_lon = load_route_coords(optimized_filename)

    plt.figure(figsize=(6, 6))
    plt.plot(baseline_lon, baseline_lat, label="Baseline route", color="#4C72B0")
    plt.plot(
        optimized_lon,
        optimized_lat,
        label="Optimized route",
        color="#55A868",
    )
    plt.scatter(
        [baseline_lon[0], baseline_lon[-1]],
        [baseline_lat[0], baseline_lat[-1]],
        color="#4C72B0",
        marker="o",
        s=30,
    )
    plt.scatter(
        [optimized_lon[0], optimized_lon[-1]],
        [optimized_lat[0], optimized_lat[-1]],
        color="#55A868",
        marker="o",
        s=30,
    )
    plt.title("Route comparison")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def resolve_cycle_filenames(
    location: str,
    baseline_cycle_filename: str | None = None,
    optimized_cycle_filename: str | None = None,
) -> tuple[str, str]:
    baseline = baseline_cycle_filename or f"{location}_cycle_baseline.yaml"
    optimized = optimized_cycle_filename or f"{location}_cycle_optimized.yaml"
    return baseline, optimized


def main(
    location: str = DEFAULT_LOCATION,
    baseline_cycle_filename: str | None = None,
    optimized_cycle_filename: str | None = None,
    params_filename: str = PARAMS_FILENAME,
) -> None:
    baseline_filename, optimized_filename = resolve_cycle_filenames(
        location,
        baseline_cycle_filename=baseline_cycle_filename,
        optimized_cycle_filename=optimized_cycle_filename,
    )
    try:
        baseline_cycle = load_drive_cycle(baseline_filename)
        optimized_cycle = load_drive_cycle(optimized_filename)
        params = load_vehicle_params(params_filename)

        baseline_result = simulate_energy(baseline_cycle, params, plot=PLOT_ENERGY_TRACES)
        optimized_result = simulate_energy(optimized_cycle, params, plot=PLOT_ENERGY_TRACES)

        baseline_summary = summarize_result(
            f"{baseline_cycle['name']} (baseline)",
            baseline_cycle,
            baseline_result,
        )
        optimized_summary = summarize_result(
            f"{optimized_cycle['name']} (optimized)",
            optimized_cycle,
            optimized_result,
        )

        net_improvement = percent_improvement(
            baseline_summary["net_energy_Wh"],
            optimized_summary["net_energy_Wh"],
        )
        wh_per_km_improvement = percent_improvement(
            baseline_summary["wh_per_km"],
            optimized_summary["wh_per_km"],
        )

        baseline_summary["net_energy_improvement_percent"] = 0.0
        baseline_summary["wh_per_km_improvement_percent"] = 0.0
        optimized_summary["net_energy_improvement_percent"] = net_improvement
        optimized_summary["wh_per_km_improvement_percent"] = wh_per_km_improvement

        print_summary(baseline_summary)
        print_summary(optimized_summary)

        print(f"Net energy improvement: {net_improvement:.1f}%")
        print(f"Wh/km improvement: {wh_per_km_improvement:.1f}%")

        json_path, csv_path = save_comparison([baseline_summary, optimized_summary])
        print(f"Saved comparison summary → {json_path}")
        print(f"Saved comparison summary → {csv_path}")

        if PLOT_COMPARISON:
            plot_comparison_summary([baseline_summary, optimized_summary])
        if PLOT_ROUTES:
            plot_route_comparison(baseline_filename, optimized_filename)

    except Exception as e:
        print(f"[Fatal error] Simulation failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare baseline and optimized drive cycles.")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Location label for cycle filenames.")
    parser.add_argument("--baseline-cycle", dest="baseline_cycle", help="Baseline cycle YAML filename.")
    parser.add_argument("--optimized-cycle", dest="optimized_cycle", help="Optimized cycle YAML filename.")
    parser.add_argument("--params", dest="params_filename", default=PARAMS_FILENAME, help="Vehicle params YAML.")
    args = parser.parse_args()

    main(
        location=args.location,
        baseline_cycle_filename=args.baseline_cycle,
        optimized_cycle_filename=args.optimized_cycle,
        params_filename=args.params_filename,
    )
