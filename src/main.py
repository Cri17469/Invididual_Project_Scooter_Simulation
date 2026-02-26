import argparse
from pathlib import Path

import yaml

from config import DEFAULT_LOCATION
from cycle_loader import load_drive_cycle
from energy_model import simulate_energy
from utils import get_data_dir
from vehicle_params import load_vehicle_params


PARAMS_FILENAME = "scooter_params.yaml"
DEFAULT_RUNS = 30
OUTPUT_FILENAME = "paired_differences.yaml"


def resolve_cycle_filenames(
    location: str,
    baseline_cycle_filename: str | None = None,
    optimized_cycle_filename: str | None = None,
) -> tuple[str, str]:
    baseline = baseline_cycle_filename or f"{location}_cycle_baseline.yaml"
    optimized = optimized_cycle_filename or f"{location}_cycle_optimized.yaml"
    return baseline, optimized


def total_consumed_energy_wh(result: dict) -> float:
    return float(result["E_Wh"]) + float(result["E_regen_Wh"])


def run_paired_simulations(
    baseline_cycle: dict,
    optimized_cycle: dict,
    params,
    runs: int,
    plot: bool = False,
) -> dict:
    total_energy_consumed_diff_wh: list[float] = []
    regen_recovered_diff_wh: list[float] = []

    for _ in range(runs):
        baseline_result = simulate_energy(baseline_cycle, params, plot=plot)
        optimized_result = simulate_energy(optimized_cycle, params, plot=plot)

        total_energy_diff = total_consumed_energy_wh(baseline_result) - total_consumed_energy_wh(
            optimized_result
        )
        regen_diff = float(baseline_result["E_regen_Wh"]) - float(optimized_result["E_regen_Wh"])

        total_energy_consumed_diff_wh.append(total_energy_diff)
        regen_recovered_diff_wh.append(regen_diff)

    return {
        "total_energy_consumed_diff_Wh": total_energy_consumed_diff_wh,
        "regen_recovered_diff_Wh": regen_recovered_diff_wh,
    }


def save_differences(differences: dict, output_filename: str = OUTPUT_FILENAME) -> Path:
    data_dir = get_data_dir()
    output_path = data_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(differences, f, sort_keys=False, allow_unicode=True)
    return output_path


def main(
    location: str = DEFAULT_LOCATION,
    baseline_cycle_filename: str | None = None,
    optimized_cycle_filename: str | None = None,
    params_filename: str = PARAMS_FILENAME,
    runs: int = DEFAULT_RUNS,
    output_filename: str = OUTPUT_FILENAME,
    plot: bool = False,
) -> None:
    baseline_filename, optimized_filename = resolve_cycle_filenames(
        location,
        baseline_cycle_filename=baseline_cycle_filename,
        optimized_cycle_filename=optimized_cycle_filename,
    )

    baseline_cycle = load_drive_cycle(baseline_filename)
    optimized_cycle = load_drive_cycle(optimized_filename)
    params = load_vehicle_params(params_filename)

    differences = run_paired_simulations(
        baseline_cycle=baseline_cycle,
        optimized_cycle=optimized_cycle,
        params=params,
        runs=runs,
        plot=plot,
    )
    output_path = save_differences(differences, output_filename=output_filename)

    print(f"Completed {runs} paired simulations for location '{location}'.")
    print(f"Saved paired differences -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run paired baseline/optimized simulations and save only paired differences "
            "for total consumed energy and regenerated energy."
        )
    )
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Location label for cycle filenames.")
    parser.add_argument("--baseline-cycle", dest="baseline_cycle", help="Baseline cycle YAML filename.")
    parser.add_argument("--optimized-cycle", dest="optimized_cycle", help="Optimized cycle YAML filename.")
    parser.add_argument("--params", dest="params_filename", default=PARAMS_FILENAME, help="Vehicle params YAML.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of paired runs to perform.")
    parser.add_argument(
        "--output",
        dest="output_filename",
        default=OUTPUT_FILENAME,
        help="YAML filename to save paired differences in data directory.",
    )
    parser.add_argument("--plot", action="store_true", help="Enable per-run energy trace plotting.")

    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be a positive integer.")

    main(
        location=args.location,
        baseline_cycle_filename=args.baseline_cycle,
        optimized_cycle_filename=args.optimized_cycle,
        params_filename=args.params_filename,
        runs=args.runs,
        output_filename=args.output_filename,
        plot=args.plot,
    )
