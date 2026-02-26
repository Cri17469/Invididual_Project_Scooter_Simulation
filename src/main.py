import argparse
from pathlib import Path

import numpy as np
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


def perturb_cycle_speed(cycle: dict, speed_noise_std: float, rng: np.random.Generator) -> dict:
    """Return a copy of the cycle with Gaussian speed perturbation applied."""
    if speed_noise_std <= 0.0:
        return cycle

    perturbed_cycle = dict(cycle)
    speed = np.asarray(cycle["v"], dtype=float)
    noise = rng.normal(loc=0.0, scale=speed_noise_std, size=speed.shape)
    perturbed_cycle["v"] = np.clip(speed + noise, a_min=0.0, a_max=None)
    return perturbed_cycle


def run_paired_simulations(
    baseline_cycle: dict,
    optimized_cycle: dict,
    params,
    runs: int,
    plot: bool = False,
    speed_noise_std: float = 0.0,
    seed: int | None = None,
) -> dict:
    baseline_total_energy_consumed_wh: list[float] = []
    optimized_total_energy_consumed_wh: list[float] = []
    baseline_regen_recovered_wh: list[float] = []
    optimized_regen_recovered_wh: list[float] = []
    total_energy_consumed_diff_wh: list[float] = []
    regen_recovered_diff_wh: list[float] = []
    rng = np.random.default_rng(seed)

    for _ in range(runs):
        baseline_cycle_run = perturb_cycle_speed(baseline_cycle, speed_noise_std=speed_noise_std, rng=rng)
        optimized_cycle_run = perturb_cycle_speed(optimized_cycle, speed_noise_std=speed_noise_std, rng=rng)

        baseline_result = simulate_energy(baseline_cycle_run, params, plot=plot)
        optimized_result = simulate_energy(optimized_cycle_run, params, plot=plot)

        total_energy_diff = total_consumed_energy_wh(baseline_result) - total_consumed_energy_wh(
            optimized_result
        )
        regen_diff = float(baseline_result["E_regen_Wh"]) - float(optimized_result["E_regen_Wh"])

        baseline_total_energy_consumed_wh.append(total_consumed_energy_wh(baseline_result))
        optimized_total_energy_consumed_wh.append(total_consumed_energy_wh(optimized_result))
        baseline_regen_recovered_wh.append(float(baseline_result["E_regen_Wh"]))
        optimized_regen_recovered_wh.append(float(optimized_result["E_regen_Wh"]))

        total_energy_consumed_diff_wh.append(total_energy_diff)
        regen_recovered_diff_wh.append(regen_diff)

    return {
        "baseline_total_energy_consumed_Wh": baseline_total_energy_consumed_wh,
        "optimized_total_energy_consumed_Wh": optimized_total_energy_consumed_wh,
        "baseline_regen_recovered_Wh": baseline_regen_recovered_wh,
        "optimized_regen_recovered_Wh": optimized_regen_recovered_wh,
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
    speed_noise_std: float = 0.0,
    seed: int | None = None,
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
        speed_noise_std=speed_noise_std,
        seed=seed,
    )
    output_path = save_differences(differences, output_filename=output_filename)

    print(f"Completed {runs} paired simulations for location '{location}'.")
    print(f"Saved paired differences -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run paired baseline/optimized simulations and save both per-run raw values "
            "and paired differences for total consumed energy and regenerated energy."
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
    parser.add_argument(
        "--speed-noise-std",
        type=float,
        default=0.0,
        help=(
            "Gaussian speed perturbation standard deviation in m/s applied independently each run. "
            "Use a value > 0 to make repeated runs stochastic."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible noise.")

    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be a positive integer.")
    if args.speed_noise_std < 0:
        raise ValueError("--speed-noise-std must be non-negative.")

    main(
        location=args.location,
        baseline_cycle_filename=args.baseline_cycle,
        optimized_cycle_filename=args.optimized_cycle,
        params_filename=args.params_filename,
        runs=args.runs,
        output_filename=args.output_filename,
        plot=args.plot,
        speed_noise_std=args.speed_noise_std,
        seed=args.seed,
    )
