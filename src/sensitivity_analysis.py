import argparse
from pathlib import Path

import numpy as np
import yaml

from cycle_loader import load_drive_cycle
from energy_model import simulate_energy
from utils import get_data_dir
from vehicle_params import load_vehicle_params

PARAMS_FILENAME = "scooter_params.yaml"
DEFAULT_LOCATION = "London"
DEFAULT_RUNS = 20
DEFAULT_SPEED_NOISE_STD = 0.2
DEFAULT_OUTPUT_FILENAME = "sensitivity_analysis.yaml"


def resolve_cycle_filenames(
    location: str,
    baseline_cycle_filename: str | None = None,
    optimized_cycle_filename: str | None = None,
) -> tuple[str, str]:
    baseline = baseline_cycle_filename or f"{location}_cycle_baseline.yaml"
    optimized = optimized_cycle_filename or f"{location}_cycle_optimized.yaml"
    return baseline, optimized


def perturb_cycle_speed(cycle: dict, speed_noise_std: float, rng: np.random.Generator) -> dict:
    if speed_noise_std <= 0.0:
        return cycle

    perturbed_cycle = dict(cycle)
    speed = np.asarray(cycle["v"], dtype=float)
    noise = rng.normal(loc=0.0, scale=speed_noise_std, size=speed.shape)
    perturbed_cycle["v"] = np.clip(speed + noise, a_min=0.0, a_max=None)
    return perturbed_cycle


def run_optimized_regen_sensitivity(
    optimized_cycle: dict,
    params,
    runs: int = DEFAULT_RUNS,
    speed_noise_std: float = DEFAULT_SPEED_NOISE_STD,
    seed: int | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    optimized_regen_recovered_wh: list[float] = []

    for _ in range(runs):
        optimized_cycle_run = perturb_cycle_speed(optimized_cycle, speed_noise_std=speed_noise_std, rng=rng)
        optimized_result = simulate_energy(optimized_cycle_run, params, plot=False)
        optimized_regen_recovered_wh.append(float(optimized_result["E_regen_Wh"]))

    return optimized_regen_recovered_wh


def summarize_uncertainty(samples: np.ndarray, confidence: float = 0.95) -> dict:
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Need at least 2 one-dimensional samples to estimate uncertainty.")

    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    stderr = float(std / np.sqrt(samples.size))

    # With 20 runs (df=19), t_0.975 ≈ 2.093 for a 95% CI.
    if np.isclose(confidence, 0.95):
        t_critical = 2.093
    else:
        # Fallback to normal approximation for non-default confidence levels.
        t_critical = 1.96

    half_width = float(t_critical * stderr)

    return {
        "n": int(samples.size),
        "mean_Wh": mean,
        "std_Wh": std,
        "stderr_Wh": stderr,
        "confidence": confidence,
        "ci_half_width_Wh": half_width,
        "ci_lower_Wh": float(mean - half_width),
        "ci_upper_Wh": float(mean + half_width),
    }


def save_results(results: dict, output_filename: str = DEFAULT_OUTPUT_FILENAME) -> Path:
    output_path = get_data_dir() / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, allow_unicode=True)
    return output_path


def main(
    location: str = DEFAULT_LOCATION,
    baseline_cycle_filename: str | None = None,
    optimized_cycle_filename: str | None = None,
    params_filename: str = PARAMS_FILENAME,
    runs: int = DEFAULT_RUNS,
    speed_noise_std: float = DEFAULT_SPEED_NOISE_STD,
    seed: int | None = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> None:
    if runs != DEFAULT_RUNS:
        raise ValueError(f"This sensitivity script is fixed to {DEFAULT_RUNS} runs.")
    if speed_noise_std < 0:
        raise ValueError("--speed-noise-std must be non-negative.")

    _, optimized_filename = resolve_cycle_filenames(
        location,
        baseline_cycle_filename=baseline_cycle_filename,
        optimized_cycle_filename=optimized_cycle_filename,
    )

    optimized_cycle = load_drive_cycle(optimized_filename)
    params = load_vehicle_params(params_filename)

    regen_samples = run_optimized_regen_sensitivity(
        optimized_cycle=optimized_cycle,
        params=params,
        runs=runs,
        speed_noise_std=speed_noise_std,
        seed=seed,
    )

    summary = summarize_uncertainty(np.asarray(regen_samples, dtype=float), confidence=0.95)
    results = {
        "location": location,
        "optimized_cycle_filename": optimized_filename,
        "params_filename": params_filename,
        "runs": runs,
        "speed_noise_std": speed_noise_std,
        "seed": seed,
        "optimized_regen_recovered_Wh_samples": regen_samples,
        "optimized_regen_recovered_Wh_uncertainty": summary,
    }

    output_path = save_results(results, output_filename=output_filename)

    print(f"Completed {runs} noisy simulations for optimized route: '{optimized_filename}'.")
    print("Optimized route regenerated energy (Wh):")
    print(f"  mean = {summary['mean_Wh']:.6f}")
    print(f"  std  = {summary['std_Wh']:.6f}")
    print(
        f"  95% CI = [{summary['ci_lower_Wh']:.6f}, {summary['ci_upper_Wh']:.6f}] "
        f"(±{summary['ci_half_width_Wh']:.6f})"
    )
    print(f"Saved sensitivity summary -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for optimized-route regenerated energy: run exactly 20 "
            "noisy simulations and report uncertainty metrics."
        )
    )
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Location label for cycle filenames.")
    parser.add_argument("--baseline-cycle", dest="baseline_cycle", help="Baseline cycle YAML filename (optional).")
    parser.add_argument("--optimized-cycle", dest="optimized_cycle", help="Optimized cycle YAML filename.")
    parser.add_argument("--params", dest="params_filename", default=PARAMS_FILENAME, help="Vehicle params YAML.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Must be 20 for this script.")
    parser.add_argument(
        "--speed-noise-std",
        type=float,
        default=DEFAULT_SPEED_NOISE_STD,
        help="Gaussian speed perturbation standard deviation in m/s (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible noise.")
    parser.add_argument(
        "--output",
        dest="output_filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="YAML filename to save sensitivity results in data directory.",
    )

    args = parser.parse_args()

    main(
        location=args.location,
        baseline_cycle_filename=args.baseline_cycle,
        optimized_cycle_filename=args.optimized_cycle,
        params_filename=args.params_filename,
        runs=args.runs,
        speed_noise_std=args.speed_noise_std,
        seed=args.seed,
        output_filename=args.output_filename,
    )
