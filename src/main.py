import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
VELOCITY_PLOT_FILENAME = "velocity.jpg"
ELEVATION_PLOT_FILENAME = "net_elevation_vs_distance.jpg"


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


def save_velocity_plot(
    cycle: dict,
    speed_noise_std: float,
    rng: np.random.Generator,
    output_filename: str = VELOCITY_PLOT_FILENAME,
) -> Path:
    """Save a speed comparison plot for original vs stochastic velocity."""
    time_s = np.asarray(cycle["t"], dtype=float)
    original_velocity_ms = np.asarray(cycle["v"], dtype=float)
    perturbed_velocity_ms = np.asarray(
        perturb_cycle_speed(cycle, speed_noise_std=speed_noise_std, rng=rng)["v"],
        dtype=float,
    )
    if original_velocity_ms.size == 0:
        pure_sine_reference_velocity_ms = original_velocity_ms
    else:
        cycles = 4.0
        duration_s = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 1.0
        phase = 2.0 * np.pi * cycles * (time_s - time_s[0]) / max(duration_s, np.finfo(float).eps)
        min_velocity_ms = float(np.min(original_velocity_ms))
        max_velocity_ms = float(np.max(original_velocity_ms))
        mean_velocity_ms = 0.5 * (max_velocity_ms + min_velocity_ms)
        amplitude_ms = 0.5 * (max_velocity_ms - min_velocity_ms)
        pure_sine_reference_velocity_ms = mean_velocity_ms + amplitude_ms * np.sin(phase)

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, original_velocity_ms * 3.6, label="Original cycle velocity", linewidth=2)
    plt.plot(
        time_s,
        perturbed_velocity_ms * 3.6,
        label="Velocity with stochastic perturbation",
        linewidth=2,
        alpha=0.85,
    )
    plt.plot(
        time_s,
        pure_sine_reference_velocity_ms * 3.6,
        linestyle="--",
        linewidth=1.8,
        color="black",
        alpha=0.75,
        label="Pure sine reference",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title("Velocity profile comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = get_data_dir() / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def _distance_and_elevation_from_cycle(cycle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct cumulative distance (km) and relative elevation (m) from cycle dynamics."""
    time_s = np.asarray(cycle["t"], dtype=float)
    velocity_ms = np.asarray(cycle["v"], dtype=float)
    theta_rad = np.asarray(cycle["theta"], dtype=float)

    if time_s.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if time_s.size == 1:
        return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)

    delta_t_s = np.diff(time_s)
    segment_velocity_ms = 0.5 * (velocity_ms[:-1] + velocity_ms[1:])
    segment_distance_m = np.clip(segment_velocity_ms * delta_t_s, a_min=0.0, a_max=None)
    cumulative_distance_m = np.concatenate(([0.0], np.cumsum(segment_distance_m)))

    segment_theta_rad = 0.5 * (theta_rad[:-1] + theta_rad[1:])
    segment_delta_h_m = segment_distance_m * np.tan(segment_theta_rad)
    cumulative_elevation_m = np.concatenate(([0.0], np.cumsum(segment_delta_h_m)))

    return cumulative_distance_m / 1000.0, cumulative_elevation_m


def _undulation_indicator(cumulative_elevation_m: np.ndarray) -> float:
    if cumulative_elevation_m.size < 2:
        return 0.0
    elevation_steps_m = np.diff(cumulative_elevation_m)
    return float(np.sum(np.abs(elevation_steps_m)))


def save_net_elevation_plot(
    baseline_cycle: dict,
    optimized_cycle: dict,
    output_filename: str = ELEVATION_PLOT_FILENAME,
) -> Path:
    """Save a net elevation vs distance plot with undulation shading."""
    baseline_distance_km, baseline_elevation_m = _distance_and_elevation_from_cycle(baseline_cycle)
    optimized_distance_km, optimized_elevation_m = _distance_and_elevation_from_cycle(optimized_cycle)

    def _net_line(distance_km: np.ndarray, elevation_m: np.ndarray) -> np.ndarray:
        if elevation_m.size == 0:
            return elevation_m
        return np.linspace(elevation_m[0], elevation_m[-1], elevation_m.size)

    baseline_net_line_m = _net_line(baseline_distance_km, baseline_elevation_m)
    optimized_net_line_m = _net_line(optimized_distance_km, optimized_elevation_m)

    baseline_undulation_m = _undulation_indicator(baseline_elevation_m)
    optimized_undulation_m = _undulation_indicator(optimized_elevation_m)
    baseline_net_change_m = (
        float(baseline_elevation_m[-1] - baseline_elevation_m[0]) if baseline_elevation_m.size else 0.0
    )
    optimized_net_change_m = (
        float(optimized_elevation_m[-1] - optimized_elevation_m[0]) if optimized_elevation_m.size else 0.0
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(baseline_distance_km, baseline_elevation_m, color="#1f77b4", linewidth=2.2, label="Route A (baseline)")
    ax.plot(
        optimized_distance_km,
        optimized_elevation_m,
        color="#d62728",
        linewidth=2.2,
        label="Route B (optimized)",
    )

    ax.plot(
        baseline_distance_km,
        baseline_net_line_m,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label=f"Route A net Δh = {baseline_net_change_m:.1f} m",
    )
    ax.plot(
        optimized_distance_km,
        optimized_net_line_m,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label=f"Route B net Δh = {optimized_net_change_m:.1f} m",
    )

    ax.fill_between(
        baseline_distance_km,
        baseline_elevation_m,
        baseline_net_line_m,
        color="#1f77b4",
        alpha=0.18,
        label=f"Route A undulation indicator = {baseline_undulation_m:.1f} m",
    )
    ax.fill_between(
        optimized_distance_km,
        optimized_elevation_m,
        optimized_net_line_m,
        color="#d62728",
        alpha=0.18,
        label=f"Route B undulation indicator = {optimized_undulation_m:.1f} m",
    )

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Relative elevation (m)")
    ax.set_title("Net elevation versus distance with undulation shading")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_path = get_data_dir() / output_filename
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
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
    rng = np.random.default_rng(seed)

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
    velocity_plot_path = save_velocity_plot(
        baseline_cycle,
        speed_noise_std=speed_noise_std,
        rng=rng,
    )
    elevation_plot_path = save_net_elevation_plot(
        baseline_cycle=baseline_cycle,
        optimized_cycle=optimized_cycle,
    )

    print(f"Completed {runs} paired simulations for location '{location}'.")
    print(f"Saved paired differences -> {output_path}")
    print(f"Saved velocity comparison plot -> {velocity_plot_path}")
    print(f"Saved net elevation comparison plot -> {elevation_plot_path}")


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
