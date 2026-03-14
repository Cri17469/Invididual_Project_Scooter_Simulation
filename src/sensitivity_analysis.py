import argparse
from pathlib import Path

import numpy as np
import yaml

from utils import get_data_dir

DEFAULT_INPUT_FILENAME = "paired_differences.yaml"
DEFAULT_OUTPUT_FILENAME = "sensitivity_analysis.yaml"
DEFAULT_RUNS = 30
DEFAULT_CONFIDENCE = 0.95
SAMPLES_KEY = "optimized_regen_recovered_Wh"


def load_regen_samples(input_filename: str = DEFAULT_INPUT_FILENAME, key: str = SAMPLES_KEY) -> list[float]:
    input_path = get_data_dir() / input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {input_path}, got {type(data).__name__}.")

    if key not in data:
        raise KeyError(f"Missing key in YAML: {key}")

    values = np.asarray(data[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D list for key '{key}', got shape {values.shape}.")

    if values.size != DEFAULT_RUNS:
        raise ValueError(
            f"This sensitivity script is fixed to {DEFAULT_RUNS} groups, got {values.size} from '{key}'."
        )

    return values.astype(float).tolist()


def summarize_uncertainty(samples: np.ndarray, confidence: float = DEFAULT_CONFIDENCE) -> dict:
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Need at least 2 one-dimensional samples to estimate uncertainty.")

    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    stderr = float(std / np.sqrt(samples.size))

    # With 30 groups (df=29), t_0.975 ≈ 2.045 for a 95% CI.
    if np.isclose(confidence, 0.95):
        t_critical = 2.045
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
    input_filename: str = DEFAULT_INPUT_FILENAME,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> None:
    regen_samples = load_regen_samples(input_filename=input_filename, key=SAMPLES_KEY)

    summary = summarize_uncertainty(np.asarray(regen_samples, dtype=float), confidence=DEFAULT_CONFIDENCE)
    results = {
        "input_filename": input_filename,
        "source_key": SAMPLES_KEY,
        "runs": DEFAULT_RUNS,
        "optimized_regen_recovered_Wh_samples": regen_samples,
        "optimized_regen_recovered_Wh_uncertainty": summary,
    }

    output_path = save_results(results, output_filename=output_filename)

    print(f"Loaded {DEFAULT_RUNS} groups from '{input_filename}' key '{SAMPLES_KEY}'.")
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
            "Sensitivity analysis for optimized-route regenerated energy from "
            "data/paired_differences.yaml using exactly 30 groups."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_filename",
        default=DEFAULT_INPUT_FILENAME,
        help="Input YAML filename in data directory (default: paired_differences.yaml).",
    )
    parser.add_argument(
        "--output",
        dest="output_filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="YAML filename to save sensitivity results in data directory.",
    )

    args = parser.parse_args()

    main(
        input_filename=args.input_filename,
        output_filename=args.output_filename,
    )
