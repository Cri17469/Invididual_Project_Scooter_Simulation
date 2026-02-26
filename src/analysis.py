import argparse
import math

import numpy as np
import yaml

from utils import get_data_dir

DEFAULT_INPUT_FILENAME = "paired_differences.yaml"
ALTERNATIVES = ("greater", "less")


def load_paired_differences(input_filename: str) -> dict:
    input_path = get_data_dir() / input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {input_path}, got {type(data).__name__}.")

    return data


def t_pdf(x: float, df: int) -> float:
    coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return coeff * (1 + (x * x) / df) ** (-(df + 1) / 2)


def simpson_integral_t_pdf(a: float, b: float, df: int, intervals: int = 10000) -> float:
    if intervals % 2 == 1:
        intervals += 1
    h = (b - a) / intervals

    s = t_pdf(a, df) + t_pdf(b, df)
    for i in range(1, intervals):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * t_pdf(x, df)

    return s * h / 3


def t_cdf(x: float, df: int) -> float:
    if x == 0:
        return 0.5
    if x > 0:
        return 0.5 + simpson_integral_t_pdf(0.0, x, df)
    return 0.5 - simpson_integral_t_pdf(0.0, -x, df)


def one_sided_p_value_from_t(t_stat: float, df: int, alternative: str) -> float:
    cdf_val = t_cdf(t_stat, df)
    if alternative == "greater":
        return max(0.0, min(1.0, 1.0 - cdf_val))
    if alternative == "less":
        return max(0.0, min(1.0, cdf_val))
    raise ValueError(f"Unsupported alternative: {alternative}")


def run_one_sided_paired_t_test(values: np.ndarray, alternative: str) -> dict:
    if values.ndim != 1:
        raise ValueError("Paired difference data must be one-dimensional.")
    if values.size < 2:
        raise ValueError("At least 2 paired samples are required for a t-test.")

    n = int(values.size)
    df = n - 1
    mean_diff = float(np.mean(values))
    std_diff = float(np.std(values, ddof=1))

    if math.isclose(std_diff, 0.0, abs_tol=1e-15):
        if math.isclose(mean_diff, 0.0, abs_tol=1e-15):
            t_stat = 0.0
        else:
            t_stat = math.copysign(math.inf, mean_diff)
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))

    if math.isfinite(t_stat):
        p_value = one_sided_p_value_from_t(t_stat, df=df, alternative=alternative)
    else:
        if alternative == "greater":
            p_value = 0.0 if t_stat > 0 else 1.0
        else:
            p_value = 0.0 if t_stat < 0 else 1.0

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def get_metric_array(data: dict, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key in YAML: {key}")

    values = np.asarray(data[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D list for key '{key}', got shape {values.shape}.")
    return values


def format_result(name: str, result: dict, alpha: float, alternative: str) -> str:
    reject_null = result["p_value"] < alpha
    decision = "Reject H0" if reject_null else "Fail to reject H0"
    if alternative == "greater":
        hypothesis = "H1: baseline - optimized > 0  (optimized < baseline)"
    else:
        hypothesis = "H1: baseline - optimized < 0  (optimized > baseline)"

    return (
        f"[{name}] one-sided paired t-test (alpha={alpha:.2f}, alternative='{alternative}')\n"
        f"  {hypothesis}\n"
        f"  n={result['n']}\n"
        f"  mean(diff)={result['mean_diff']:.6f} Wh\n"
        f"  std(diff)={result['std_diff']:.6f} Wh\n"
        f"  t={result['t_stat']:.6f}\n"
        f"  p={result['p_value']:.6g}\n"
        f"  decision: {decision}\n"
    )


def main(
    input_filename: str = DEFAULT_INPUT_FILENAME,
    alpha: float = 0.05,
    total_alternative: str = "greater",
    regen_alternative: str = "greater",
) -> None:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    data = load_paired_differences(input_filename)

    total_diff = get_metric_array(data, "total_energy_consumed_diff_Wh")
    regen_diff = get_metric_array(data, "regen_recovered_diff_Wh")

    total_result = run_one_sided_paired_t_test(total_diff, alternative=total_alternative)
    regen_result = run_one_sided_paired_t_test(regen_diff, alternative=regen_alternative)

    print(format_result("Total energy consumed", total_result, alpha, total_alternative))
    print(format_result("Regenerated energy", regen_result, alpha, regen_alternative))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run one-sided (single-tail) paired t-tests at alpha=0.05 by default on paired "
            "differences from data/paired_differences.yaml (where diff = baseline - optimized)."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_filename",
        default=DEFAULT_INPUT_FILENAME,
        help="Input YAML filename in data directory (default: paired_differences.yaml).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for hypothesis tests.")
    parser.add_argument(
        "--total-alternative",
        choices=ALTERNATIVES,
        default="greater",
        help=(
            "One-sided alternative for total_energy_consumed_diff_Wh (diff=baseline-optimized): "
            "greater => optimized < baseline, less => optimized > baseline."
        ),
    )
    parser.add_argument(
        "--regen-alternative",
        choices=ALTERNATIVES,
        default="greater",
        help=(
            "One-sided alternative for regen_recovered_diff_Wh (diff=baseline-optimized): "
            "greater => optimized < baseline, less => optimized > baseline."
        ),
    )

    args = parser.parse_args()

    main(
        input_filename=args.input_filename,
        alpha=args.alpha,
        total_alternative=args.total_alternative,
        regen_alternative=args.regen_alternative,
    )
