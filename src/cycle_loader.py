import yaml
import numpy as np
from pathlib import Path
from utils import get_data_dir

def load_drive_cycle(filename: str = "cycle.yaml") -> dict:
    """
    从 data/ 目录读取 drive cycle。
    """
    data_dir = get_data_dir()
    file_path = data_dir / filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] cycle data is not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] failed analyzing YAML: {e}")

    try:
        t = np.array(data["time_s"], dtype=float)
        v = np.array(data["speed_kmh"], dtype=float) / 3.6
        grades = data.get("grade_percent", [0.0] * len(t))
        theta = np.arctan(np.array(grades, dtype=float) / 100.0)
    except KeyError as e:
        raise KeyError(f"[ERROR] MISSING significant data: {e}")

    return {
        "name": data.get("name", "cycle"),
        "description": data.get("description", ""),
        "t": t, "v": v, "theta": theta
    }
