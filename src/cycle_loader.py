import yaml
import numpy as np
from pathlib import Path
from utils import get_data_dir

def load_drive_cycle(filename: str = "cycle.yaml") -> dict:
    data_dir = get_data_dir()
    file_path = data_dir / filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"[cycle_loader] Cannot read YAML: {e}")

    # === Strictly read only necessary fields ===
    try:
        t = np.array(data["time_s"], dtype=float)
        v = np.array(data["speed_kmh"], dtype=float) / 3.6
        grade_raw = data.get("grade_percent", [0.0] * len(t))
        g = np.array(grade_raw, dtype=float)

        # Check length consistency
        if not (len(t) == len(v) == len(g)):
            raise ValueError(
                f"[cycle_loader] Mismatched lengths: "
                f"time={len(t)}, speed={len(v)}, grade={len(g)}"
            )

        theta = np.arctan(g / 100.0)

    except KeyError as e:
        raise RuntimeError(f"[cycle_loader] Missing mandatory key: {e}")
    except Exception as e:
        raise RuntimeError(f"[cycle_loader] Invalid format: {e}")

    # === All other entries in the YAML are ignored ===
    return {
        "name": data.get("name", "cycle"),
        "description": data.get("description", ""),
        "t": t,
        "v": v,
        "theta": theta
    }
