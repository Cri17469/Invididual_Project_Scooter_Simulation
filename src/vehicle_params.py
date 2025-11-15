import yaml
from dataclasses import dataclass
from pathlib import Path
from utils import get_data_dir

@dataclass
class VehicleParams:
    mass_kg: float
    wheel_radius_m: float
    CdA_m2: float
    Cr: float
    eta_m: float
    eta_regen: float
    Pchg_max: float
    Pdis_max: float
    Cap_Wh: float
    V_nom: float
    rho_air: float
    g: float
    soc0: float

def load_vehicle_params(filename: str = "scooter_params.yaml") -> VehicleParams:
    """
    从 data/ 目录读取车辆参数。
    """
    data_dir = get_data_dir()
    file_path = data_dir / filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[错误] 未找到车辆参数文件: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"[错误] 解析 YAML 文件失败: {e}")

    try:
        v, m, b, e = data["vehicle"], data["motor"], data["battery"], data["environment"]
    except KeyError as e:
        raise KeyError(f"[错误] YAML 文件中缺少字段: {e}")

    return VehicleParams(
        mass_kg=v["mass_kg"],
        wheel_radius_m=v["wheel_radius_m"],
        CdA_m2=v["CdA_m2"],
        Cr=v["Cr"],
        eta_m=m["efficiency"],
        eta_regen=m["regen_efficiency"],
        Pchg_max=b["charge_power_limit_W"],
        Pdis_max=b["discharge_power_limit_W"],
        Cap_Wh=b["capacity_Wh"],
        V_nom=b["nominal_voltage_V"],
        rho_air=e["air_density"],
        g=e["gravity"],
        soc0=b["initial_soc"],
    )
