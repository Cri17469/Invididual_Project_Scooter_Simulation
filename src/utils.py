from pathlib import Path


def get_data_dir() -> Path:
    """
    Automatically resolve the data directory (at the same level as src).
    """
    try:
        # The parent of src is the project root directory
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data folder not found: {data_dir}")
        return data_dir
    except Exception as e:
        print(f"[Error] Unable to resolve data path: {e}")
        raise

def get_params_dir() -> Path:
    """
    Automatically resolve the vehicle parameters directory (inside data).
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        params_dir = project_root / "params"
        if not params_dir.exists():
            raise FileNotFoundError(f"Vehicle parameters folder not found: {params_dir}")
        return params_dir
    except Exception as e:
        print(f"[Error] Unable to resolve vehicle parameters path: {e}")
        raise