from pathlib import Path

def get_data_dir() -> Path:
    """
    自动解析 data 目录（与 src 同级）。
    """
    try:
        # src 的上级路径就是项目根目录
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"未找到 data 文件夹: {data_dir}")
        return data_dir
    except Exception as e:
        print(f"[错误] 无法解析 data 路径: {e}")
        raise
