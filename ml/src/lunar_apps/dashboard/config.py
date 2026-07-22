import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "config.yaml"

def _load_config():
    """Reads the YAML file and returns it as a Python dictionary."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Could not find config file at {CONFIG_PATH}")
        
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

CONFIG = _load_config()
