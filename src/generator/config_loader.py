import json
from typing import Any, Dict


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)
