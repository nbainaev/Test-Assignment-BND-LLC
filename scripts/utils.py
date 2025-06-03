from pathlib import Path
from typing import Any
from ruamel.yaml import YAML

def read_config(filepath: str) -> Any:
    """Функция для чтения файла конфигурации."""
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r', encoding='utf-8') as config_io:
        yaml = YAML()
        return yaml.load(config_io)