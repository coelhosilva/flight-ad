from json import load
from pathlib import Path


def retrieve_json(json_path: Path):
    with open(json_path, 'r') as j:
        content = load(j)
    return content
