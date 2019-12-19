from pathlib import Path

def to_Path(path: str) -> Path:
    return Path(path).expanduser().resolve()
