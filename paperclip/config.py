"""
Manages local daemon config stored in ~/.paperclip/config.json
"""
import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".paperclip"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_API_URL = "http://localhost:8000"


def load() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def get(key: str, default=None):
    return load().get(key, default)


def set_value(key: str, value) -> None:
    cfg = load()
    cfg[key] = value
    save(cfg)


def api_url() -> str:
    return os.getenv("PAPERCLIP_API_URL") or get("api_url") or DEFAULT_API_URL


def device_token() -> Optional[str]:
    return os.getenv("PAPERCLIP_DEVICE_TOKEN") or get("device_token")


def is_mock() -> bool:
    return os.getenv("PAPERCLIP_MOCK", "").lower() in ("1", "true", "yes")
