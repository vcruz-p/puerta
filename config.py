import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"

DEFAULT_CONFIG = {
    "camera": {
        "ip": "192.168.1.100",
        "username": "root",
        "password": "",
        "rtsp_port": 554,
        "rtsp_path": "/axis-media/media.amp?videocodec=h264",
        "transport": "tcp"
    },
    "door": {
        "ip": "192.168.1.101",
        "port": 80,
        "open_endpoint": "/Api/Open/OpenDoor",
        "method": "GET",
        "timeout": 5
    },
    "automation": {
        "enabled": False,
        "standing_enabled": True,
        "standing_seconds": 4.0,
        "face_enabled": False,
        "gesture_enabled": False,
        "logic": "ANY",
        "yolo_confidence": 0.50,
        "face_threshold": 0.48,
        "gesture": "open_hand",
        "cooldown_seconds": 8.0,
        "face_capture_count": 5
    }
}


def _deep_copy_default():
    return json.loads(json.dumps(DEFAULT_CONFIG))


def load_config():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        cfg = _deep_copy_default()
        save_config(cfg)
        return cfg

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = _deep_copy_default()
        for section in ("camera", "door", "automation"):
            if isinstance(data.get(section), dict):
                cfg[section].update(data[section])

        return cfg
    except Exception:
        return _deep_copy_default()


def save_config(config):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def faces_dir():
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    return FACES_DIR
