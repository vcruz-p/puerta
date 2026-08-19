import requests


class DoorController:
    """Controlador de puerta. Solo existe operación de APERTURA."""

    def __init__(self, config, logger=None):
        self.cfg = config
        self.logger = logger or (lambda msg: None)

    def update_config(self, config):
        self.cfg = config

    def _url(self):
        d = self.cfg["door"]
        endpoint = str(d.get("open_endpoint", "")).strip()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        return f"http://{d['ip']}:{d['port']}{endpoint}"

    def open(self):
        d = self.cfg["door"]
        method = str(d.get("method", "GET")).upper()
        timeout = float(d.get("timeout", 5))
        url = self._url()

        self.logger(f"{method} {url}")

        if method != "GET":
            raise ValueError("La puerta está configurada únicamente para GET.")

        response = requests.get(url, timeout=timeout)
        text = response.text[:500] if response.text else ""
        return response.status_code, text
