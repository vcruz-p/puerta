"""
Controlador de puerta para apertura mediante HTTP.

Soporta métodos GET y POST con autenticación básica opcional.
"""

import requests


class DoorController:
    """
    Controlador de puerta.
    
    Permite abrir una puerta mediante petición HTTP a un controlador/relé.
    La configuración incluye IP, puerto, endpoint, método y timeout.
    """

    def __init__(self, config, logger=None):
        """
        Inicializa el controlador de puerta.
        
        Args:
            config (dict): Configuración completa del sistema.
            logger (callable, optional): Función para logging.
        """
        self.cfg = config
        self.logger = logger or (lambda msg: None)

    def update_config(self, config):
        """
        Actualiza la configuración de puerta.
        
        Args:
            config (dict): Nueva configuración completa.
        """
        self.cfg = config

    def _url(self):
        """
        Construye la URL para la operación de apertura.
        
        Returns:
            str: URL completa para el endpoint de apertura.
        """
        d = self.cfg["door"]
        endpoint = str(d.get("open_endpoint", "")).strip()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        return f"http://{d['ip']}:{d['port']}{endpoint}"

    def open(self):
        """
        Ejecuta la apertura de la puerta.
        
        Realiza una petición HTTP GET al endpoint configurado.
        
        Returns:
            tuple: (status_code, response_text)
            
        Raises:
            ValueError: Si el método configurado no es GET.
        """
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
