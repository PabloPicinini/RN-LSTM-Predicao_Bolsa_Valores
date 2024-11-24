import psutil
import time
from prometheus_client import Gauge
from threading import Thread
from app.utils.prometheus.monitor import CPU_USAGE, MEMORY_USAGE  # Importa as métricas diretamente

# Função para monitorar recursos
def monitor_resources():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        time.sleep(10)  # Coleta de métricas a cada 10 segundos

# Função para iniciar o monitoramento de recursos em uma thread separada
def start_resource_monitoring():
    resource_thread = Thread(target=monitor_resources, daemon=True)
    resource_thread.start()
