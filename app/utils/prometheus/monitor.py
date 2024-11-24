from prometheus_client import Summary, Counter, Gauge, start_http_server, CollectorRegistry
import time
from app.utils.constants.settings import PROMETHEUS_PORT

# Usando um único registro para todas as métricas
registry = CollectorRegistry()

# Métricas do Prometheus
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request', registry=registry)
REQUEST_COUNT = Counter('request_count', 'App Request Count', registry=registry)
CPU_USAGE = Gauge('cpu_usage', 'CPU Usage', registry=registry)
MEMORY_USAGE = Gauge('memory_usage', 'Memory Usage', registry=registry)

# Novas métricas para predições
PREDICTION_COUNT = Counter('prediction_count', 'Total number of predictions made', registry=registry)
PREDICTION_TIME = Summary('prediction_processing_seconds', 'Time spent processing predictions', registry=registry)

# Métricas para cada horizonte de previsão
LAST_PREDICTION_1_DAY = Gauge('last_prediction_1_day', 'Prediction for the 1-day horizon', registry=registry)
LAST_PREDICTION_3_DAYS = Gauge('last_prediction_3_days', 'Prediction for the 3-day horizon', registry=registry)
LAST_PREDICTION_7_DAYS = Gauge('last_prediction_7_days', 'Prediction for the 7-day horizon', registry=registry)
LAST_PREDICTION_15_DAYS = Gauge('last_prediction_15_days', 'Prediction for the 15-day horizon', registry=registry)

# Função para iniciar o servidor de métricas do Prometheus
def start_prometheus():
    start_http_server(PROMETHEUS_PORT, registry=registry)

# Middleware para registrar tempo de resposta
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_TIME.observe(process_time)
    REQUEST_COUNT.inc()
    
    return response
