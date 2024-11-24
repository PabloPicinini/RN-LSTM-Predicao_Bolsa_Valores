from app.utils.prometheus.monitor import (
    PREDICTION_COUNT,
    PREDICTION_TIME,
    LAST_PREDICTION_1_DAY,
    LAST_PREDICTION_3_DAYS,
    LAST_PREDICTION_7_DAYS,
    LAST_PREDICTION_15_DAYS,
)
from data.data import save_metrics_to_db 

# Dicionário para mapeamento das métricas por horizonte
LAST_PREDICTIONS = {
    1: LAST_PREDICTION_1_DAY,
    3: LAST_PREDICTION_3_DAYS,
    7: LAST_PREDICTION_7_DAYS,
    15: LAST_PREDICTION_15_DAYS,
}

def update_prediction_metrics(predictions, process_time):
    """
    Atualiza as métricas do Prometheus para as previsões e as salva no banco de dados.

    Args:
        predictions: Lista com os valores das previsões para os horizontes de 1, 3, 7 e 15 dias.
        process_time: O tempo que levou para processar a previsão, em segundos, utilizado para atualizar a métrica de tempo.

    Returns:
        None
    """
    PREDICTION_COUNT.inc()  # Incrementa o contador de predições
    PREDICTION_TIME.observe(process_time)  # Observa o tempo de processamento

    # Atualiza as métricas para cada horizonte
    for horizon, value in zip(LAST_PREDICTIONS.keys(), predictions):
        LAST_PREDICTIONS[horizon].set(value)
    
    # Salvar as métricas e as predições no Banco de Dados
    save_metrics_to_db(predictions, process_time)
