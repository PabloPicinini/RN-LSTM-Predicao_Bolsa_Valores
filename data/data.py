import sqlite3
import numpy as np
import psutil
from datetime import datetime
import pandas as pd
from  app.utils.prometheus.monitor import REQUEST_COUNT


def create_db():
    """
    Cria o banco de dados e as tabelas necessárias, caso não existam.

    Essa função realiza a criação de duas tabelas no banco de dados SQLite:
    1. `metrics_archive`: Armazena métricas relacionadas à utilização do sistema e previsões.
    2. `model_performance`: Armazena o desempenho do modelo, incluindo previsões e valores reais de fechamento.

    Retorna:
        None
    """
    conn = sqlite3.connect('data/metrics.db')
    cursor = conn.cursor()
    
    # Criar a tabela de métricas se não existir
    cursor.execute('''CREATE TABLE IF NOT EXISTS metrics_archive (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        request_count INTEGER,
                        cpu_usage REAL,
                        memory_usage REAL,
                        request_processing_seconds REAL,
                        forecast_horizon_1_day REAL,
                        forecast_horizon_3_days REAL,
                        forecast_horizon_7_days REAL,
                        forecast_horizon_15_days REAL
                    )''')
    
    # Criar a tabela de desempenho do modelo, se não existir
    cursor.execute('''CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT,
                        prediction_date TEXT,
                        target_date_1d TEXT,
                        target_date_3d TEXT,
                        target_date_7d TEXT,
                        target_date_15d TEXT,
                        predicted_close_1d REAL,
                        predicted_close_3d REAL,
                        predicted_close_7d REAL,
                        predicted_close_15d REAL,
                        real_close_1d REAL,
                        real_close_3d REAL,
                        real_close_7d REAL,
                        real_close_15d REAL
                    )''')
    
    conn.commit()
    conn.close()

def save_metrics_to_db(predictions, process_time):
    """
    Salva as métricas de uso do sistema e previsões no banco de dados.

    Coleta métricas como uso de CPU, memória, tempo de processamento da solicitação e previsões para diferentes horizontes de previsão,
    e insere esses dados na tabela `metrics_archive` do banco de dados.

    Args:
        predictions: Lista com as previsões para os horizontes de 1, 3, 7 e 15 dias.
        process_time: Tempo que levou para processar a previsão, em segundos.

    Retorna:
        None
    """

    request_count = REQUEST_COUNT._value.get() 
    cpu_usage = psutil.cpu_percent()  # Uso da CPU
    memory_usage = psutil.virtual_memory().percent  # Uso da memória

    # Timestamp atual
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Conectar ao banco de dados
    conn = sqlite3.connect('data/metrics.db')
    cursor = conn.cursor()

    # Inserir os dados na tabela
    cursor.execute('''INSERT INTO metrics_archive (
                        timestamp, 
                        request_count, 
                        cpu_usage, 
                        memory_usage, 
                        request_processing_seconds,
                        forecast_horizon_1_day, 
                        forecast_horizon_3_days, 
                        forecast_horizon_7_days, 
                        forecast_horizon_15_days
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        timestamp,
                        request_count,
                        cpu_usage,
                        memory_usage,
                        process_time,
                        predictions[0],
                        predictions[1],
                        predictions[2],
                        predictions[3]
                    ))

    # Salvar e fechar conexão
    conn.commit()
    conn.close()


def save_prediction_to_db(ticker, predictions, prediction_date):
    """
    Salva as previsões de fechamento e as metas no banco de dados, apenas em dias úteis.

    Args:
        ticker: Símbolo da ação para a qual as previsões estão sendo feitas.
        predictions: Lista com as previsões de fechamento para os horizontes de 1, 3, 7 e 15 dias.
        prediction_date: Data da previsão, que deve ser um dia útil.

    Retorna:
        bool: Retorna `True` se as previsões foram salvas com sucesso, `False` se já existirem previsões para essa data.
    """
    # Usar BDay para adicionar dias úteis apenas
    prediction_date = pd.Timestamp(prediction_date)
    
    # Verificar se o prediction_date é um dia de semana (segunda a sexta-feira)
    if prediction_date.weekday() >= 5:  # 5 = sábado, 6 = domingo
        print(f"Previsão para {ticker} na data {prediction_date.strftime('%Y-%m-%d')} não será inserida, pois é um final de semana.")
        return False

    target_dates = [
        (prediction_date + pd.offsets.BDay(i)).strftime('%Y-%m-%d')
        for i in [1, 3, 7, 15]
    ]
    
    # Conectar ao banco de dados
    conn = sqlite3.connect('data/metrics.db')
    cursor = conn.cursor()
    
    # Verificar se já existe uma previsão para essa data
    cursor.execute('''SELECT 1 FROM model_performance WHERE prediction_date = ? AND ticker = ?''', 
                   (prediction_date.strftime('%Y-%m-%d'), ticker))
    result = cursor.fetchone()

    # Se a data de previsão já existir, não insere nada
    if result:
        print(f"Previsão para {ticker} na data {prediction_date.strftime('%Y-%m-%d')} já existe no banco.")
        return False
    else:
        # Inserir as previsões e metas na tabela de desempenho do modelo
        cursor.execute('''INSERT INTO model_performance (
                            ticker, prediction_date, 
                            target_date_1d, target_date_3d, target_date_7d, target_date_15d,
                            predicted_close_1d, predicted_close_3d, predicted_close_7d, predicted_close_15d
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                            ticker, prediction_date.strftime('%Y-%m-%d'),
                            *target_dates,
                            predictions[0], predictions[1], predictions[2], predictions[3]
                        ))

        conn.commit()
    
    conn.close()

    return True



def update_real_close(ticker, prediction_date_str, real_close_value):
    """
    Atualiza os valores reais de fechamento no banco de dados.

    A função atualiza os valores de fechamento real para os horizontes de 1, 3, 7 e 15 dias, conforme a data de previsão,
    na tabela `model_performance` no banco de dados.

    Args:
        ticker: Símbolo da ação.
        prediction_date_str: Data da previsão.
        real_close_value: Valor real de fechamento da ação.

    Retorna:
        None
    """
    conn = sqlite3.connect('data/metrics.db')
    cursor = conn.cursor()

    # Atualizar a tabela 'model_performance' com o valor real do fechamento
    cursor.execute('''UPDATE model_performance
                      SET real_close_1d = CASE WHEN target_date_1d = ? THEN ? ELSE real_close_1d END,
                          real_close_3d = CASE WHEN target_date_3d = ? THEN ? ELSE real_close_3d END,
                          real_close_7d = CASE WHEN target_date_7d = ? THEN ? ELSE real_close_7d END,
                          real_close_15d = CASE WHEN target_date_15d = ? THEN ? ELSE real_close_15d END
                      WHERE ticker = ?''', (
        prediction_date_str, real_close_value,
        prediction_date_str, real_close_value,
        prediction_date_str, real_close_value,
        prediction_date_str, real_close_value,
        ticker
    ))

    # Commit das mudanças e fechamento da conexão
    conn.commit()
    conn.close()



def select_predictions(ticker):
    """
    Realiza a consulta das previsões e seus valores reais.
    
    Args:
        ticker (str): Símbolo do ticker.
    
    Retorna:
        DataFrame: Resultados da consulta.
    """
    conn = sqlite3.connect('data/metrics.db')
    cursor = conn.cursor()

    # Consulta SQL
    query = '''SELECT * FROM model_performance WHERE ticker = ?'''
    cursor.execute(query, (ticker,))

    # Obter resultados e fechar conexão
    columns = [description[0] for description in cursor.description]
    results = cursor.fetchall()
    conn.close()

    # Criar DataFrame
    df = pd.DataFrame(results, columns=columns)

    # Substituir valores inválidos (infinitos e NaN)
    df = df.replace([np.inf, -np.inf], None)  # Substitui infinitos por None
    df = df.where(pd.notnull(df), None)      # Substitui NaN por None

    # Opcional: converter explicitamente colunas com números para tipo "object"
    for col in df.columns:
        if df[col].dtype == np.float64:  # Apenas colunas float64
            df[col] = df[col].astype(object)


    return df