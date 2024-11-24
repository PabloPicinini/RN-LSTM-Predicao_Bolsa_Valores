from fastapi import HTTPException, UploadFile, File, APIRouter
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List
import time
from datetime import datetime, timedelta

from requests import request

from app.utils.data_utils import update_real_data
from app.utils.prometheus.resources import start_resource_monitoring
from app.utils.prometheus.monitor import start_prometheus
from app.utils.prometheus.metrics_manager import update_prediction_metrics

from app.utils.y_finance_utils import ult_60_close
from app.utils.ml_utils import (
    process_csv, 
    load_model_and_scaler, 
    process_prediction
)
from data.data import (
    create_db, 
    save_prediction_to_db,
    select_predictions
)

# Inicialização
create_db() # Criar o banco de dados e a tabela ao iniciar

router = APIRouter()
start_prometheus()
start_resource_monitoring()

# Carrega o modelo e o scaler
model, scaler = load_model_and_scaler()

# Classe para definir o formato dos dados de entrada
class PriceData(BaseModel):
    Open: List[float]
    High: List[float]
    Low: List[float]
    Volume: List[float]
    Close: List[float]

# Endpoint de previsão
@router.post("/predict_archive",
             tags=['CSV'],
             summary='Obter Predição a partir de dados históricos fornecidos', 
             description=   "Função para carregar e processar um arquivo CSV contendo dados financeiros.\n\n"
                            "O arquivo CSV deve seguir o formato esperado com as seguintes colunas obrigatórias:\n\n"
                            "- **Open**: Preço de abertura.\n"
                            "- **High**: Maior preço do dia.\n"
                            "- **Low**: Menor preço do dia.\n"
                            "- **Volume**: Volume negociado.\n"
                            "- **Close**: Preço de fechamento.\n\n"
                            "Requisitos:\n"
                            "- O CSV deve conter **pelo menos 60 registros** para o processamento.\n"
                            "- Arquivo com separador padrão de vírgula (`,`).\n\n"
                            "Retorna um dicionário contendo as últimas 60 linhas das colunas obrigatórias. "
                            "Caso o CSV não contenha as colunas esperadas ou tenha menos de 60 registros, será retornado um erro."
            )
async def predict(file: UploadFile = File(...)):
    start_time = time.time() # Inicia a contagem do tempo de execução
    data_dict = process_csv(file.file) # Processa o arquivo CSV para obter o dicionário de dados
    predictions = process_prediction(data_dict, model, scaler, len(data_dict.keys()))
    update_prediction_metrics(predictions, time.time() - start_time) # Atualizando as métricas do Prometheus

    # Retornando as previsões
    return {f"forecast_horizon_{h}_day": p for h, p in zip([1, 3, 7, 15], predictions)}


# Endpoint para previsão usando dados do Yahoo Finance
@router.get("/predict_today",
            tags=['Diária'],
            summary='Obter Predição do dia atual', 
            description='Retorna a predição de 1, 3, 7 e 15 Dias do dia atual e armazena no banco de dados as predições e valores reais para medir a performance ao longo do tempo')
async def predict_today():
    start_time = time.time() # Inicia a contagem do tempo de execução

    # Baixar os dados históricos do Yahoo Finance
    df, df_model = ult_60_close() # df contendo a data, e df_model retirando a coluna Data
    predictions = process_prediction(df_model, model, scaler, len(df_model.columns))

    prediction_date = datetime.today().strftime("%Y-%m-%d")

    ticker = 'BBAS3.SA'
    save_prediction_to_db(ticker, predictions, prediction_date) # Salvando as Predições no Banco
    update_real_data(df, ticker) # Realizando o Update dos valores reais no Banco
    update_prediction_metrics(predictions, time.time() - start_time) # Atualizando as métricas do Prometheus

    # Retornando as previsões
    return {f"forecast_horizon_{h}_day": p for h, p in zip([1, 3, 7, 15], predictions)}



@router.get('/get_predictions',
            tags=['Buscar Predições'],
            summary='Consultar as Predições no Banco de Dados',
            description='Retorna a consulta do Banco de Dados o qual foi salvo utilizando as predições diárias')
async def get_predictions():
    """
    Endpoint para obter as previsões com base no ticker fornecido.
    """
    ticker = 'BBAS3.SA'

    try:
        # Consultar dados
        data = select_predictions(ticker)

        # Verificar se há dados
        if data.empty:
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado para o ticker fornecido")

        # Substituir valores inválidos por None
        data = data.replace([np.inf, -np.inf], None)  # Substitui valores infinitos
        data = data.where(pd.notnull(data), None)     # Substitui NaN por None

        # Garantir que os dados são convertidos para um formato JSON compatível
        response = data.to_dict(orient="records")

        # Retornar a resposta como um dicionário
        return {"status": "success", "data": response}

    except ValueError as ve:
        # Erro relacionado a valores inválidos
        raise HTTPException(status_code=500, detail=f"Erro de valor: {str(ve)}")
    except Exception as e:
        # Qualquer outro erro
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")