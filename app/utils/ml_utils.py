import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from typing import Dict
from app.utils.constants.settings import SEQ_LENGTH
from fastapi import HTTPException

def load_model_and_scaler():
    """
    Função para carregar o modelo treinado e o scaler.

    Essa função carrega o modelo LSTM salvo no formato `.h5` e o scaler 
    (usado para normalizar os dados) salvo no formato `.joblib`.

    Returns:
        model: O modelo LSTM carregado.
        scaler: O scaler carregado para normalização dos dados.
    """
    model = tf.keras.models.load_model("LSTM/modelo_teste4.h5")
    scaler = joblib.load("LSTM/scaler.joblib")
    return model, scaler

def create_sequence_and_scaler(input_data: Dict, scaler):
    """
    Função para processar os dados de entrada em uma sequência para alimentar o modelo.

    A função transforma os dados de entrada em uma sequência com o tamanho definido por `SEQ_LENGTH`,
    e aplica a transformação do scaler nos dados.

    Args:
        input_data: Dicionário contendo as colunas 'Open', 'High', 'Low', 'Volume', 'Close'.
        scaler: Scaler que será usado para normalizar os dados.

    Returns:
        input_scaled: Dados de entrada normalizados e redimensionados para o formato adequado para o modelo.
    """
    sequence = []
    for i in range(SEQ_LENGTH):
        sequence.append([
            input_data['Open'][i],
            input_data['High'][i],
            input_data['Low'][i],
            input_data['Volume'][i],
            input_data['Close'][i]
        ])
    input_sequence = np.array(sequence)
    input_scaled = scaler.transform(input_sequence)

    # Redimensionando para o formato (1, SEQ_LENGTH, número de features)
    input_scaled = np.expand_dims(input_scaled, axis=0)

    return input_scaled

    #return scaler.transform(np.expand_dims(sequence, axis=0))

def invert_normalized(predicted_scaled, scaler, lenght):
    """
    Função para inverter a normalização das previsões do modelo.

    Após a previsão, a normalização aplicada precisa ser revertida para que os valores previstos 
    sejam trazidos de volta à escala original. Essa função faz isso para cada previsão de fechamento.

    Args:
        predicted_scaled: Previsões normalizadas realizadas pelo modelo.
        scaler: O scaler que foi usado para normalizar os dados.
        lenght: O comprimento da sequência de entrada original, usado para ajustar a inversão.

    Returns:
        predictions: Lista com as previsões desnormalizadas (valores de fechamento previstos).
    """
    predictions = []
    for i in range(predicted_scaled.shape[1]):
        prediction_reshaped = np.zeros((1, lenght))  # ajustando o shape para o scaler
        prediction_reshaped[0, -1] = predicted_scaled[0, i]  # Colocando o valor de 'Close' previsto na última posição
        prediction_inversed = scaler.inverse_transform(prediction_reshaped)[0, -1]
        predictions.append(prediction_inversed)
    
    return predictions

def process_csv(file):
    """
    Função para carregar e processar um arquivo CSV contendo dados financeiros.

    Essa função carrega os dados de um arquivo CSV, valida se ele contém as colunas esperadas e,
    em seguida, seleciona as últimas 60 linhas de dados necessários para alimentar o modelo.

    Args:
        file: Arquivo CSV contendo os dados financeiros da ação.

    Returns:
        dict: Um dicionário com as últimas 60 linhas de dados das colunas 'Open', 'High', 'Low', 'Volume', 'Close'.

    Raises:
        HTTPException: Se o CSV não contiver as colunas esperadas ou se houver menos de 60 registros.
    """
    try:
        df = pd.read_csv(file)
        # Removendo espaços em branco dos nomes das colunas
        df.columns = df.columns.str.strip()

        # Verifica se o CSV contém as colunas esperadas
        required_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError("O CSV deve conter as colunas: Open, High, Low, Volume, Close.")
        
        # Verifica se há pelo menos 60 registros
        if len(df) < SEQ_LENGTH:
            raise ValueError("O CSV deve conter pelo menos 60 registros.")
        
        # Pega apenas as últimas 60 linhas e converte para o formato necessário
        df = df[required_columns].tail(SEQ_LENGTH)
        return df.to_dict(orient='list')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def process_prediction(data, model, scaler, lenght):
    """
    Função para processar os dados de entrada, realizar a previsão e inverter a normalização.

    Essa função chama a função `create_sequence_and_scaler` para preparar os dados, 
    realiza a previsão utilizando o modelo LSTM e, por fim, inverte a normalização 
    das previsões para obter os valores de fechamento reais previstos.

    Args:
        data: Dados de entrada para a previsão.
        model: Modelo treinado usado para fazer a previsão.
        scaler: Scaler usado para normalizar os dados.
        lenght: Comprimento da sequência de dados de entrada.

    Returns:
        predictions: Lista com as previsões desnormalizadas (valores de fechamento previstos).
    """
    input_scaled = create_sequence_and_scaler(data, scaler)
    predicted_scaled = model.predict(input_scaled)
    predictions = invert_normalized(predicted_scaled, scaler, lenght)
    return predictions
