# **Predição de Valores do BBAS3 utlizando Redes Neurais**

Este projeto tem como objetivo desenvolver uma API para realizar previsões de valores da ação BBAS3 (Banco do Brasil) usando um modelo de Redes Neurais Recorrentes (RNN) com camadas LSTM. A API oferece duas rotas principais:

1. Previsão de Valores a Partir de Dados Históricos: Nesta rota, o usuário pode fornecer dados históricos em formato CSV para obter previsões dos valores da ação BBAS3 em diferentes horizontes temporais, como 1, 3, 7 e 15 dias.

2. Previsão Diária com Armazenamento e Análise de Performance: Nesta rota, o modelo gera previsões diárias e armazena os valores preditos e reais em um banco de dados, permitindo o acompanhamento e análise de performance do modelo ao longo do tempo.

O projeto permite a visualização dos resultados em tempo real através do Grafana e integra o monitoramento de métricas de performance utilizando Prometheus. Com isso, o sistema oferece uma plataforma completa para prever, armazenar e monitorar as previsões dos valores de ações, facilitando a análise contínua do desempenho da rede neural e a detecção de possíveis ajustes necessários.

---

## **Estrutura do Projeto**
```markdown
RN_Predicao_Acao/
├── app/
│   ├── main.py # Aplicação Principal da API
│   │
│   ├── routes/
│   │   └── predictions.py # Rotas da API de predições
│   └── utils/
│       ├── data_utils.py # Processar e Salvar prdições no Banco de Dados
│       ├── ml_utils.py # Processar o Modelo
│       ├── y_finance_utils.py # Carregar os dados da API do Yahoo Finance
│       │
│       ├── constants/
│       │   └── settings.py # Constantes
│       └── prometheus/
│           ├── metrics_manager.py # Atualização e Guardar métricas
│           ├── monitor.py # Base do Prometheus com as Métricas
│           └── resources.py # Monitoramento
│           
│
├── data/
│   ├── data.py # Realiza as operações no Banco de Dados
│   └── metrics.db # Banco de Dados
│
├── LSTM/ # Modelo De Redes Neurais
│   ├── models/ # Contém Históricos de Treinamento com diversos parâmetros
│   ├── modelo_llm.h5 # Modelo utilizado para predição na API
│   ├── scaler.joblib # Scaler utilizado para normalização dos dados
│   └── modelo.ipynb # Conjunto de testes e treinamentos de modelos e processamento dos dados
│
└── readme.md
```

## **Modelo de Redes Neurais**

Foram realizados diversos testes com diferentes parâmetros, conforme documentado no arquivo ``LSTM/modelo.ipynb``, utilizando os dados da ação ``BBAS3.SA`` de ``2000-01-01`` a ``2024-10-01``. Durante a modelagem, selecionamos as features ``Open``, ``High``, ``Low``, ``Volume`` e ``Close`` para o treinamento e previsão. Essas variáveis são essenciais para que o modelo capte padrões de mercado e capture a volatilidade e a demanda nas negociações, melhorando a precisão da previsão de fechamento.

Se utilizássemos apenas o valor de fechamento como feature, o modelo teria uma visão limitada do mercado, perdendo informações sobre flutuações diárias e volume de negociações que são cruciais para a previsão. Assim, o uso de todas as features é fundamental para um modelo mais robusto.

As avaliações foram conduzidas no notebook com as métricas MSE, RMSE, MAE e MAPE para diferentes horizontes de previsão.

O melhor modelo foi implementado utilizando o framework TensorFlow/Keras com as seguintes configurações:

### **Arquitetura**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# Camadas LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Camada de saída com múltiplas previsões
model.add(Dense(units=len(forecast_horizons)))  # Saída para 1, 3, 7 e 15 dias

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

```

### **Treinamento**
O modelo foi treinado com os seguintes parâmetros:

- Épocas: 20
- Tamanho do Lote (Batch Size): 32
- Otimizador: Adam
- Função de Perda: Mean Squared Error (MSE)

Código utilizado para treinamento:
```python
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

### **Métricas de Avaliação**
Os resultados do modelo para diferentes horizontes de previsão foram os seguintes:

| Horizonte       | MSE    | RMSE   | MAE    | MAPE   |
|-----------------|--------|--------|--------|--------|
| 1 Dia           | 0.000816 | 0.028569 | 0.020808 | 3.38%  |
| 3 Dias          | 0.001249 | 0.035345 | 0.025892 | 4.19%  |
| 7 Dias          | 0.002663 | 0.051608	| 0.039086 | 5.92%  |
| 15 Dias         | 0.004242 | 0.065130 | 0.048969 | 7.73%  | 



### **Monitoramento de Performance**
Foi utilizado o **Prometheus** para monitorar métricas de performance e CPU do modelo. Essas informações são exibidas em dashboards interativos no **Grafana**. O monitoramento permite verificar a eficiência do modelo e compará-lo com os dados reais.

## **Banco de Dados**
O projeto utiliza um banco de dados SQLite para armazenar informações de métricas de uso do sistema e desempenho de previsões de preços das ações. O banco de dados é composto por duas tabelas principais:

### **Estrutura das Tabelas**
1. Tabela ``metrics_archive``

    Essa tabela armazena informações sobre o uso do sistema, como a utilização de CPU e memória, além do tempo de processamento de cada requisição e os valores previstos para os próximos dias. Ela é utilizada para monitoramento de métricas de desempenho.

    -  Colunas:
        - ``id``: Identificador único da métrica (INTEGER, PRIMARY KEY).
        - ``timestamp``: Data e hora da coleta das métricas (TEXT).
        - ``request_count``: Número de requisições processadas (INTEGER).
        - ``cpu_usage``: Percentual de uso da CPU no momento da coleta (REAL).
        - `memory_usage`: Percentual de uso da memória no momento da coleta (REAL).
        - ``request_processing_seconds``: Tempo de processamento da requisição em segundos (REAL).
        - ``forecast_horizon_1_day``: Previsão para 1 dia à frente (REAL).
        - ``forecast_horizon_3_days``: Previsão para 3 dias à frente (REAL).
        - ``forecast_horizon_7_days``: Previsão para 7 dias à frente (REAL).
        - ``forecast_horizon_15_days``: Previsão para 15 dias à frente (REAL).

2. Tabela ``model_performance``

    Esta tabela armazena as previsões realizadas pelo modelo para o fechamento das ações BBAS3 em diferentes horizontes temporais, bem como os valores reais observados para cada um desses horizontes. Ela é usada para análise e acompanhamento da performance do modelo.

    - Colunas:
        - ``id``: Identificador único da previsão (INTEGER, PRIMARY KEY).
        - ``ticker``: Símbolo da ação para a qual a previsão foi feita (TEXT).
        - ``prediction_date``: Data da previsão realizada (TEXT).
        - ``target_date_1d``, ``target_date_3d``, ``target_date_7d``, ``target_date_15d``: Datas alvo para os horizontes de 1, 3, 7 e 15 dias, respectivamente (TEXT).
        - ``predicted_close_1d``, ``predicted_close_3d``, ``predicted_close_7d``, ``predicted_close_15d``: Valores previstos de fechamento para os horizontes de 1, 3, 7 e 15 dias, respectivamente (REAL).
        - ``real_close_1d``, ``real_close_3d``, ``real_close_7d``, ``real_close_15d``: Valores reais de fechamento observados para os horizontes de 1, 3, 7 e 15 dias, respectivamente (REAL).

## **Rotas da API**
### 1. **Predição com Preços Históricos**
Essa rota permite enviar um arquivo CSV com dados históricos para realizar predições.

- Requisitos:
    - O CSV deve conter pelo menos 60 registros.
    - As colunas devem seguir o formato esperado pelo modelo.
- Endpoint: /predict/historical
- Método HTTP: POST
- Entrada:
    - Arquivo CSV com os dados históricos com as colunas: 'Open', 'High', 'Low', 'Volume', 'Close'.
- Saída:
    ```json
    {
        "forecast_horizon_1d": valor,
        "forecast_horizon_3d": valor,
        "forecast_horizon_7d": valor,
        "forecast_horizon_15d": valor
    }
    ```

### **2. Predição para o Dia Atual**
Essa rota utiliza os dados atuais armazenados no banco para realizar a predição e registrar os resultados.

- Endpoint: /predict/today
- Método HTTP: GET
- Entrada: Não requer parâmetros.
- Saída
    ```json
    {
        "forecast_horizon_1d": valor,
        "forecast_horizon_3d": valor,
        "forecast_horizon_7d": valor,
        "forecast_horizon_15d": valor
    }
    ```

Além disso, a API atualiza continuamente o banco de dados com os valores reais, permitindo comparações diretas entre os valores previstos e os reais no Grafana.


## **Visualização no Grafana**
Os valores previstos e reais são registrados no banco de dados e visualizados em gráficos no Grafana, que incluem:

- Comparação entre valores reais e previstos.
- Métricas de desempenho do modelo:
    - MAPE:
        ```sql 
        SELECT
        AVG(ABS((real_close_1d - predicted_close_1d) / real_close_1d) * 100) AS "MAPE Prev 1Dia",
        AVG(ABS((real_close_3d - predicted_close_3d) / real_close_3d) * 100) AS "MAPE Prev 3Dias",
        AVG(ABS((real_close_7d - predicted_close_7d) / real_close_7d) * 100) AS "MAPE Prev 7Dias",
        AVG(ABS((real_close_15d - predicted_close_15d) / real_close_15d) * 100) AS "MAPE Prev 15Dias" 
        FROM 
        model_performance
        ``` 
    - MAE
        ```sql
        SELECT AVG(ABS(real_close_1d - predicted_close_1d)) AS "MAE 1Dia", 
        AVG(ABS(real_close_3d - predicted_close_3d)) AS "MAE 3Dias", 
        AVG(ABS(real_close_7d - predicted_close_7d)) AS "MAE 7Dias", 
        AVG(ABS(real_close_15d - predicted_close_15d)) AS "MAE 15Dias"
        FROM 
        model_performance
        ```
    - MSE 
        ```sql
        SELECT 
        AVG(POWER(real_close_1d - predicted_close_1d, 2)) AS "MSE 1Dia", 
        AVG(POWER(real_close_3d - predicted_close_3d, 2)) AS "MSE 3Dias", 
        AVG(POWER(real_close_7d - predicted_close_7d, 2)) AS "MSE 7Dias",
        AVG(POWER(real_close_15d - predicted_close_15d, 2)) AS "MSE 15Dias"
        FROM 
        model_performance
        ```
    - RMSE
        ```sql
        SELECT 
        SQRT(AVG(POWER(real_close_1d - predicted_close_1d, 2))) AS "RMSE 1Dia", 
        SQRT(AVG(POWER(real_close_3d - predicted_close_3d, 2))) AS "RMSE 3Dias", 
        SQRT(AVG(POWER(real_close_7d - predicted_close_7d, 2))) AS "RMSE 7Dias", 
        SQRT(AVG(POWER(real_close_15d - predicted_close_15d, 2))) AS "RMSE 15Dias"
        FROM 
        model_performance
        ```
    
- Utilização de CPU e memória durante a execução do modelo.

## **Requisitos do Sistema**
- Python 3.8 ou superior.
- Bibliotecas:
    - tensorflow
    - keras
    - pandas
    - numpy
    - fastapi
    - uvicorn
    - Integração com Prometheus e Grafana.

## **Como Executar**
### 1. **Instale as dependências:**

```bash
pip install -r requirements.txt
```
### 2. **Execute a aplicação:**

```bash
uvicorn app.main:app --reload
```

### 3. **Acesse a API no navegador em:**

```bash
http://127.0.0.1:8000
```

## **Contato**
Em caso de dúvidas ou sugestões, entre em contato com o desenvolvedor:

- Nome: Pablo de Oliveira Picinini
- E-mail: pablopicinini@gmail.com