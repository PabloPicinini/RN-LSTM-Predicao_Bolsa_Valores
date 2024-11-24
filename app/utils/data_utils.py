from data.data import save_prediction_to_db, update_real_close

def save_predictions_to_db(ticker, predictions, prediction_date):
    """
    Função para salvar as previsões no banco de dados.

    Essa função recebe o ticker da ação, as previsões realizadas para o período de 1, 3, 7 e 15 dias,
    e a data da previsão. Ela então chama a função `save_prediction_to_db` para armazenar essas informações
    no banco de dados.

    Args:
        ticker: O símbolo da ação para a qual as previsões foram feitas.
        predictions: Lista com as previsões para os horizontes de 1, 3, 7 e 15 dias.
        prediction_date: Data em que as previsões foram feitas.

    Returns:
        saved: Resultado da operação de salvamento, `True` ou `False`.
    """
    saved = save_prediction_to_db(ticker, predictions, prediction_date)
    return saved

def update_real_data(df, ticker):
    """
    Função para atualizar os valores reais de fechamento no banco de dados.

    Essa função pega o valor de fechamento real mais recente dos dados fornecidos (`df`), 
    e atualiza o banco de dados com esse valor. A função `update_real_close` é chamada para 
    realizar a atualização no banco.

    Args:
        df: DataFrame contendo os dados históricos de preços da ação, incluindo a coluna 'Date'.
        ticker: O símbolo da ação para a qual os dados reais de fechamento serão atualizados.

    Returns:
        None
    """
    latest_real_date = df['Date'].max()
    latest_real_date_str = latest_real_date.strftime('%Y-%m-%d')
    latest_real_close_value = df[df['Date'] == latest_real_date].iloc[0]['Close']
    update_real_close(ticker, latest_real_date_str, latest_real_close_value)
