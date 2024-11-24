from datetime import datetime, timedelta
import yfinance as yf

def get_stock_data(symbol: str, days_back: int = 120):
    """
    Obtém dados históricos de uma ação específica usando o Yahoo Finance.
    
    Args:
        symbol (str): Código da ação (e.g., 'BBAS3.SA').
        days_back (int): Quantidade de dias no passado para buscar dados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados financeiros.
    """
    # Datas de início e fim
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    # Download dos dados
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)  # Transforma o índice 'Date' em coluna
    df.columns = ['Date'] + [col[0] for col in df.columns[1:]]  # Ajusta MultiIndex
    df['Ticker'] = symbol  # Adiciona coluna com o ticker

    # Remove finais de semana
    df = df[df['Date'].dt.weekday < 5]
    return df


def ult_60_close(symbol: str = 'BBAS3.SA', days: int = 60):
    """
    Obtém os últimos 60 dias úteis de dados financeiros de uma ação específica.

    Args:
        symbol (str): Código da ação (e.g., 'BBAS3.SA').
        days (int): O número de dias úteis mais recentes para retornar (padrão: 60).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame completo com colunas selecionadas.
            - DataFrame preparado para o modelo (colunas ajustadas).
    """
    df = get_stock_data(symbol)

    # # Seleciona colunas úteis
    df = df[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']]

    # Filtra os últimos 60 registros úteis
    df = df.tail(60).reset_index(drop=True)

    # DataFrame preparado para o modelo
    df_model = df[['Open', 'High', 'Low', 'Volume', 'Close']]

    return df, df_model


