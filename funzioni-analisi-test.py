# ============================
# Portfolio Analysis Helpers
# ============================
import pandas as pd
from tools.database.db_tools import get_current_portfolio
from tools.api.api_tools import get_historical_prices, get_latest_close_prices

# -------------------------------
# Helper
# -------------------------------
def unwrap_db_response(resp):
    """Restituisce i dati estratti da un DB response standard."""
    return resp["data"] if resp and resp.get("status") == "ok" else []

# -------------------------------
# Portfolio Value Over Time
# -------------------------------
def portfolio_value_over_time(start_date, end_date):
    """
    Calcola il valore giornaliero del portafoglio a partire dallo stato attuale.
    Utilizza i prezzi storici per i ticker presenti nel portfolio.
    
    Parameters:
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
    
    Returns:
        pandas Series: index = date, values = portfolio value per giorno
    """
    portfolio = unwrap_db_response(get_current_portfolio())
    if not portfolio:
        return pd.Series(dtype=float)
    
    # DataFrame sintetico con ticker e quantità
    df = pd.DataFrame(portfolio, columns=['ticker','name','sector','total_quantity','avg_price','invested_value'])
    df = df[['ticker','total_quantity']].rename(columns={'total_quantity':'quantity'})
    
    tickers = df['ticker'].unique().tolist()
    if not tickers:
        return pd.Series(dtype=float)
    
    # Prendi prezzi storici
    price_df = get_historical_prices(tickers, start_date, end_date)
    if price_df.empty:
        return pd.Series(dtype=float)
    
    # Quantità giornaliera costante (stato corrente)
    qty_df = pd.DataFrame(index=price_df.index, columns=tickers)
    for ticker in tickers:
        qty_df[ticker] = df.loc[df['ticker']==ticker, 'quantity'].values[0]
    
    # Valore giornaliero portfolio
    portfolio_values = (qty_df * price_df).sum(axis=1)
    return portfolio_values

# -------------------------------
# Portfolio Current vs Purchase Value
# -------------------------------
def portfolio_current_vs_purchase():
    """
    Calcola per ciascun ticker il valore a prezzo medio di acquisto
    e il valore corrente basato sugli ultimi prezzi di chiusura.
    
    Returns:
        pandas DataFrame: index = tickers
                          columns = ['purchase_value', 'current_value']
    """
    portfolio = unwrap_db_response(get_current_portfolio())
    if not portfolio:
        return pd.DataFrame(columns=['purchase_value','current_value'])
    
    df = pd.DataFrame(portfolio, columns=['ticker','name','sector','total_quantity','avg_price','invested_value'])
    df = df[['ticker','total_quantity','avg_price']].rename(columns={'total_quantity':'quantity'})
    
    tickers = df['ticker'].tolist()
    latest_prices = get_latest_close_prices(tickers)
    
    df['purchase_value'] = df['quantity'] * df['avg_price']
    df['current_value'] = df['quantity'] * df['ticker'].map(latest_prices)
    
    result = df.set_index('ticker')[['purchase_value','current_value']]
    return result


# ============================
# Quick Test
# ============================
if __name__ == "__main__":
    print("=== Testing portfolio_value_over_time ===")
    start_date = "2026-01-01"
    end_date = "2026-01-05"
    pv = portfolio_value_over_time(start_date, end_date)
    if pv.empty:
        print("No portfolio data available for plotting.")
    else:
        print(pv)

    print("\n=== Testing portfolio_current_vs_purchase ===")
    pcv = portfolio_current_vs_purchase()
    if pcv.empty:
        print("No portfolio data available for current vs purchase.")
    else:
        print(pcv)
