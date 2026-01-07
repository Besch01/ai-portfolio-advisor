"""
API Tools for Portfolio Visualization
-------------------------------------
Functions to retrieve historical and current stock prices for tickers
in the portfolio, to use for plotting value over time, performance comparison,
and advanced visualizations.
"""

# ============================
# Imports
# ============================
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# -------------------------------
# Get historical daily prices for a list of tickers
# -------------------------------
def get_historical_prices(tickers, start_date, end_date):
    """
    Download historical daily close prices for the given tickers.

    Parameters:
        tickers (list of str): list of ticker symbols
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'

    Returns:
        pandas DataFrame:
            index = dates
            columns = tickers
            values = closing prices
    """
    if not tickers:
        return pd.DataFrame()
    
    # yfinance supports multiple tickers at once
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']
    
    # If only one ticker, ensure DataFrame with proper column name
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # Sort + Fill missing prices (forward-fill)
    data = data.sort_index()
    data.ffill(inplace=True)
    
    return data


# -------------------------------
# Get the latest price for a list of tickers
# -------------------------------
def get_latest_prices(tickers):
    """
    Get the latest available price (usually yesterday close) for each ticker.

    Parameters:
        tickers (list of str): list of ticker symbols

    Returns:
        dict: {ticker: latest_price}
    """
    latest_prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')  # last 5 days to ensure at least yesterday
        if not hist.empty:
            latest_prices[ticker] = hist['Close'][-1]
        else:
            latest_prices[ticker] = None
    return latest_prices


# -------------------------------
# Generate portfolio value over time
# -------------------------------
def portfolio_value_over_time(transactions_df, price_df):
    """
    Calculate daily portfolio value given transactions and price data.

    Parameters:
        transactions_df (DataFrame):
            columns = ['date','ticker','quantity']
        price_df (DataFrame):
            index = dates
            columns = tickers
            values = closing prices

    Returns:
        pandas Series: index=dates, values=total portfolio value per day
    """
    # Ensure date column is datetime
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    # Create empty DataFrame for cumulative quantities per ticker per day
    all_dates = price_df.index
    tickers = price_df.columns
    qty_df = pd.DataFrame(0, index=all_dates, columns=tickers)
    
    # Aggregate cumulative quantities
    for ticker in tickers:
        # Filter transactions for ticker
        df_t = transactions_df[transactions_df['ticker'] == ticker]
        # Sum cumulatively by date
        df_t = df_t.groupby('date')['quantity'].sum().cumsum()
        # Reindex to all_dates, forward-fill quantities
        df_t = df_t.reindex(all_dates, method='ffill').fillna(0)
        qty_df[ticker] = df_t
    
    # Calculate daily portfolio value
    portfolio_values = (qty_df * price_df).sum(axis=1)
    return portfolio_values


# -------------------------------
# Generate portfolio current vs purchase value
# -------------------------------
def portfolio_current_vs_purchase(transactions_df, latest_prices):
    """
    Calculate for each ticker the value at average purchase price and
    the value at latest price.

    Parameters:
        transactions_df (DataFrame):
            columns = ['ticker','quantity','price']
        latest_prices (dict): {ticker: latest_price}

    Returns:
        pandas DataFrame:
            index = tickers
            columns = ['purchase_value', 'current_value']
    """
    df = transactions_df.copy()
    df['total'] = df['quantity'] * df['price']
    
    purchase_value = df.groupby('ticker')['total'].sum()
    quantity_total = df.groupby('ticker')['quantity'].sum()
    
    current_value = quantity_total * pd.Series(latest_prices)
    
    result = pd.DataFrame({
        'purchase_value': purchase_value,
        'current_value': current_value
    }).fillna(0)
    
    return result

"""
API Tools for Portfolio Database

"""

import yfinance as yf

from database_tools import get_connection, insert_transaction


def insert_market_transaction(ticker, quantity, name=None, sector=None):
    """
    Inserts a new transaction in the database at the current market price.

    Parameters:
    - ticker (str)
    - quantity (float): positive for buy, negative for sell
    - name (str, optional): company name
    - sector (str, optional): company sector

    Returns:
    - dict with inserted transaction details
    """

    # -----------------------------
    # 1. Get current market price
    # -----------------------------
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")

    if hist.empty:
        raise ValueError(f"Market price not available for ticker {ticker}")

    market_price = round(hist["Close"].iloc[-1], 2)

    # -----------------------------
    # 2. Get metadata if missing
    # -----------------------------
    info = stock.info

    if name is None:
        name = info.get("shortName", ticker)

    if sector is None:
        sector = info.get("sector", "Unknown")

    # -----------------------------
    # 3. Insert transaction
    # -----------------------------
    transaction_date = date.today().isoformat()

    conn = get_connection()
    insert_transaction(
        conn=conn,
        date=transaction_date,
        ticker=ticker,
        name=name,
        sector=sector,
        quantity=quantity,
        price=market_price
    )
    conn.close()

    # -----------------------------
    # 4. Return summary
    # -----------------------------
    return {
        "ticker": ticker,
        "quantity": quantity,
        "price": market_price,
        "date": transaction_date,
        "name": name,
        "sector": sector
    }



def get_best_returns_api(conn):
    """
    Calculates the real return for each title in the portfolio using the current market prices, via yahoofinance.

    It returns a list of:
    (ticker, name, sector, total_quantity, avg_price, market_price, return_pct)
    ordered for return_pct descendant.
    """
    cur = conn.cursor()

    # take the actual state of the portfolio from the view
    cur.execute("SELECT ticker, name, sector, total_quantity, avg_price FROM current_portfolio")
    portfolio = cur.fetchall()

    results = []
    for ticker, name, sector, total_quantity, avg_price in portfolio:
        try:
            # create the ticker for the title
            stock = yf.Ticker(ticker)
            
            # obtain the average current market price
            info = stock.info
            market_price = info.get("regularMarketPrice")

            # if there isn't regularMarketPrice, try to obtain a more recent close
            if market_price is None:
                hist = stock.history(period="1d")
                if not hist.empty:
                    market_price = hist["Close"].iloc[-1]
                else:
                    market_price = avg_price  # fallback if no data available

            # calculates the percentage return
            return_pct = round((market_price - avg_price) / avg_price * 100, 2)

            results.append(
                (ticker, name, sector, total_quantity, avg_price, market_price, return_pct)
            )

        except Exception as e:
            # API/connection errors don't stop the entire calculation
            print(f"Errore nell’ottenere il prezzo per {ticker}: {e}")
            continue

    # order for descendant returns
    results.sort(key=lambda x: x[-1], reverse=True)
    return results
