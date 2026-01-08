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
