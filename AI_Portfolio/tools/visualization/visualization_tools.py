"""
Visualization tools for AI Portfolio Manager.
--------------------------------------------
This module contains functions to visualize:
- portfolio composition
- sector allocation
- portfolio value and performance
- comparisons vs benchmarks or other assets
"""

# ============================
# Imports
# ============================
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from tools.database.db_tools import (
    get_connection, 
    get_current_portfolio, 
    get_transactions_by_ticker, 
    get_transactions_by_date,
    get_sector_allocation
)
from tools.api.api_tools import (
    get_historical_prices,
    get_latest_close_prices,
    #portfolio_value_over_time,
    #portfolio_current_vs_purchase
)


# ============================
# Portfolio Composition
# ============================
def plot_portfolio_composition(conn):
    """
    Plot the portfolio composition showing the invested value per asset.
    Bars are colored by sector.
    """
    portfolio = get_current_portfolio(conn)

    if not portfolio:
        print("Portfolio is empty. No data to plot.")
        return

    # Unpack data from VIEW
    tickers = []
    values = []
    sectors = []

    for ticker, name, sector, total_quantity, avg_price, invested_value in portfolio:
        tickers.append(ticker)
        values.append(invested_value)
        sectors.append(sector)

    # Create a color map for sectors
    unique_sectors = sorted(set(sectors))
    cmap = plt.get_cmap("tab10")  # good default categorical colormap
    sector_colors = {
        sector: cmap(i % cmap.N) for i, sector in enumerate(unique_sectors)
    }

    bar_colors = [sector_colors[sector] for sector in sectors]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, values, color=bar_colors)
    plt.xlabel("Asset")
    plt.ylabel("Invested Value")
    plt.title("Portfolio Composition by Asset")
    plt.xticks(rotation=45)

    # Legend
    legend_elements = [
        Patch(facecolor=sector_colors[sector], label=sector)
        for sector in unique_sectors
    ]
    plt.legend(handles=legend_elements, title="Sector")

    plt.tight_layout()
    plt.show()


# ============================
# Portfolio Sector Allocation
# ============================
def plot_sector_allocation(portfolio_id: int):
    """
    Plot in a pie chart the sector allocation of a portfolio.
    """
    conn = get_connection()

    allocation = get_sector_allocation(conn)
    sectors = [x[0] for x in allocation]
    values = [x[1] for x in allocation]

    plt.figure(figsize=(8,6))
    plt.pie(values, labels=sectors, autopct='%1.1f%%', startangle=90)
    plt.title("Asset Allocation per Sector")
    plt.show()

    conn.close()
    return "Sector allocation plotted"


# ============================
# Portfolio Performance over time
# ============================
def plot_portfolio_value_over_time(conn, start_date=None, end_date=None, show_trades=True):
    """
    Plot the portfolio value over time using historical prices.

    - Portfolio value computed daily using historical prices
    - Optional markers for buy/sell days (aggregated per day)
    """

    # Default date range
    if start_date is None:
        start_date = '2025-01-02'
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Get transactions
    transactions_rows = get_transactions_by_date(conn, start_date, end_date)
    if not transactions_rows:
        print("No transactions in the specified date range.")
        return

    # Build DataFrame
    transactions_df = pd.DataFrame(
        transactions_rows,
        columns=['id', 'date', 'ticker', 'name', 'sector', 'quantity', 'price']
    )

    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    transactions_df = transactions_df[['date', 'ticker', 'quantity']]

    tickers = transactions_df['ticker'].unique().tolist()

    # Get historical prices
    price_df = get_historical_prices(tickers, start_date, end_date)

    # Compute portfolio value over time
    portfolio_series = portfolio_value_over_time(transactions_df, price_df)

    # ---- Plot portfolio value ----
    plt.figure(figsize=(11, 6))
    plt.plot(
        portfolio_series.index,
        portfolio_series.values,
        label='Portfolio Value',
        color = 'tab:blue'
    )

    # ---- Plot buy/sell markers ----
    if show_trades:
        # Aggregate transactions per day
        daily_trades = (
            transactions_df
            .groupby('date')['quantity']
            .sum()
            .reset_index()
        )

        buy_days = daily_trades[daily_trades['quantity'] > 0]['date']
        sell_days = daily_trades[daily_trades['quantity'] < 0]['date']
        mixed_days = daily_trades[daily_trades['quantity'] == 0]['date']

        # Align with portfolio series
        buy_values = portfolio_series.loc[portfolio_series.index.isin(buy_days)]
        sell_values = portfolio_series.loc[portfolio_series.index.isin(sell_days)]
        mixed_values = portfolio_series.loc[portfolio_series.index.isin(mixed_days)]

        plt.scatter(
            buy_values.index,
            buy_values.values,
            marker='^',
            color='tab:green',
            label='Net Buy',
            zorder=3
        )

        plt.scatter(
            sell_values.index,
            sell_values.values,
            marker='v',
            color='tab:red',
            label='Net Sell',
            zorder=3
        )

        plt.scatter(
            mixed_values.index,
            mixed_values.values,
            marker='o',
            color='tab:orange',
            s=80,          
            edgecolors='k',
            label='Buy & Sell',
            zorder=3
        )

    # ---- Styling ----
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Portfolio's Assets Comparison
# ============================
def plot_portfolio_performance(conn):
    """
    Compare invested value vs current value per asset
    using latest available market prices.
    """

    portfolio = get_current_portfolio(conn)

    if not portfolio:
        print("Portfolio is empty.")
        return

    tickers = []
    quantities = []
    invested_values = []

    for ticker, name, sector, total_quantity, avg_price, invested_value in portfolio:
        tickers.append(ticker)
        quantities.append(total_quantity)
        invested_values.append(invested_value)

    # Get latest prices
    latest_prices = get_latest_prices(tickers)

    current_values = []
    pnl_pct = []

    for ticker, qty, invested in zip(tickers, quantities, invested_values):
        price = latest_prices.get(ticker)

        if price is None:
            current_values.append(invested)
            pnl_pct.append(0)
            continue

        current_value = qty * price
        current_values.append(current_value)
        pnl_pct.append((current_value - invested) / invested * 100)

    # Plot
    x = np.arange(len(tickers))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, invested_values, width, label="Invested Value")
    plt.bar(x + width/2, current_values, width, label="Current Value")

    for i, pct in enumerate(pnl_pct):
        plt.text(
            x[i],
            max(invested_values[i], current_values[i]) * 1.02,
            f"{pct:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.xticks(x, tickers, rotation=45)
    plt.ylabel("Value")
    plt.title("Portfolio Performance by Asset")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Sector Comparison
# ============================
def plot_sector_performance(conn):
    """
    Compare invested vs current value aggregated by sector.
    """

    portfolio = get_current_portfolio(conn)

    if not portfolio:
        print("Portfolio is empty.")
        return

    # Prepare asset-level data
    rows = []
    tickers = []

    for ticker, name, sector, qty, avg_price, invested in portfolio:
        rows.append({
            "ticker": ticker,
            "sector": sector,
            "quantity": qty,
            "invested": invested
        })
        tickers.append(ticker)

    df = pd.DataFrame(rows)

    # Latest prices
    latest_prices = get_latest_prices(tickers)
    df["latest_price"] = df["ticker"].map(latest_prices)

    # Current value
    df["current_value"] = df["quantity"] * df["latest_price"]

    # Aggregate by sector
    sector_df = (
        df.groupby("sector")[["invested", "current_value"]]
        .sum()
        .reset_index()
    )

    sector_df["pnl_pct"] = (
        (sector_df["current_value"] - sector_df["invested"])
        / sector_df["invested"] * 100
    )

    # Plot
    x = np.arange(len(sector_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, sector_df["invested"], width, label="Invested Value")
    plt.bar(x + width/2, sector_df["current_value"], width, label="Current Value")

    for i, pct in enumerate(sector_df["pnl_pct"]):
        plt.text(
            x[i],
            max(sector_df.loc[i, "invested"], sector_df.loc[i, "current_value"]) * 1.02,
            f"{pct:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.xticks(x, sector_df["sector"], rotation=45)
    plt.ylabel("Value")
    plt.title("Portfolio Performance by Sector")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Portfolio vs Benchmark
# ============================
def plot_portfolio_vs_benchmark(conn, benchmark_ticker='^GSPC', start_date=None, end_date=None):
    """
    Plot portfolio value over time compared to a benchmark (default S&P500),
    normalized to 100 at the date of the last portfolio transaction.

    Parameters:
        conn: database connection
        benchmark_ticker: ticker symbol for benchmark (default S&P500 ^GSPC)
        start_date: optional, start date for transactions
        end_date: optional, end date (default today)
    """
    # 1. Transactions
    transactions_rows = get_transactions_by_date(conn, start_date or '2025-01-02', end_date or pd.Timestamp.today().strftime('%Y-%m-%d'))
    if not transactions_rows:
        print("No transactions available in the specified range.")
        return

    transactions_df = pd.DataFrame(
        transactions_rows, columns=['id','date','ticker','name','sector','quantity','price']
    )
    transactions_df = transactions_df[['date','ticker','quantity']]

    tickers = transactions_df['ticker'].unique().tolist()

    # 2. Last transaction date -> start for normalization
    last_txn_date = transactions_df['date'].max()

    # 3. Portfolio series
    price_df = get_historical_prices(tickers, last_txn_date, end_date or pd.Timestamp.today().strftime('%Y-%m-%d'))
    portfolio_series = portfolio_value_over_time(transactions_df, price_df)

    # 4. Extend to latest prices if today > last date in series
    latest_prices = get_latest_prices(tickers)
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    latest_value = sum(
        transactions_df.groupby('ticker')['quantity'].sum().get(t,0) * latest_prices.get(t,0)
        for t in tickers
    )
    portfolio_series.loc[today] = latest_value

    # 5. Normalize portfolio to 100 at last transaction date
    normalization_factor = 100 / portfolio_series.loc[last_txn_date]
    portfolio_series = portfolio_series * normalization_factor

    # 6. Plot portfolio
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_series.index, portfolio_series.values, label="Portfolio")

    # 7. Plot transactions as triangles
    buy_trades = transactions_df[transactions_df['quantity']>0]
    sell_trades = transactions_df[transactions_df['quantity']<0]
    for df, color, marker in [(buy_trades,'green','^'), (sell_trades,'red','v')]:
        for _, row in df.iterrows():
            date = row['date']
            if date in portfolio_series.index:
                plt.scatter(date, portfolio_series.loc[date], color=color, marker=marker, s=100, zorder=5)

    # 8. Benchmark
    try:
        benchmark_df = get_historical_prices([benchmark_ticker], last_txn_date, today)
        benchmark_series = benchmark_df[benchmark_ticker]
        # Normalize benchmark to 100 at last transaction date
        benchmark_series = benchmark_series / benchmark_series.iloc[0] * 100
        plt.plot(benchmark_series.index, benchmark_series.values, label=f"Benchmark ({benchmark_ticker})", linestyle='--')
    except Exception as e:
        print(f"Warning: cannot fetch benchmark {benchmark_ticker}: {e}")

    # 9. Final touches
    plt.title("Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Comparison between Assets
# ============================
def plot_normalized_comparison(tickers, start_date=None, end_date=None):
    """
    Plot normalized price comparison for two or more tickers, starting from 100.
    Uses api_tools.get_historical_prices to fetch data.

    Parameters:
        tickers (list of str): tickers to compare
        start_date (str, optional): "YYYY-MM-DD", default earliest available
        end_date (str, optional): "YYYY-MM-DD", default today
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from tools.api.api_tools import get_historical_prices  # usa la funzione esistente

    if not tickers:
        print("No tickers provided.")
        return

    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Get historical prices via api_tools
    price_df = get_historical_prices(tickers, start_date, end_date)

    # Forward-fill missing values
    price_df.ffill(inplace=True)

    # Normalize to 100 at the first available date
    normalized = price_df / price_df.iloc[0] * 100

    # Plot
    plt.figure(figsize=(12,6))
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)

    plt.title("Normalized Price Comparison")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Start=100)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Stock Price over time 
# ============================
def plot_stock_price(ticker, start_date='2025-01-02', end_date=None):
    """
    Plot absolute stock price over time.

    Parameters:
        ticker: str, stock ticker
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD, defaults today
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Get historical prices
    df = get_historical_prices([ticker], start_date, end_date)
    if df.empty or ticker not in df:
        print(f"No data found for {ticker}")
        return

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df[ticker])
    plt.title(f"{ticker} Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
