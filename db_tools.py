"""
Database tools for portfolio manager
"""

import sqlite3
import os




def get_connection():
    """
    It returns a connection to the database 'portfolio_manager.db'.
    The connession is centralized so that all the functions use the same logic to open the db.
    If we need to change the path, we modify only this function.
    (In the use, The connection must be closed by 'conn.close()'.)
    """
    # find the path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "portfolio_manager.db")
    
    # create the connection SQLite
    conn = sqlite3.connect(db_path)
    
    return conn

conn = get_connection()
print("I am using the database:", os.path.abspath(conn.execute("PRAGMA database_list").fetchall()[0][2]))

def insert_transaction(conn, date, ticker, name, sector, quantity, price):
    """
    It insert a new transaction in the database.

    Parameters:
    - conn: refers to the connection to the db;
    - date: transaction's date, "YYYY-MM-DD"
    - ticker: title's symbol
    - name: title's name
    - sector: title's sector
    - quantity: if buy, positive; if sell, negative;
    - price: unitary price
    """
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO transactions (date, ticker, name, sector, quantity, price) VALUES (?, ?, ?, ?, ?, ?)",
        (date, ticker, name, sector, quantity, price)
    )
    
    conn.commit()  # saves the modifies


def get_current_portfolio(conn):
    """
    It returns the current portfolio, showing: ticker, name, sector, total quantity, average price and invested value.
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM current_portfolio")
    return cur.fetchall()


def get_historical_portfolio(conn, date):
    """
    Returns the portfolio until the given date.
    Calculates total quantity and weighted average price manually.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ticker,
            SUM(quantity) AS total_quantity,
            ROUND(
                SUM(CASE WHEN quantity > 0 THEN quantity * price ELSE 0 END)
                / NULLIF(SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END), 0), 2
            ) AS avg_price,
            ROUND(
                SUM(quantity) *
                (SUM(CASE WHEN quantity > 0 THEN quantity * price ELSE 0 END)
                / NULLIF(SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END), 0)), 2
            ) AS invested_value
        FROM transactions
        WHERE date <= ?
        GROUP BY ticker
        HAVING total_quantity > 0
    """, (date,))
    return cur.fetchall()

def get_best_avg_price(conn):
    """
    It returns the title with the highest average purchase price from the current portfolio.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker, name, sector, avg_price
        FROM current_portfolio
        ORDER BY avg_price DESC
        LIMIT 1
    """)
    return cur.fetchone()

def get_transactions_by_ticker(conn, ticker):
    """
    It returns all the transactions of a title.
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM transactions WHERE ticker = ?", (ticker,))
    return cur.fetchall()

def get_transactions_by_date(conn, start_date, end_date):
    """
    It returns all the transactions in a date interval.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM transactions
        WHERE date BETWEEN ? AND ?
    """, (start_date, end_date))
    return cur.fetchall()

def get_portfolio_by_sector(conn):
    """
    It returns the total invested value per sector from the current portfolio.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT sector, SUM(invested_value) AS total_invested
        FROM current_portfolio
        GROUP BY sector
    """)
    return cur.fetchall()

def update_transaction(conn, transaction_id, **kwargs):
    """
    It updates the field of a transaction.
    kwargs: date, ticker, name, sector, quantity, price
    """
    cur = conn.cursor()
    fields = []
    values = []
    for key, value in kwargs.items():
        fields.append(f"{key} = ?")
        values.append(value)
    values.append(transaction_id)
    sql = f"UPDATE transactions SET {', '.join(fields)} WHERE id = ?"
    cur.execute(sql, values)
    conn.commit()

def get_portfolio_summary(conn):
    """
    It returns total quantity, total value and average price of the portfolio.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            SUM(total_quantity) AS total_quantity,
            ROUND(SUM(invested_value), 2) AS total_invested,
            ROUND(SUM(total_quantity * avg_price) / NULLIF(SUM(total_quantity), 0), 2) AS weighted_avg_price
        FROM current_portfolio
    """)
    return cur.fetchone()

def get_sector_allocation(conn):
    """
    It returns the percentage of investments for sector.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT sector, SUM(invested_value) AS total_invested
        FROM current_portfolio
        GROUP BY sector
    """)
    data = cur.fetchall()
    total = sum(row[1] for row in data)
    allocation = [(row[0], round(row[1]/total*100, 2)) for row in data]
    return allocation

