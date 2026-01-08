"""
Database tools for portfolio management

Functions to manage transactions and portfolio data

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
    
    # allows the access through column' name (essential for analysis_tools)
    conn.row_factory = sqlite3.Row 
    return conn


def insert_transaction(transaction_data):
    """
    Insert a new transaction in the database using a dict from get_market_transaction.

    Parameters:
    - transaction_data (dict): keys = date, ticker, name, sector, quantity, price
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO transactions (date, ticker, name, sector, quantity, price)
            VALUES (:date, :ticker, :name, :sector, :quantity, :price)
            """,
            transaction_data
        )
        conn.commit()
    finally:
        conn.close()


def get_current_portfolio():
    """
     It returns the current portfolio, showing: ticker, name, sector, total quantity, average price and invested value.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM current_portfolio")
        return cur.fetchall()
    finally:
        conn.close()


def get_historical_portfolio(date):
    """
    Returns the portfolio until the given date.
    Calculates total quantity and weighted average price.
    """
    conn = get_connection()
    try:
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
    finally:
        conn.close()
 
        
def get_best_avg_price():
    """
    Returns the title with the highest average purchase price from the current portfolio.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker, name, sector, avg_price
            FROM current_portfolio
            ORDER BY avg_price DESC
            LIMIT 1
        """)
        return cur.fetchone()
    finally:
        conn.close()


def get_transactions_by_ticker(ticker):
    """
    Returns all the transactions of a specific ticker.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM transactions WHERE ticker = ?", (ticker,))
        return cur.fetchall()
    finally:
        conn.close()


def get_transactions_by_date(start_date, end_date):
    """
    Returns all transactions within a date interval.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM transactions
            WHERE date BETWEEN ? AND ?
        """, (start_date, end_date))
        return cur.fetchall()
    finally:
        conn.close()


def get_portfolio_by_sector():
    """
    Returns the total invested value per sector from the current portfolio.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT sector, SUM(invested_value) AS total_invested
            FROM current_portfolio
            GROUP BY sector
        """)
        return cur.fetchall()
    finally:
        conn.close()

def delete_transaction(transaction_id=None):
    """
    Deletes a transaction from the database with user confirmation.
    
    Parameters:
        - transaction_id (int, optional): ID of the transaction to delete.
          If None, deletes the last transaction.
          
    Returns:
        - str: message about the result
    """
    conn = get_connection()
    cur = conn.cursor()

    # If no ID, take the last transaction
    if transaction_id is None:
        cur.execute("SELECT id, date, ticker, quantity FROM transactions ORDER BY id DESC LIMIT 1")
        transaction = cur.fetchone()
        if transaction is None:
            conn.close()
            return "No transactions to delete."
        transaction_id = transaction[0]
    else:
        cur.execute("SELECT id, date, ticker, quantity FROM transactions WHERE id = ?", (transaction_id,))
        transaction = cur.fetchone()
        if transaction is None:
            conn.close()
            return f"No transaction with ID {transaction_id}."

    # Show the details of the transaction and ask for confirm
    print(f"Selected transaction: ID {transaction[0]}, Date: {transaction[1]}, Ticker: {transaction[2]}, Quantity: {transaction[3]}")
    confirm = input(f"Are you sure to delete this transaction? (y/n): ").strip().lower()
    if confirm != 'y':
        conn.close()
        return "The user stopped the deleting."

    # Delete the transaction
    cur.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()
    
    return f"Transaction ID {transaction_id} deleted successfully."


def update_transaction(transaction_id, **kwargs):
    """
    Update fields of a transaction.
    kwargs: date, ticker, name, sector, quantity, price
    """
    conn = get_connection()
    try:
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
    finally:
        conn.close()


def get_sector_allocation():
    """
    Returns percentage of investments per sector.
    """
    conn = get_connection()
    try:
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
    finally:
        conn.close()


def get_portfolio_summary():
    """
    Returns total quantity, total invested value, and weighted average price of the portfolio.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                SUM(total_quantity) AS total_quantity,
                ROUND(SUM(invested_value), 2) AS total_invested,
                ROUND(SUM(total_quantity * avg_price) / NULLIF(SUM(total_quantity), 0), 2) AS weighted_avg_price
            FROM current_portfolio
        """)
        return cur.fetchone()
    finally:
        conn.close()
