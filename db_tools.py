"""
Database tools for portfolio manager
"""

import sqlite3
import os


def get_connection():
    """
    Returns a connection to the portfolio database
    """
    pass


def insert_transaction(conn, date, ticker, name, sector, quantity, price):
    """
    Inserts a new transaction into the database
    """
    pass


def get_current_portfolio(conn):
    """
    Returns the current portfolio (aggregated)
    """
    pass
