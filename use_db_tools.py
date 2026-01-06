"""
Test file for database_tools
"""

from db_tools import (
    get_connection,
    insert_transaction,
    get_current_portfolio,
    get_historical_portfolio,
    get_best_avg_price,
    get_transactions_by_ticker,
    get_transactions_by_date,
    get_portfolio_by_sector,
    get_portfolio_summary,
    get_sector_allocation
)
import datetime

# =========================
# 1. Apri connessione
# =========================
conn = get_connection()

# =========================
# 2. Inserimento transazioni manuali
# =========================
insert_transaction(conn, "2025-01-02", "AAPL", "Apple Inc.", "Technology", 10, 242.75)
insert_transaction(conn, "2025-06-02", "AAPL", "Apple Inc.", "Technology", -3, 201.28)
insert_transaction(conn, "2025-01-02", "MSFT", "Microsoft Corp.", "Technology", 8, 415.51)

# =========================
# 3. Portfolio corrente
# =========================
print("=== Portfolio corrente ===")
portfolio = get_current_portfolio(conn)
for row in portfolio:
    print(row)

# =========================
# 4. Portfolio storico a una data
# =========================
print("\n=== Portfolio storico al 2025-06-01 ===")
historical = get_historical_portfolio(conn, "2025-06-01")
for row in historical:
    print(row)

# =========================
# 5. Titolo con prezzo medio più alto
# =========================
best_avg = get_best_avg_price(conn)
print("\n=== Titolo con prezzo medio più alto ===")
print(best_avg)

# =========================
# 6. Transazioni di un ticker
# =========================
print("\n=== Transazioni AAPL ===")
tx_aapl = get_transactions_by_ticker(conn, "AAPL")
for row in tx_aapl:
    print(row)

# =========================
# 7. Transazioni per intervallo di date
# =========================
print("\n=== Transazioni tra 2025-01-01 e 2025-06-02 ===")
tx_range = get_transactions_by_date(conn, "2025-01-01", "2025-06-02")
for row in tx_range:
    print(row)

# =========================
# 8. Summary del portfolio
# =========================
print("\n=== Summary del portfolio ===")
summary = get_portfolio_summary(conn)
print(summary)

# =========================
# 9. Allocazione per settore
# =========================
print("\n=== Allocazione per settore ===")
allocation = get_sector_allocation(conn)
for sector, perc in allocation:
    print(f"{sector}: {perc}%")

# =========================
# 10. Totale investito per settore
# =========================
print("\n=== Totale investito per settore ===")
sector_totals = get_portfolio_by_sector(conn)
for sector, total in sector_totals:
    print(f"{sector}: {total}")

# =========================
# 11. Chiudi connessione
# =========================
conn.close()
