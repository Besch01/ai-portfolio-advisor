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

# 1. Open connection
conn = get_connection()

#2. Insert manual transactions
insert_transaction(conn, "2025-01-02", "AAPL", "Apple Inc.", "Technology", 10, 242.75)
insert_transaction(conn, "2025-06-02", "AAPL", "Apple Inc.", "Technology", -3, 201.28)
insert_transaction(conn, "2025-01-02", "MSFT", "Microsoft Corp.", "Technology", 8, 415.51)

#3. Current portfolio
print("Current portfolio")
portfolio = get_current_portfolio(conn)
for row in portfolio:
    print(row)

#4. Historical Portfolio at a certain date
print("\nHistorical portfolio at the 2025-06-01")
historical = get_historical_portfolio(conn, "2025-06-01")
for row in historical:
    print(row)

#5. Title with the best average price
best_avg = get_best_avg_price(conn)
print("\nTitle with the best average price")
print(best_avg)

#6. Ticker transactions
print("\nAAPL's transactions")
tx_aapl = get_transactions_by_ticker(conn, "AAPL")
for row in tx_aapl:
    print(row)

#7. Transactions for a dates interval
print("\nTransactions between 2025-01-01 and 2025-06-02")
tx_range = get_transactions_by_date(conn, "2025-01-01", "2025-06-02")
for row in tx_range:
    print(row)

#8. Sectors' allocations
print("\nSectors' allocations")
allocation = get_sector_allocation(conn)
for sector, perc in allocation:
    print(f"{sector}: {perc}%")

#9. Total invested per sector
print("\nTotal invested per sector")
sector_totals = get_portfolio_by_sector(conn)
for sector, total in sector_totals:
    print(f"{sector}: {total}")

#10. Summary of the Portfolio
print("\nSummary of the portfolio")
summary = get_portfolio_summary(conn)
print(summary)















#11. Close connection
conn.close()
