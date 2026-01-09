"""
Test file for db_tools
Checks all main functions and prints results.
"""

from db_tools import (
    insert_transaction,
    get_current_portfolio,
    get_historical_portfolio,
    get_best_avg_price,
    get_transactions_by_ticker,
    get_transactions_by_date,
    get_portfolio_by_sector,
    get_sector_allocation,
    get_portfolio_summary,
    update_transaction,
    delete_transaction
)

# --- Inserimento transazioni di test ---
print("=== Inserimento transazioni di test ===")
tx1 = {"date": "2026-01-08", "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "quantity": 10, "price": 150.0}
tx2 = {"date": "2026-01-08", "ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "quantity": 5, "price": 300.0}

res1 = insert_transaction(tx1)
res2 = insert_transaction(tx2)

print(res1)
print(res2)


# --- Portfolio corrente ---
print("\n=== Portfolio corrente ===")
portfolio = get_current_portfolio()
print(portfolio)


# --- Portfolio storico fino a oggi ---
print("\n=== Portfolio storico fino a oggi ===")
historical = get_historical_portfolio("2026-01-08")
print(historical)


# --- Titolo con prezzo medio più alto ---
print("\n=== Titolo con prezzo medio più alto ===")
best_avg = get_best_avg_price()
print(best_avg)


# --- Transazioni per AAPL ---
print("\n=== Transazioni per AAPL ===")
tx_aapl = get_transactions_by_ticker("AAPL")
print(tx_aapl)

# --- Aggiornamento ultima transazione AAPL ---
if tx_aapl["status"] == "ok" and tx_aapl["data"]:
    last_tx_id = tx_aapl["data"][-1]["id"]
    print(f"\n=== Update ultima transazione AAPL, ID {last_tx_id} ===")
    update_res = update_transaction(last_tx_id, quantity=20, price=150.0)
    print(update_res)
else:
    print("Nessuna transazione AAPL trovata")


# --- Summary del portfolio ---
print("\n=== Summary del portfolio ===")
summary = get_portfolio_summary()
print(summary)


# --- Allocazione per settore ---
print("\n=== Allocazione per settore ===")
sector_alloc = get_sector_allocation()
print(sector_alloc)


# --- Portfolio per settore (totale investito) ---
print("\n=== Portfolio per settore (totale investito) ===")
by_sector = get_portfolio_by_sector()
print(by_sector)


# --- Transazioni per data ---
print("\n=== Transazioni tra 2026-01-01 e 2026-01-08 ===")
tx_date = get_transactions_by_date("2026-01-01", "2026-01-08")
print(tx_date)


# --- Test delete transaction (ultima inserita) ---
print("\n=== Test delete transaction (ultima inserita) ===")
last_tx = get_transactions_by_ticker("MSFT")
if last_tx["status"] == "ok" and last_tx["data"]:
    last_tx_id = last_tx["data"][-1]["id"]
    del_res = delete_transaction(last_tx_id, confirm=True)
    print(del_res)
else:
    print("Nessuna transazione MSFT trovata")

