"""
Test file for database_tools
"""

from db_tools import (
    insert_transaction,
    get_current_portfolio,
    get_historical_portfolio,
    get_best_avg_price,
    get_transactions_by_ticker,
    get_transactions_by_date,
    get_portfolio_by_sector,
    update_transaction,
    get_portfolio_summary,
    get_sector_allocation,
    delete_transaction
)
from datetime import date

# ===============================
# 1. Inserimento transazioni di test
# ===============================
print("\n=== Inserimento transazioni di test ===")
tx1 = {
    "date": date.today().isoformat(),
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "quantity": 10,
    "price": 150.0
}

tx2 = {
    "date": date.today().isoformat(),
    "ticker": "MSFT",
    "name": "Microsoft Corp.",
    "sector": "Technology",
    "quantity": 5,
    "price": 300.0
}

insert_transaction(tx1)
insert_transaction(tx2)
print("Transazioni inserite.")

# ===============================
# 2. Portfolio corrente
# ===============================
print("\n=== Portfolio corrente ===")
portfolio = get_current_portfolio()
for row in portfolio:
    print(dict(row))  # row_factory sqlite3.Row permette di convertire in dict

# ===============================
# 3. Portfolio storico fino a oggi
# ===============================
print("\n=== Portfolio storico fino a oggi ===")
historical = get_historical_portfolio(date.today().isoformat())
for row in historical:
    print(dict(row))

# ===============================
# 4. Miglior prezzo medio
# ===============================
print("\n=== Titolo con prezzo medio più alto ===")
best_avg = get_best_avg_price()
print(dict(best_avg) if best_avg else "Nessun dato")

# ===============================
# 5. Transazioni per ticker
# ===============================
print("\n=== Transazioni per AAPL ===")
aapl_tx = get_transactions_by_ticker("AAPL")
for row in aapl_tx:
    print(dict(row))

# ===============================
# 6. Aggiornamento transazione
# ===============================
if aapl_tx:
    last_aapl_id = aapl_tx[-1]["id"]
    print(f"\n=== Aggiornamento ultima transazione AAPL, ID {last_aapl_id} ===")
    update_transaction(last_aapl_id, quantity=20)
    print("Aggiornamento completato.")
    # Verifica
    updated = get_transactions_by_ticker("AAPL")[-1]
    print(dict(updated))

# ===============================
# 7. Riepilogo portfolio
# ===============================
print("\n=== Summary del portfolio ===")
summary = get_portfolio_summary()
print(dict(summary) if summary else "Nessun dato")

# ===============================
# 8. Allocazione per settore
# ===============================
print("\n=== Allocazione per settore ===")
allocation = get_sector_allocation()
for row in allocation:
    print(row)

# ===============================
# 9. Eliminazione transazione
# ===============================
print("\n=== Test delete transaction (ultima inserita) ===")
msg = delete_transaction()  # eliminerà ultima transazione con conferma utente
print(msg)
