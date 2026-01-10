from tools.api.api_tools import get_market_transaction_data
from tools.database.db_tools import insert_transaction, get_current_portfolio
import sqlite3
import os

def run_full_test():
    print("=== INIZIO TEST INTEGRAZIONE AGENTE ===")
    ticker = "NVDA"
    quantity = 10

    try:
        # --- OPZIONALE: PULIZIA AUTOMATICA PER IL TEST ---
        # Questo assicura che il test parta sempre da zero
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "tools", "database", "portfolio_manager.db")
        # Se vuoi pulire il DB ogni volta che lanci il test, decommenta le due righe sotto:
        # conn = sqlite3.connect(db_path)
        # conn.execute("DELETE FROM transactions") 
        # conn.commit()

        # 1. Recupero dati reali (Yahoo Finance)
        print(f"[*] Recupero dati di mercato per {ticker}...")
        market_data = get_market_transaction_data(ticker, quantity)
        print(f"[OK] Dati ricevuti: {market_data['name']} - Prezzo: ${market_data['price']}")

        # 2. Inserimento nel Database
        print("[*] Inserimento nel database locale...")
        res = insert_transaction(market_data)
        
        if res["status"] == "ok":
            print(f"[OK] Salvataggio completato! ID Transazione: {res['data']['transaction_id']}")
        else:
            print(f"[ERRORE] Inserimento fallito: {res.get('message')}")
            return

        # 3. Verifica finale e stampa dettagliata
        print("[*] Verifica stato portafoglio...")
        port = get_current_portfolio()
        
        found = False
        if port["status"] == "ok":
            for row in port['data']:
                if row['ticker'] == ticker:
                    print(f"\n✅ TEST SUPERATO: {ticker} trovato nel portafoglio!")
                    print(f"   DETTAGLI -> Nome: {row['name']}")
                    print(f"   DETTAGLI -> Settore: {row['sector']}")
                    print(f"   DETTAGLI -> Quantità Totale: {row['total_quantity']}")
                    found = True
        
        if not found:
            print("\n❌ TEST FALLITO: Dati non trovati nella View.")

    except Exception as e:
        print(f"\n❌ ERRORE DURANTE IL TEST: {e}")

if __name__ == "__main__":
    run_full_test()
    