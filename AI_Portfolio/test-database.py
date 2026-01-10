# test_integration.py (nella cartella root)

# Importiamo le funzioni dai pacchetti corretti
from tools.api.api_tools import get_market_transaction_data
from tools.database.db_tools import insert_transaction, get_current_portfolio

def run_full_test():
    print("=== INIZIO TEST INTEGRAZIONE AGENTE ===")
    ticker = "NVDA"
    quantity = 10

    try:
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

        # 3. Verifica finale
        print("[*] Verifica stato portafoglio...")
        port = get_current_portfolio()
        if any(item['ticker'] == ticker for item in port['data']):
            print("\n✅ TEST SUPERATO: Il flusso completo funziona perfettamente!")
        else:
            print("\n❌ TEST FALLITO: Dati non trovati nella View.")

    except Exception as e:
        print(f"\n❌ ERRORE DURANTE IL TEST: {e}")

if __name__ == "__main__":
    run_full_test()