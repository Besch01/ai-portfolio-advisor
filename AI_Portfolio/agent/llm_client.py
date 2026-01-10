import json

class LLMClient:
    """
    LLM fittizio per testare l'agent senza dipendere da API reali.
    Analizza l'input e restituisce il JSON corretto per attivare i tool.
    """

    def chat(self, messages):
        user_input = messages[-1]["content"].lower().strip()
        
        # AGGIUNGI QUESTI DUE PRINT PER VEDERE COSA SUCCEDE DENTRO
        print(f"--- DEBUG CLIENT ---")
        print(f"Input ricevuto: '{user_input}'")
        
        if "portfolio" in user_input:
            print("DEBUG: Entrato nel caso PORTFOLIO") # <--- Aggiungi questo
            response = {
                "thought": "Fetching portfolio data.",
                "tool": "get_current_portfolio",
                "args": {}
            }
        elif "buy" in user_input:
            print("DEBUG: Entrato nel caso BUY") # <--- Aggiungi questo
            response = {
                "thought": "Buying stocks.",
                "tool": "buy_stock_flow",
                "args": {"ticker": "NVDA", "quantity": 10}
            }
        elif "price" in user_input:
            print("DEBUG: Entrato nel caso PRICE")
            response = {
                "thought": "Checking price.",
                "tool": "get_market_transaction_data",
                "args": {"ticker": "NVDA", "quantity": 1}
            }
        else:
            print("DEBUG: Entrato nel caso ELSE (Saluto)") # <--- Aggiungi questo
            response = {
                "thought": "Standard greeting.",
                "tool": None,
                "args": {}
            }
            
        return json.dumps(response)