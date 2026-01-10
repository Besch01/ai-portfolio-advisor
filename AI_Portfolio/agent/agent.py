# agent/agent.py

import json
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# --- 1. IMPORT DEI TOOL REALI ---
# Qui colleghiamo i muscoli (API e DB) al cervello (Agente)
from tools.api.api_tools import get_market_transaction_data
from tools.database.db_tools import insert_transaction, get_sector_allocation, get_current_portfolio
#from tools.visualization.visualization_tools import plot_sector_allocation 
from tools.api.api_tools import buy_stock_flow

TOOLS = {
    # <--- Aggiungi questo!
    "get_current_portfolio": get_current_portfolio,
    #"plot_sector_allocation": plot_sector_allocation,
    "get_market_transaction_data": get_market_transaction_data,
    "insert_transaction": insert_transaction,
    "buy_stock_flow": buy_stock_flow
}

class Agent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.system_prompt = SYSTEM_PROMPT

    def run(self, user_input):
        # Costruzione del prompt per l'LLM
        prompt = USER_PROMPT_TEMPLATE.format(user_input=user_input)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Chiamata LLM (qui interviene il tuo FakeLLMClient)
        response = self.llm.chat(messages)

        # Parsing JSON
        try:
            decision = json.loads(response)
        except json.JSONDecodeError:
            return "Error: LLM did not return valid JSON."

        tool_name = decision.get("tool")
        args = decision.get("args", {})

        # --- 3. ESECUZIONE TOOL ---
        if tool_name and tool_name in TOOLS:
            try:
                # L'agente lancia la funzione corrispondente nel dizionario TOOLS
                result = TOOLS[tool_name](**args)
                
                # Se il tool è l'API di mercato, l'output va salvato nel DB automaticamente?
                # Per ora restituiamo il risultato, poi vedremo come concatenarli.
                return result
            except Exception as e:
                return f"Error running tool '{tool_name}': {e}"
        else:
            # Nessun tool necessario (es. una domanda generica)
            return decision.get("thought", "No action taken.")
