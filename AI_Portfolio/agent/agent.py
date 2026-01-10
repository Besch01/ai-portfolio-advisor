# agent/agent.py

import json
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Import dei tool reali
from tools.database.db_tools import get_current_portfolio, get_portfolio_by_sector  # esempio
from tools.visualization.visualization_tools import plot_sector_allocation  # esempio

# Registry dei tool disponibili
TOOLS = {
    "plot_sector_allocation": plot_sector_allocation
}

class Agent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.system_prompt = SYSTEM_PROMPT

    def run(self, user_input):
        """
        Esegue una richiesta dell’utente:
        1. Invia prompt al LLM (fittizio o reale)
        2. Analizza la risposta JSON
        3. Chiama il tool se necessario
        4. Restituisce il risultato
        """
        # Costruzione del prompt per l'LLM
        prompt = USER_PROMPT_TEMPLATE.format(user_input=user_input)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Chiamata LLM
        response = self.llm.chat(messages)

        # Parsing JSON
        try:
            decision = json.loads(response)
        except json.JSONDecodeError:
            return "Error: LLM did not return valid JSON."

        tool_name = decision.get("tool")
        args = decision.get("args", {})

        # Esecuzione tool se specificato
        if tool_name and tool_name in TOOLS:
            try:
                result = TOOLS[tool_name](**args)
                return result
            except Exception as e:
                return f"Error running tool '{tool_name}': {e}"
        else:
            # Nessun tool necessario, restituisco il thought
            return decision.get("thought", "No action taken.")
