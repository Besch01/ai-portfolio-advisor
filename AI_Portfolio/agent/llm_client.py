# agent/llm_client.py

import json

class LLMClient:
    """
    LLM fittizio per testare l'agent senza dipendere da API reali.
    Restituisce sempre un JSON simulato che indica quale tool chiamare.
    """

    def chat(self, messages):
        """
        messages: lista di dict {"role":..., "content":...}
        ritorna JSON simulato come stringa
        """
        # Per ora il mock chiama sempre get_portfolio
        response = {
            "thought": "Simulazione: sto processando la richiesta",
            "tool": "get_portfolio",  # esempio di tool chiamato
            "args": {}  # argomenti del tool
        }
        return json.dumps(response)
