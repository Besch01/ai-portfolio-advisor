
# main.py
from agent.agent import Agent
from agent.llm_client import LLMClient
import os

def main():
    # Inizializziamo il "cervello" (Fake LLM) e l'"orchestratore" (Agent)
    llm_client = LLMClient()
    portfolio_agent = Agent(llm_client)

    print("=== AI PORTFOLIO ADVISOR - PRONTO ===")
    print("Scrivi un comando (es. 'Mostra portafoglio' o 'Compra 10 NVDA')")
    print("Scrivi 'esci' per chiudere.\n")

    while True:
        user_input = input("Tu: ")
        
        if user_input.lower() in ["esci", "quit", "exit"]:
            print("Chiusura assistente. Arrivederci!")
            break

        # L'agente analizza e risponde
        try:
            response = portfolio_agent.run(user_input)
            
            # Se la risposta è un dizionario (es. i dati del DB), la stampiamo bene
            if isinstance(response, dict):
                print(f"\nAgente: Ecco i dati richiesti:")
                # Se è il portafoglio, cicliamo i dati
                if 'data' in response and isinstance(response['data'], list):
                    for item in response['data']:
                        print(f"- {item['ticker']}: {item['name']} | Quantità: {item['total_quantity']} | Settore: {item['sector']}")
                else:
                    print(response)
            else:
                # Se è una risposta testuale semplice (il "thought")
                print(f"\nAgente: {response}")
                
        except Exception as e:
            print(f"\n[ERRORE]: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()