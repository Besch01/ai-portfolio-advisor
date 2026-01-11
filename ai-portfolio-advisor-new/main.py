# main.py
from agent.agent import Agent
from agent.llm_client import LLMClient
import os
import pandas as pd

def main():
    # Initialize the "brain" (Fake LLM) and the "orchestrator" (Agent)
    llm_client = LLMClient()
    portfolio_agent = Agent(llm_client)

    # Ensure plot directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    print("\n" + "="*50)
    print("=== FINANCIAL AI PORTFOLIO ADVISOR ===")
    print("="*50)
    print("- Database: portfolio [date], transactions [ticker/dates], summary")
    print("- Analysis: performance, sector diversification, optimize, news")
    print("- Charts: pie chart, benchmark chart, correlation chart, advice chart")
    print("- Reporting: PDF portfolio overview, PDF risk & optimization")
    print("- FinGPT: consultant & risk manager (optional insights)")
    print("-" * 50)
    print("Type 'exit' to close.\n")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Closing assistant. Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            # The Agent analyzes the input and runs the tools
            response = portfolio_agent.run(user_input)
            
            # --- 1. Handle Dictionary Responses (DB Data, Tools, Charts) ---
            if isinstance(response, dict):
                # CASE A: Standard Tool Message
                if 'message' in response:
                    print(f"\nAgent: {response['message']}")
                
                # CASE B: Chart/Visualization Output
                if 'image_path' in response and response['image_path']:
                    print(f"📊 Success: Plot saved to: {response['image_path']}")
                
                # CASE C: Portfolio List (Current or Historical)
                if 'data' in response and isinstance(response['data'], list):
                    print(f"\nAgent: I found the following holdings:")
                    for item in response['data']:
                        ticker = item.get('ticker', 'N/A')
                        name = item.get('name', 'N/A')
                        qty = item.get('total_quantity', item.get('quantity', 0))
                        price = item.get('avg_price', item.get('price', 0))
                        print(f" • {ticker:5} | {name:20} | Qty: {qty:4} | Avg Price: ${price:.2f}")
                
                # CASE D: Optimization or Sentiment Results (Nested Dict)
                elif 'optimized_weights' in response:
                    print("\nAgent: Markowitz Optimization Results:")
                    print(f" - Est. Volatility: {response.get('estimated_volatility', 0)*100:.2f}%")
                    for t, w in response['optimized_weights'].items():
                        print(f" • {t}: {w*100:.2f}%")
                
                elif 'sentiment_label' in response:
                    print(f"\nAgent: Sentiment for {response.get('ticker')}: {response.get('sentiment_label')}")
                    print(f" - Average Score: {response.get('average_score')} (based on {response.get('article_count')} articles)")

                # Default fallback for dict
                elif not any(k in response for k in ['data', 'image_path', 'message']):
                    print(f"\nAgent Data: {response}")

            # --- 2. Handle DataFrame Responses (Comparison tables, Drift, etc.) ---
            elif isinstance(response, pd.DataFrame):
                print(f"\nAgent: Analysis Table:")
                if response.empty:
                    print("No data available for this analysis.")
                else:
                    print(response.to_string(index=False))

            # --- 3. Handle Simple Text Responses (Thoughts/Greetings) ---
            else:
                print(f"\nAgent: {response}")
                
        except Exception as e:
            print(f"\n[SYSTEM ERROR]: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()