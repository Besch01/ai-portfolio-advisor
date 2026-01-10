import json

class LLMClient:
    """
    Fake LLM Client for testing.
    Processes English keywords and maps them to financial and visualization tools.
    """

    def chat(self, messages):
        # Get the user message and clean it
        user_input = messages[-1]["content"].lower().strip()
        
        print(f"\n--- DEBUG CLIENT ---")
        print(f"Input received: '{user_input}'")
        
        # 1. PORTFOLIO
        if "portfolio" in user_input:
            print("DEBUG: Case PORTFOLIO")
            response = {
                "thought": "Fetching the latest data from your portfolio database.",
                "tool": "get_current_portfolio",
                "args": {}
            }
        
        # 2. BUY TRANSACTION
        elif "buy" in user_input:
            print("DEBUG: Case BUY")
            response = {
                "thought": "Executing buy transaction flow for NVDA.",
                "tool": "buy_stock_flow",
                "args": {"ticker": "NVDA", "quantity": 10}
            }

        # 3. ANALYSIS: ROI / PERFORMANCE
        elif "performance" in user_input or "roi" in user_input:
            print("DEBUG: Case ANALYSIS (ROI)")
            response = {
                "thought": "Calculating portfolio returns based on cost vs market prices.",
                "tool": "compute_roi",
                "args": {}
            }

        # 4. ANALYSIS: OPTIMIZATION (MARKOWITZ)
        elif "optimize" in user_input:
            print("DEBUG: Case ANALYSIS (MARKOWITZ)")
            response = {
                "thought": "Running Markowitz optimization for a 10% annual return target.",
                "tool": "optimize_portfolio",
                "args": {"target_return_annualized": 0.10}
            }

        # 5. ANALYSIS: SENTIMENT
        elif "news" in user_input or "sentiment" in user_input:
            print("DEBUG: Case ANALYSIS (SENTIMENT)")
            response = {
                "thought": "Performing sentiment analysis on recent market news for NVDA.",
                "tool": "analyze_sentiment",
                "args": {"ticker": "NVDA"}
            }

        # 6. VISUALIZATION: CHARTS
        elif "chart" in user_input or "plot" in user_input or "show" in user_input:
            print("DEBUG: Case VISUALIZATION")
            response = {
                "thought": "Generating visualization for portfolio composition.",
                "tool": "show_composition_chart",
                "args": {"save_path": "plots/composition.png"}
            }

        # 7. MARKET PRICE
        elif "price" in user_input:
            print("DEBUG: Case PRICE")
            response = {
                "thought": "Retrieving real-time market data for NVDA.",
                "tool": "get_market_transaction_data",
                "args": {"ticker": "NVDA", "quantity": 1}
            }

        # 8. DEFAULT GREETING
        else:
            print("DEBUG: Case ELSE (Greeting)")
            response = {
                "thought": "Hello! I am your Portfolio Assistant. Use keywords like 'portfolio', 'buy', 'performance', 'optimize', or 'chart'.",
                "tool": None,
                "args": {}
            }
            
        return json.dumps(response)
    