# agent/agent.py

import json
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# --- 1. REAL TOOLS IMPORT ---
# Connecting the "muscles" (APIs and DB) to the "brain" (Agent)

# Database tools
from tools.database.db_tools import (
    insert_transaction, 
    get_sector_allocation, 
    get_current_portfolio
)

# API tools
from tools.api.api_tools import (
    get_market_transaction_data, 
    buy_stock_flow
)

# Analysis tools
from tools.analysis.analysis_tools import (
    tool_compute_returns,
    tool_optimize_markowitz_target,
    tool_sentiment_analysis
)

# Visualization tools
from tools.visualization.visualization_tools import (
    plot_portfolio_composition,
    plot_sector_allocation,
    plot_portfolio_value_over_time
)

# --- 2. TOOL REGISTRY ---
# Mapping string names from the LLM to actual Python functions
TOOLS = {
    # Data & Transactions
    "get_current_portfolio": get_current_portfolio,
    "get_market_transaction_data": get_market_transaction_data,
    "insert_transaction": insert_transaction,
    "buy_stock_flow": buy_stock_flow,
    
    # Financial Analysis
    "compute_roi": tool_compute_returns,
    "optimize_portfolio": tool_optimize_markowitz_target,
    "analyze_sentiment": tool_sentiment_analysis,
    
    # Visualizations
    "show_composition_chart": plot_portfolio_composition,
    "show_performance_chart": plot_portfolio_value_over_time,
    "show_sector_chart": plot_sector_allocation
}



class Agent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.system_prompt = SYSTEM_PROMPT

    def run(self, user_input):
        # Build the prompt for the LLM using the template
        prompt = USER_PROMPT_TEMPLATE.format(user_input=user_input)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # LLM Call (The FakeLLMClient or real FinGPT will process this)
        response = self.llm.chat(messages)

        # JSON Parsing
        try:
            # Handle both string responses and direct dictionary responses
            if isinstance(response, dict):
                decision = response
            else:
                decision = json.loads(response)
        except json.JSONDecodeError:
            return "Error: LLM did not return valid JSON."

        tool_name = decision.get("tool")
        args = decision.get("args", {})

        # --- 3. TOOL EXECUTION ---
        if tool_name and tool_name in TOOLS:
            try:
                # The agent triggers the corresponding function in the TOOLS dictionary
                # The **args syntax passes dictionary keys as function arguments
                result = TOOLS[tool_name](**args)
                return result
            except Exception as e:
                return f"Error running tool '{tool_name}': {e}"
        else:
            # No tool required (e.g., a general question or greeting)
            return decision.get("thought", "No action taken.")
        