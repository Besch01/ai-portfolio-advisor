# main.py

from tools.visualization.visualization_tools import plot_sector_allocation  # esempio

# 1️⃣ Tool registry (cuore dell'agent)
TOOLS = {
    "plot_sector_allocation": plot_sector_allocation
}

# 2️⃣ Fake LLM response (simula JSON dell'LLM)
def fake_llm_response(user_input: str) -> dict:
    """
    Simula la risposta strutturata di un LLM.
    """
    print(f"\n[LLM INPUT]: {user_input}")

    # Hardcoded per il test
    return {
        "tool": "plot_sector_allocation",
        "args": {
            "portfolio_id": 1
        }
    }

# 3️⃣ Agent executor
def run_agent(user_input: str):
    response = fake_llm_response(user_input)

    tool_name = response.get("tool")
    args = response.get("args", {})

    if tool_name not in TOOLS:
        raise ValueError(f"Tool '{tool_name}' not found")

    print(f"[AGENT]: Calling tool '{tool_name}' with args {args}")

    result = TOOLS[tool_name](**args)

    print(f"[RESULT]: {result}")

# 4️⃣ Entry point
if __name__ == "__main__":
    print("=== AI Portfolio Advisor (TEST MODE) ===")

    while True:
        user_input = input("\nUser > ")

        if user_input.lower() in {"exit", "quit"}:
            print("Bye 👋")
            break

        run_agent(user_input)
