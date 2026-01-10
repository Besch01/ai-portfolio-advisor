# test_analysis_tools.py
import pprint
from tools.analysis.analysis_tools import agent_tools  # Assumendo che le funzioni siano qui

def test_analysis_tools():
    """
    Test rapido di tutte le funzioni principali di analysis_tools.
    Stampa a terminale gli output per verifica.
    """
    pp = pprint.PrettyPrinter(indent=2)

    print("\n--- TEST: ROI Portfolio ---")
    roi_output = agent_tools['returns_analyzer']()
    pp.pprint(roi_output)

    print("\n--- TEST: Best Returns per Asset ---")
    best_returns = agent_tools['best_returns_data']()
    pp.pprint(best_returns[:5])  # Mostriamo solo i primi 5 per brevità

    print("\n--- TEST: Sector Diversification ---")
    diversification = agent_tools['diversification_expert']()
    pp.pprint(diversification)

    print("\n--- TEST: Markowitz Portfolio Optimization ---")
    optimizer = agent_tools['portfolio_optimizer'](0.10)
    pp.pprint(optimizer)

    print("\n--- TEST: Market Sentiment ---")
    sentiment = agent_tools['market_sentiment'](ticker='GOOGL')
    pp.pprint(sentiment)

if __name__ == "__main__":
    test_analysis_tools()
