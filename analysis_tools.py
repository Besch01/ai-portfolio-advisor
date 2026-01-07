import sqlite3
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Import functions from your API tools file
# Ensure api_tools.py is in the same directory
from api_tools import get_latest_prices, get_historical_prices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "portfolio_manager.db")

def get_db_connection():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# 1. COMPUTE RETURNS
def tool_compute_returns():
    """
    Calculate the total portfolio return (ROI) using real-time market prices.
    Compares the current market value against the average purchase cost.
    """
    conn = get_db_connection()
    portfolio = conn.execute("SELECT * FROM current_portfolio WHERE total_quantity > 0").fetchall()
    conn.close()
    
    if not portfolio: 
        return "The portfolio is currently empty."

    tickers = [p['ticker'] for p in portfolio]
    # Fetch real-time prices via API
    current_prices = get_latest_prices(tickers)
    
    total_purchase_cost = 0
    total_current_value = 0

    for p in portfolio:
        ticker = p['ticker']
        qty = p['total_quantity']
        avg_price = p['avg_price']
        
        current_price = current_prices.get(ticker)
        
        if current_price:
            total_purchase_cost += qty * avg_price
            total_current_value += qty * current_price
    
    if total_purchase_cost == 0: 
        return "Error: Total purchase cost is zero."
    
    roi = ((total_current_value - total_purchase_cost) / total_purchase_cost) * 100
    return f"Total Real Return: {roi:.2f}% (Current Value: ${total_current_value:,.2f})"

# 2. SECTOR DIVERSIFICATION
def tool_sector_diversification():
    """
    Analyze capital distribution across different market sectors.
    Weights are calculated based on current market valuations.
    """
    conn = get_db_connection()
    query = """
    SELECT ticker, sector, SUM(quantity) as total_qty 
    FROM transactions GROUP BY ticker, sector HAVING total_qty > 0
    """
    df_portfolio = pd.read_sql_query(query, conn)
    conn.close()

    if df_portfolio.empty: 
        return "No data available for diversification analysis."

    # Update values with latest market prices
    tickers = df_portfolio['ticker'].tolist()
    prices = get_latest_prices(tickers)
    df_portfolio['current_price'] = df_portfolio['ticker'].map(prices)
    df_portfolio['total_value'] = df_portfolio['total_qty'] * df_portfolio['current_price']

    # Group by sector
    sector_dist = df_portfolio.groupby('sector')['total_value'].sum()
    total_val = sector_dist.sum()
    
    report = "Sector Diversification (Current Market Value):\n"
    for sector, val in sector_dist.items():
        percentage = (val / total_val) * 100
        report += f"- {sector}: {percentage:.1f}%\n"
    return report

# 3. MARKOWITZ OPTIMIZATION

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta

def tool_optimize_markowitz_target(target_return_annualized):
    """
    Trova l'allocazione ottimale per minimizzare il rischio dato un rendimento target.
    
    Args:
    target_return_annualized (float): Il rendimento target annuo (es. 0.10 per 10%)
    """
    
    # 1. Recupero dati (Simulato come nel tuo codice precedente)
    conn = get_db_connection()
    tickers = [row['ticker'] for row in conn.execute("SELECT ticker FROM current_portfolio WHERE total_quantity > 0").fetchall()]
    conn.close()

    if len(tickers) < 2: 
        return "Sono richiesti almeno 2 asset per l'ottimizzazione."

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') # Usiamo 1 anno per statistica più solida
    data = get_historical_prices(tickers, start_date, end_date)
    
    # 2. Calcolo metriche fondamentali
    daily_returns = data.pct_change().dropna()
    
    # Rendimenti attesi (Media storica annualizzata)
    expected_returns = daily_returns.mean() * 252
    
    # Matrice di Covarianza annualizzata
    cov_matrix = daily_returns.cov() * 252 

    # --- CONTROLLO DI FATTIBILITÀ ---
    # Non possiamo chiedere un rendimento più alto del titolo migliore, né più basso del peggiore
    max_possible_return = expected_returns.max()
    min_possible_return = expected_returns.min()
    
    if target_return_annualized > max_possible_return:
        return f"Target irrealistico: Il massimo rendimento storico tra i tuoi asset è {max_possible_return*100:.1f}%."

    # 3. Funzione Obiettivo: Minimizzare la varianza (RISCHIO)
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # 4. Vincoli (Constraints)
    # Vincolo 1: La somma dei pesi deve fare 1 (100% del capitale investito)
    cons_sum_weights = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Vincolo 2 (NUOVO): Il rendimento del portafoglio deve essere uguale al target richiesto
    # Formula: Somma(Peso * Rendimento_Atteso) - Target = 0
    cons_target_return = {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return_annualized}

    constraints = [cons_sum_weights, cons_target_return]

    # Bounds: No short selling (0 <= peso <= 1)
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    
    # Punto di partenza (Initial Guess)
    init_guess = [1/len(tickers)] * len(tickers)

    # 5. Ottimizzazione
    try:
        optimized = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    except Exception as e:
        return f"Ottimizzazione fallita: {str(e)}"

    if not optimized.success:
        return "L'ottimizzatore non è riuscito a trovare una soluzione valida per questo target."

    # 6. Report Risultati
    volatility = np.sqrt(optimized.fun) # La volatilità è la radice della varianza
    
    report = f"--- Markowitz Target: {target_return_annualized*100:.1f}% ---\n"
    report += f"Rischio Stimato (Volatilità): {volatility*100:.2f}%\n"
    report += "Allocazione Consigliata:\n"
    
    for i, ticker in enumerate(tickers):
        weight = optimized.x[i]
        # Mostriamo solo asset con peso rilevante (> 1%)
        if weight > 0.01:
            report += f"- {ticker}: {weight*100:.1f}%\n"
            
    return report

# 4. SENTIMENT ANALYSIS USING NEWSAPI AND NLP

from newsapi import NewsApiClient
from textblob import TextBlob

import sqlite3
from newsapi import NewsApiClient
from textblob import TextBlob

# --- CONFIGURAZIONE ---
# Incolla qui la tua API Key di NewsAPI
NEWS_API_KEY = 'f478f541562347d38b316ef6a2d19cac' 

def tool_sentiment_analysis(ticker=None):
    """
    Scarica le ultime news finanziarie e calcola un punteggio di sentiment
    usando NLP (TextBlob).
    """
    
    # 1. Se non viene passato un ticker, prendiamo quello con più valore nel portafoglio
    if not ticker:
        try:
            conn = sqlite3.connect('portfolio.db') # Assicurati che il nome del DB sia corretto
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query per trovare l'asset con la quantità totale maggiore
            row = cursor.execute("SELECT ticker FROM current_portfolio ORDER BY total_quantity DESC LIMIT 1").fetchone()
            conn.close()
            
            if not row:
                return "Errore: Nessun asset trovato nel database per l'analisi."
            
            ticker = row['ticker']
        except Exception as e:
            return f"Errore Database: {str(e)}"

    # 2. Inizializzazione NewsAPI
    if NEWS_API_KEY == 'LA_TUA_API_KEY_QUI':
        return "Errore: Devi inserire una API Key valida nel codice."
        
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Scarichiamo le news (ultimi 28 giorni max per il piano free)
        # Usiamo 'intitle' per cercare il ticker proprio nel titolo, per maggiore rilevanza
        print(f"--- Ricerca news per: {ticker} ---")
        headlines = newsapi.get_everything(
            q=ticker,
            language='en',
            sort_by='publishedAt', # Le più recenti prima
            page_size=5
        )
        
        articles = headlines.get('articles', [])
        if not articles:
            return f"Nessuna notizia recente trovata per {ticker}."

        scores = []
        report = f"\n--- Analisi Sentiment: {ticker} ---\n"
        
        for i, art in enumerate(articles):
            title = art['title']
            
            # 3. Analisi con TextBlob
            # Polarity va da -1 (Molto Negativo) a +1 (Molto Positivo)
            analysis = TextBlob(title)
            score = analysis.sentiment.polarity
            scores.append(score)
            
            # Etichetta leggibile
            label = "🟢 Positiva" if score > 0.1 else "🔴 Negativa" if score < -0.1 else "⚪️ Neutra"
            
            # Aggiungiamo al report (tronchiamo il titolo se troppo lungo)
            clean_title = (title[:75] + '..') if len(title) > 75 else title
            report += f"{i+1}. {clean_title}\n   -> Sentiment: {label} ({score:.2f})\n"

        # 4. Calcolo Finale
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.1:
            final_sentiment = "BULLISH (Rialzista) 🚀"
        elif avg_score < -0.1:
            final_sentiment = "BEARISH (Ribassista) 📉"
        else:
            final_sentiment = "NEUTRAL (Laterale) 😐"

        report += f"\n=== GIUDIZIO FINALE: {final_sentiment} ==="
        report += f"\n(Score Medio: {avg_score:.2f})"
        
        return report

    except Exception as e:
        return f"Errore API News: {str(e)}"



# AGENT TOOLS DICTIONARY
agent_tools = {
    "returns_analyzer": tool_compute_returns,
    "diversification_expert": tool_sector_diversification,
    "portfolio_optimizer": tool_optimize_markowitz_target,
    "market_sentiment": tool_sentiment_analysis
}

if __name__ == "__main__":
    print("--- Running Analysis Tools ---")
    print(tool_compute_returns())
    print("\n" + tool_sector_diversification())
    print("\n" + tool_optimize_markowitz_target(0.10))
    print("\n" + tool_sentiment_analysis("AAPL"))