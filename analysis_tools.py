import sqlite3
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
from newsapi import NewsApiClient
from textblob import TextBlob

# Import local tools
from db_tools import get_connection, get_sector_allocation
from api_tools import get_latest_prices, get_historical_prices

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE ---
NEWS_API_KEY = 'f478f541562347d38b316ef6a2d19cac' 


# 1. COMPUTE RETURNS (Restituisce: dict)
def tool_compute_returns():
    """
    Calcola il ROI del portafoglio.
    Returns: dict {roi_pct, total_value, total_cost, currency}
    """
    conn = get_connection()
    try:
        portfolio = conn.execute("SELECT * FROM current_portfolio WHERE total_quantity > 0").fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return {} # Ritorna dizionario vuoto in caso di errore
    conn.close()
    
    if not portfolio: 
        return {}

    tickers = [p['ticker'] for p in portfolio]
    current_prices = get_latest_prices(tickers)
      
    total_purchase_cost = 0.0
    total_current_value = 0.0

    for p in portfolio:
        ticker = p['ticker']
        qty = float(p['total_quantity'])
        avg_price = float(p['avg_price'])
        
        # Prezzo corrente o fallback sul prezzo medio
        current_price = current_prices.get(ticker, avg_price)
        
        if current_price:
            total_purchase_cost += qty * avg_price
            total_current_value += qty * current_price
    
    if total_purchase_cost == 0: 
        return {"error": "Total cost is zero"}
    
    roi = ((total_current_value - total_purchase_cost) / total_purchase_cost) * 100
    
    # Restituisce solo dati numerici puri
    return {
        "roi_percentage": round(roi, 2),
        "total_current_value": round(total_current_value, 2),
        "total_invested_cost": round(total_purchase_cost, 2),
        "currency": "USD"
    }

# 2. RETURNS ANALYZER (Restituisce: list of dict)

from api_tools import get_latest_prices

def get_best_returns_data():
    """
    Calcola il rendimento reale per ogni titolo nel portafoglio.
    Incrocia i dati storici (db_tools) con i prezzi attuali (api_tools).

    Returns:
        list[dict]: Una lista di dizionari ordinata per rendimento decrescente.
        Ogni dizionario contiene: ticker, name, sector, quantity, avg_price, 
        current_price, return_pct, profit_loss.
    """
    # 1. Recupera lo stato attuale del portafoglio dal Database
    conn = get_connection()
    try:
        # Usiamo la vista 'current_portfolio' che ha già avg_price calcolato
        query = "SELECT ticker, name, sector, total_quantity, avg_price FROM current_portfolio WHERE total_quantity > 0"
        portfolio_rows = conn.execute(query).fetchall()
    except Exception as e:
        conn.close()
        return [{"error": f"Errore DB: {str(e)}"}]
    conn.close()

    if not portfolio_rows:
        return []

    # 2. Ottieni i prezzi attuali (Batch Request ottimizzata)
    # Invece di fare una chiamata per ogni titolo, le facciamo tutte insieme
    tickers = [row['ticker'] for row in portfolio_rows]
    current_prices = get_latest_prices(tickers)

    results = []

    # 3. Calcola i rendimenti per ogni titolo
    for row in portfolio_rows:
        ticker = row['ticker']
        
        # Casting esplicito per garantire i tipi di dato richiesti
        quantity = float(row['total_quantity'])
        avg_price = float(row['avg_price'])
        
        # Recupera prezzo corrente (o usa avg_price se l'API fallisce per evitare crash)
        current_price = current_prices.get(ticker)
        
        if current_price is None:
            current_price = avg_price # Fallback neutro
        
        current_price = float(current_price)

        # Calcolo matematico
        invested_value = quantity * avg_price
        current_value = quantity * current_price
        
        profit_loss = current_value - invested_value
        
        # Evitiamo divisioni per zero
        if avg_price > 0:
            return_pct = ((current_price - avg_price) / avg_price) * 100
        else:
            return_pct = 0.0

        # Creazione del dizionario dati pulito
        entry = {
            "ticker": ticker,
            "name": row['name'],
            "sector": row['sector'],
            "quantity": int(quantity),             # int
            "avg_purchase_price": round(avg_price, 2), # float
            "current_market_price": round(current_price, 2), # float
            "return_percentage": round(return_pct, 2), # float
            "profit_loss_usd": round(profit_loss, 2)   # float
        }
        
        results.append(entry)

    # 4. Ordina per rendimento percentuale decrescente (dal migliore al peggiore)
    results.sort(key=lambda x: x['return_percentage'], reverse=True)

    return results


# 3. SECTOR DIVERSIFICATION (Restituisce: pandas.DataFrame)
def tool_sector_diversification_comparison():
    """
    Confronta allocazione iniziale vs attuale.
    Returns: pandas.DataFrame
    """
    conn = get_connection()
    
    try:
        initial_allocation = get_sector_allocation(conn) # Restituisce lista di tuple
    except Exception:
        conn.close()
        return pd.DataFrame() # DataFrame vuoto in caso di errore
        
    initial_dict = {sector: perc for sector, perc in initial_allocation}

    query = """
    SELECT ticker, sector, SUM(quantity) as total_qty 
    FROM transactions GROUP BY ticker, sector HAVING total_qty > 0
    """
    df_portfolio = pd.read_sql_query(query, conn)
    conn.close()

    if df_portfolio.empty: 
        return pd.DataFrame()

    tickers = df_portfolio['ticker'].tolist()
    current_prices = get_latest_prices(tickers)
    
    df_portfolio['current_price'] = df_portfolio['ticker'].map(current_prices).fillna(0)
    df_portfolio['current_value'] = df_portfolio['total_qty'] * df_portfolio['current_price']

    current_sector_values = df_portfolio.groupby('sector')['current_value'].sum()
    total_market_value = current_sector_values.sum()
    
    # Costruiamo i dati per il DataFrame finale
    results_data = []
    all_sectors = set(initial_dict.keys()).union(set(current_sector_values.index))

    for sector in sorted(all_sectors):
        init_p = float(initial_dict.get(sector, 0.0))
        curr_val = float(current_sector_values.get(sector, 0.0))
        
        curr_p = (curr_val / total_market_value * 100) if total_market_value > 0 else 0.0
        drift = curr_p - init_p
        
        results_data.append({
            "sector": sector,
            "initial_weight_pct": round(init_p, 2),
            "current_weight_pct": round(curr_p, 2),
            "drift_pct": round(drift, 2)
        })

    # Restituisce un DataFrame pulito
    return pd.DataFrame(results_data)

# 4. MARKOWITZ OPTIMIZATION (Restituisce: dict)
def tool_optimize_markowitz_target(target_return_annualized=0.10):
    """
    Ottimizzazione portafoglio.
    Returns: dict {volatility, weights, status}
    """
    conn = get_connection()
    try:
        rows = conn.execute("SELECT ticker FROM current_portfolio WHERE total_quantity > 0").fetchall()
        tickers = [row['ticker'] for row in rows]
    except Exception:
        conn.close()
        return {"error": "Database error"}
    conn.close()

    if len(tickers) < 2: 
        return {"error": "Not enough assets"}

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = get_historical_prices(tickers, start_date, end_date)
    if data.empty:
        return {"error": "No historical data"}
    
    daily_returns = data.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252 

    max_possible = expected_returns.max()
    if target_return_annualized > max_possible:
        return {
            "error": "Target too high", 
            "max_possible_return": round(max_possible, 4)
        }

    # Ottimizzazione matematica
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return_annualized}
    ]
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = [1/len(tickers)] * len(tickers)

    try:
        optimized = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    except Exception:
        return {"error": "Optimization failed"}

    if not optimized.success:
        return {"error": "Solver failed"}

    volatility = np.sqrt(optimized.fun)
    
    # Creazione dizionario pesi (filtro < 0.1%)
    weights_dict = {}
    for i, ticker in enumerate(tickers):
        w = optimized.x[i]
        if w > 0.001: 
            weights_dict[ticker] = round(w, 4)

    return {
        "target_return": target_return_annualized,
        "estimated_volatility": round(volatility, 4),
        "optimized_weights": weights_dict
    }

# 5. SENTIMENT ANALYSIS (Restituisce: dict)
def tool_sentiment_analysis(ticker=None):
    """
    Analisi sentiment.
    Returns: dict {ticker, score, label, articles_list}
    """
    if not ticker:
        conn = get_connection()
        try:
            row = conn.execute("SELECT ticker FROM current_portfolio ORDER BY total_quantity DESC LIMIT 1").fetchone()
            ticker = row['ticker'] if row else None
        except Exception:
            return {"error": "Db error"}
        conn.close()
        
    if not ticker:
        return {"error": "No ticker found"}

    if not NEWS_API_KEY or 'YOUR_API_KEY' in NEWS_API_KEY:
        return {"error": "Invalid API Key"}

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        headlines = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        articles = headlines.get('articles', [])
        
        if not articles:
            return {"ticker": ticker, "score": 0, "label": "No Data", "articles": []}

        scores = []
        articles_data = []
        
        for art in articles:
            title = art['title']
            score = TextBlob(title).sentiment.polarity
            scores.append(score)
            
            articles_data.append({
                "title": title,
                "score": round(score, 2),
                "source": art['source']['name']
            })

        avg_score = sum(scores) / len(scores)
        label = "BULLISH" if avg_score > 0.1 else "BEARISH" if avg_score < -0.1 else "NEUTRAL"

        return {
            "ticker": ticker,
            "average_score": round(avg_score, 2),
            "sentiment_label": label,
            "article_count": len(articles),
            "articles": articles_data
        }

    except Exception as e:
        return {"error": str(e)}

# AGENT TOOLS DICTIONARY
agent_tools = {
    "returns_analyzer": tool_compute_returns,
    "best_returns_data": get_best_returns_data,
    "diversification_expert": tool_sector_diversification_comparison,
    "portfolio_optimizer": tool_optimize_markowitz_target,
    "market_sentiment": tool_sentiment_analysis
}

if __name__ == "__main__":
    # Test rapido per verificare i tipi di dato restituiti
    print("--- 1. Returns (Dict) ---")
    print(tool_compute_returns())

    print("\n--- 2. Best Returns (Dict) ---")
    print(get_best_returns_data())

    print("\n--- 3. Diversification (DataFrame) ---")
    df = tool_sector_diversification_comparison()
    print(df) # Stampa il dataframe puro
    # print(df.to_dict(orient='records')) # Se vuoi vederlo come lista di dizionari
    
    print("\n--- 4. Markowitz (Dict) ---")
    print(tool_optimize_markowitz_target(0.10))
    
    print("\n--- 5. Sentiment (Dict) ---")
    print(tool_sentiment_analysis())