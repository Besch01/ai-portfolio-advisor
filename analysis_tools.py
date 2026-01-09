import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os
import warnings

# Importiamo SOLO le funzioni necessarie da db_tools
# get_current_portfolio restituisce già una lista di dict, perfetta per Pandas
from db_tools import get_current_portfolio, get_sector_allocation
from api_tools import get_latest_close_prices, get_historical_prices

from newsapi import NewsApiClient
from textblob import TextBlob

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# 1. COMPUTE RETURNS
def tool_compute_returns():
    """
    Calcola il ROI del portafoglio incrociando i dati del DB con i prezzi API.
    """
    # Recupera i dati puliti dal DB
    portfolio = get_current_portfolio()
    
    if not portfolio: 
        return {}

    tickers = [p['ticker'] for p in portfolio]
    current_prices = get_latest_close_prices(tickers)
      
    total_purchase_cost = 0.0
    total_current_value = 0.0

    for p in portfolio:
        ticker = p['ticker']
        # Conversione in float per sicurezza matematica
        qty = float(p['total_quantity'])
        avg_price = float(p['avg_price'])
        
        # Prezzo corrente o fallback sul prezzo medio se API fallisce
        current_price = current_prices.get(ticker, avg_price)
        
        if current_price:
            total_purchase_cost += qty * avg_price
            total_current_value += qty * current_price
    
    if total_purchase_cost == 0: 
        return {"error": "Il costo totale di acquisto è zero."}
    
    roi = ((total_current_value - total_purchase_cost) / total_purchase_cost) * 100
    
    return {
        "roi_percentage": round(roi, 2),
        "total_current_value": round(total_current_value, 2),
        "total_invested_cost": round(total_purchase_cost, 2),
        "currency": "USD"
    }

# 2. BEST RETURNS
def get_best_returns_data():
    """
    Restituisce una lista di asset ordinati per performance.
    """
    portfolio_rows = get_current_portfolio()

    if not portfolio_rows:
        return []

    tickers = [row['ticker'] for row in portfolio_rows]
    current_prices = get_latest_close_prices(tickers)

    results = []

    for row in portfolio_rows:
        ticker = row['ticker']
        quantity = float(row['total_quantity'])
        avg_price = float(row['avg_price'])
        
        current_price = current_prices.get(ticker, avg_price)
        if current_price is None: current_price = avg_price
        current_price = float(current_price)

        invested_value = quantity * avg_price
        current_value = quantity * current_price
        profit_loss = current_value - invested_value
        
        return_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0.0

        results.append({
            "ticker": ticker,
            "name": row['name'],
            "sector": row['sector'],
            "quantity": int(quantity),
            "avg_purchase_price": round(avg_price, 2),
            "current_market_price": round(current_price, 2),
            "return_percentage": round(return_pct, 2),
            "profit_loss_usd": round(profit_loss, 2)
        })

    # Ordina dal migliore al peggiore
    results.sort(key=lambda x: x['return_percentage'], reverse=True)
    return results

# 3. SECTOR DIVERSIFICATION
def tool_sector_diversification_comparison():
    """
    Confronta allocazione iniziale (Costo) vs Attuale (Mercato).
    Restituisce un DataFrame Pandas.
    """
    # 1. Allocazione Iniziale (dal DB - usa invested_value)
    initial_allocation = get_sector_allocation()
    
    # 2. Allocazione Corrente (dal DB + API - Valore Mercato)
    # Riusiamo get_current_portfolio perché contiene già ticker, sector e quantity!
    portfolio_data = get_current_portfolio()

    if not portfolio_data: 
        return pd.DataFrame()

    initial_dict = {sector: perc for sector, perc in initial_allocation}
    
    # Convertiamo la lista di dict in DataFrame
    df_portfolio = pd.DataFrame(portfolio_data)
    
    tickers = df_portfolio['ticker'].tolist()
    current_prices = get_latest_close_prices(tickers)
    
    # Mappiamo i prezzi e calcoliamo il valore attuale
    df_portfolio['current_price'] = df_portfolio['ticker'].map(current_prices).fillna(0)
    df_portfolio['current_value'] = df_portfolio['total_quantity'] * df_portfolio['current_price']

    # Raggruppamento per settore
    current_sector_values = df_portfolio.groupby('sector')['current_value'].sum()
    total_market_value = current_sector_values.sum()
    
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

    return pd.DataFrame(results_data)

# 4. MARKOWITZ OPTIMIZATION
def tool_optimize_markowitz_target(target_return_annualized=0.10):
    rows = get_current_portfolio()
    tickers = [row['ticker'] for row in rows]

    if len(tickers) < 2: 
        return {"error": "Servono almeno 2 asset per l'ottimizzazione."}

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = get_historical_prices(tickers, start_date, end_date)
    if data.empty: return {"error": "Dati storici non disponibili."}
    
    daily_returns = data.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252 

    max_possible = expected_returns.max()
    if target_return_annualized > max_possible:
        return {"error": "Target troppo alto", "max_possible_return": round(max_possible, 4)}

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
        return {"error": "Ottimizzazione matematica fallita."}

    if not optimized.success: return {"error": "Il solver non ha trovato soluzioni."}

    volatility = np.sqrt(optimized.fun)
    weights_dict = {tickers[i]: round(optimized.x[i], 4) for i in range(len(tickers)) if optimized.x[i] > 0.001}

    return {
        "target_return": target_return_annualized,
        "estimated_volatility": round(volatility, 4),
        "optimized_weights": weights_dict
    }

# 5. SENTIMENT ANALYSIS
def tool_sentiment_analysis(ticker=None):
    # Se non c'è ticker, prendiamo il più grande nel portafoglio
    if not ticker:
        portfolio = get_current_portfolio()
        if portfolio:
            # Ordiniamo la lista in Python
            top_asset = sorted(portfolio, key=lambda x: x['total_quantity'], reverse=True)[0]
            ticker = top_asset['ticker']
        else:
            return {"error": "Nessun ticker trovato o portafoglio vuoto."}

    if not NEWS_API_KEY:
        return {"error": "API Key mancante. Esegui: export NEWS_API_KEY='tua_chiave' nel terminale."}

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
            articles_data.append({"title": title, "score": round(score, 2), "source": art['source']['name']})

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
        return {"error": f"Errore API News: {str(e)}"}