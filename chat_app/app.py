"""
TimesFM Chat App — Flask backend
TimesFM 3.0 · dati reali Yahoo Finance · quantili completi
"""

import re
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
import numpy as np

app = Flask(__name__, static_folder="static")

# Singleton del modello — caricato una volta sola
_model_cache = None


def get_model():
    global _model_cache
    if _model_cache is None:
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[TimesFM] Caricamento modello su device={device}...")
        config = ModelConfig(
            checkpoint_path="google/timesfm-3.0-pytorch",
            per_core_batch_size=4,
            device=device,
        )
        _model_cache = TimesFM3Evaluator(config)
        print("[TimesFM] Modello pronto ✓")
    return _model_cache


# ---------------------------------------------------------------------------
# Mapping parole naturali -> ticker Yahoo Finance
# ---------------------------------------------------------------------------
TICKER_MAP = {
    # Crypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "cardano": "ADA-USD", "ada": "ADA-USD",
    "ripple": "XRP-USD", "xrp": "XRP-USD",
    "polkadot": "DOT-USD", "dot": "DOT-USD",
    "chainlink": "LINK-USD", "link": "LINK-USD",
    "avax": "AVAX-USD", "avalanche": "AVAX-USD",
    "bnb": "BNB-USD",
    # Indici
    "sp500": "^GSPC", "s&p": "^GSPC", "s&p500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI", "dow jones": "^DJI",
    "ftse": "^FTSE",
    "dax": "^GDAXI",
    "nikkei": "^N225",
    # Azioni tech
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "meta": "META", "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    # Valute
    "euro": "EURUSD=X", "eur/usd": "EURUSD=X", "eur usd": "EURUSD=X",
    "yen": "JPY=X",
    "sterlina": "GBPUSD=X", "pound": "GBPUSD=X",
    "franco": "CHFUSD=X",
    # Materie prime
    "oro": "GC=F", "gold": "GC=F",
    "argento": "SI=F", "silver": "SI=F",
    "petrolio": "CL=F", "oil": "CL=F", "wti": "CL=F",
    "gas": "NG=F",
    "rame": "HG=F", "copper": "HG=F",
    # Italiane
    "eni": "ENI.MI",
    "enel": "ENEL.MI",
    "ferrari": "RACE",
    "stellantis": "STLA",
    "intesa": "ISP.MI",
    "unicredit": "UCG.MI",
    "mediobanca": "MB.MI",
    "generali": "G.MI",
}

HORIZON_MAP = {
    "domani": 1, "tomorrow": 1,
    "dopodomani": 2,
    "settimana": 7, "week": 7,
    "mese": 30, "month": 30,
    "trimestre": 90,
    "semestre": 180,
    "anno": 365, "year": 365,
}

# Etichette quantili restituiti da TimesFM 3.0 (9 quantili)
QUANTILE_LABELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_horizon(text: str) -> int:
    text_lower = text.lower()
    m = re.search(r"(\d+)\s*(giorni?|days?|settimane?|weeks?|mesi?|months?)", text_lower)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if "settiman" in unit or "week" in unit:
            val *= 7
        elif "mes" in unit or "month" in unit:
            val *= 30
        return min(val, 60)
    for word, days in HORIZON_MAP.items():
        if word in text_lower:
            return days
    return 7


def parse_ticker(text: str):
    text_lower = text.lower()
    for keyword in sorted(TICKER_MAP.keys(), key=len, reverse=True):
        if keyword in text_lower:
            return TICKER_MAP[keyword], keyword
    return None, None


def get_historical_data(ticker: str, days_back: int = 730):
    end = datetime.today()
    start = end - timedelta(days=days_back)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return None, None, None
    closes = df["Close"].dropna().values.flatten().astype(np.float32)
    highs  = df["High"].dropna().values.flatten().astype(np.float32)
    lows   = df["Low"].dropna().values.flatten().astype(np.float32)
    dates  = [d.strftime("%Y-%m-%d") for d in df.index]
    return closes, dates, highs, lows


def run_timesfm(context: np.ndarray, horizon: int) -> dict:
    """Esegue TimesFM 3.0 e restituisce previsione + tutti i 9 quantili."""
    try:
        forecaster = get_model()
        outputs = list(forecaster.predict_batch(
            [context],
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=True,
        ))
        result = outputs[0]
        # result.forecast shape: (horizon,)
        # result.quantiles shape: (horizon, 9)
        quantiles_by_level = {}
        for i, q in enumerate(QUANTILE_LABELS):
            quantiles_by_level[str(q)] = result.quantiles[:, i].tolist()
        return {
            "forecast": result.forecast.tolist(),        # mediana
            "quantiles": quantiles_by_level,             # tutti i 9 livelli
        }
    except Exception as e:
        raise RuntimeError(f"Errore TimesFM 3.0: {e}\n{traceback.format_exc()}")


def generate_future_dates(last_date_str: str, n: int):
    """Date future (solo giorni lavorativi per asset finanziari)."""
    last = datetime.strptime(last_date_str, "%Y-%m-%d")
    dates = []
    d = last + timedelta(days=1)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def fmt(val: float, ticker: str) -> str:
    """Formatta il prezzo in base al tipo di asset."""
    if "BTC" in ticker:
        return f"${val:,.0f}"
    if "ETH" in ticker:
        return f"${val:,.1f}"
    if any(x in ticker for x in ["SOL", "BNB", "AVAX"]):
        return f"${val:,.2f}"
    if any(x in ticker for x in ["DOGE", "ADA", "XRP", "LINK", "DOT"]):
        return f"${val:.4f}"
    if "=X" in ticker:
        return f"{val:.4f}"
    if "=F" in ticker:
        return f"${val:.2f}"
    if ".MI" in ticker:
        return f"€{val:.2f}"
    return f"${val:.2f}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/warmup", methods=["POST"])
def warmup():
    """Pre-carica il modello in background al primo avvio."""
    try:
        get_model()
        return jsonify({"status": "ready"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Messaggio vuoto"}), 400

    # 1. Identifica asset
    ticker, keyword = parse_ticker(message)
    if not ticker:
        return jsonify({
            "type": "help",
            "text": (
                "Non ho capito su cosa vuoi la previsione 🤔\n\n"
                "Prova così:\n"
                "• *Bitcoin questa settimana*\n"
                "• *Tesla prossimi 30 giorni*\n"
                "• *Nvidia domani*\n"
                "• *Oro prossimo mese*\n"
                "• *Euro questa settimana*\n\n"
                "Supporto: Bitcoin, Ethereum, Solana, Tesla, Apple, Nvidia, "
                "S&P500, Oro, Petrolio, Ferrari, Unicredit e molti altri."
            )
        })

    # 2. Orizzonte
    horizon = parse_horizon(message)

    # 3. Dati storici
    try:
        result = get_historical_data(ticker, days_back=730)
        closes, dates, highs, lows = result if len(result) == 4 else (*result, None, None)
        if closes is None or len(closes) < 30:
            return jsonify({
                "type": "error",
                "text": f"Dati non disponibili per **{ticker}**. Riprova tra qualche secondo."
            })
    except Exception as e:
        return jsonify({"type": "error", "text": f"Errore scaricamento dati: {str(e)}"})

    # 4. TimesFM 3.0
    try:
        context = closes[-min(512, len(closes)):]
        pred = run_timesfm(context, horizon)
    except Exception as e:
        return jsonify({"type": "error", "text": str(e)})

    # 5. Statistiche previsione
    last_price   = float(closes[-1])
    forecast     = pred["forecast"]              # lista di horizon float
    q10          = pred["quantiles"]["0.1"]      # scenario pessimista
    q25          = pred["quantiles"]["0.2"]
    q50          = pred["quantiles"]["0.5"]      # mediana
    q75          = pred["quantiles"]["0.8"]
    q90          = pred["quantiles"]["0.9"]      # scenario ottimista

    forecast_end = forecast[-1]
    min_forecast = min(q10)
    max_forecast = max(q90)
    pct_median   = ((forecast_end - last_price) / last_price) * 100
    pct_min      = ((min_forecast - last_price) / last_price) * 100
    pct_max      = ((max_forecast - last_price) / last_price) * 100
    color        = "bullish" if pct_median > 0 else "bearish"
    direction    = "📈 salirà" if pct_median > 0 else "📉 scenderà"

    # Volatilità storica (std% ultimi 30gg)
    recent = closes[-30:].astype(float)
    returns = np.diff(recent) / recent[:-1]
    volatility = float(np.std(returns) * 100)

    # 6. Date future e storico per grafico
    future_dates = generate_future_dates(dates[-1], len(forecast))
    hist_n = min(30, len(closes))
    hist_prices  = closes[-hist_n:].tolist()
    hist_dates   = dates[-hist_n:]
    hist_highs   = highs[-hist_n:].tolist() if highs is not None else []
    hist_lows    = lows[-hist_n:].tolist()  if lows  is not None else []

    # 7. Tabella giornaliera per la previsione
    daily_table = []
    for i, (d, med, lo, hi) in enumerate(zip(future_dates, forecast, q10, q90)):
        pct = ((med - last_price) / last_price) * 100
        daily_table.append({
            "date": d,
            "median": round(med, 4),
            "min": round(lo, 4),
            "max": round(hi, 4),
            "pct": round(pct, 2),
            "fmt_median": fmt(med, ticker),
            "fmt_min": fmt(lo, ticker),
            "fmt_max": fmt(hi, ticker),
        })

    return jsonify({
        "type": "forecast",
        "ticker": ticker,
        "keyword": keyword,
        "horizon": horizon,
        "direction": direction,
        "color": color,
        # Prezzi chiave
        "last_price": last_price,
        "forecast_end": forecast_end,
        "min_forecast": min_forecast,
        "max_forecast": max_forecast,
        # Variazioni %
        "pct_median": round(pct_median, 2),
        "pct_min": round(pct_min, 2),
        "pct_max": round(pct_max, 2),
        # Volatilità
        "volatility": round(volatility, 2),
        # Formattati
        "fmt_current": fmt(last_price, ticker),
        "fmt_median":  fmt(forecast_end, ticker),
        "fmt_min":     fmt(min_forecast, ticker),
        "fmt_max":     fmt(max_forecast, ticker),
        # Serie per grafico
        "hist_prices": hist_prices,
        "hist_dates":  hist_dates,
        "hist_highs":  hist_highs,
        "hist_lows":   hist_lows,
        "forecast":    forecast,
        "q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90,
        "future_dates": future_dates,
        # Tabella giorni
        "daily_table": daily_table,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5050)
