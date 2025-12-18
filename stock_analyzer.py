import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def safe_number(value):
    """Converte in float ignorando NaN/inf e valori non numerici"""
    try:
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


# ================== CACHING SYSTEM ==================
class DataCache:
    """Sistema di caching per evitare richieste API ripetute"""

    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, ticker, data_type):
        return self.cache_dir / f"{ticker}_{data_type}.json"

    def is_cache_valid(self, cache_path, max_age_minutes=30):
        """Verifica se la cache è ancora valida"""
        if not cache_path.exists():
            return False

        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time
        return age.total_seconds() < (max_age_minutes * 60)

    def load(self, ticker, data_type, max_age_minutes=30):
        """Carica dati dalla cache se validi"""
        cache_path = self.get_cache_path(ticker, data_type)

        if self.is_cache_valid(cache_path, max_age_minutes):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logging.info(f"✓ Cache hit per {ticker} - {data_type}")
                return data
            except json.JSONDecodeError as e:
                logging.warning(
                    f"Cache corrotta per {ticker}-{data_type}: {e}, la elimino"
                )
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as e:
                logging.warning(f"Errore lettura cache: {e}")

        return None

    def save(self, ticker, data_type, data):
        """Salva dati in cache"""
        cache_path = self.get_cache_path(ticker, data_type)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
            logging.info(f"✓ Cache salvata per {ticker} - {data_type}")
        except Exception as e:
            logging.warning(f"Errore salvataggio cache: {e}")


# Istanza globale cache
cache = DataCache()


# ================== STORICO ANALISI ==================
class TickerHistory:
    """Gestisce lo storico dei punteggi e delle metriche fondamentali"""

    def __init__(
        self, history_path=".cache/ticker_history.json", max_entries=200, window_size=30
    ):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(exist_ok=True)
        self.max_entries = max_entries
        self.window_size = window_size

    def _load(self):
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Errore lettura storico: {e}")
        return {}

    def _save(self, data):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True, indent=2)
        except Exception as e:
            logging.warning(f"Errore salvataggio storico: {e}")

    def _median_for_key(self, entries, key):
        values = [
            entry[key]
            for entry in entries
            if key in entry and isinstance(entry[key], (int, float))
        ]
        if not values:
            return None
        return float(np.median(values))

    def record_snapshot(self, ticker, snapshot):
        data = self._load()
        entries = data.get(ticker, [])
        entries.append(snapshot)
        data[ticker] = entries[-self.max_entries :]
        self._save(data)
        return data[ticker]

    def summarize(self, ticker, entries=None):
        if entries is None:
            entries = self._load().get(ticker, [])
        if not entries:
            return {}

        window_entries = entries[-self.window_size :]

        median_score = self._median_for_key(window_entries, "buy_score")
        median_price = self._median_for_key(window_entries, "price")
        median_pe = self._median_for_key(window_entries, "pe_ratio")
        median_rev = self._median_for_key(window_entries, "revenue_growth")
        median_rr = self._median_for_key(window_entries, "risk_reward")

        latest = entries[-1]

        def pct_diff(current, reference):
            if reference in (None, 0) or current is None:
                return None
            return ((current - reference) / reference) * 100

        return {
            "observations": len(entries),
            "window_used": len(window_entries),
            "median_buy_score": median_score,
            "median_price": median_price,
            "median_pe": median_pe,
            "median_revenue_growth": median_rev,
            "median_risk_reward": median_rr,
            "score_vs_median_pct": pct_diff(latest.get("buy_score"), median_score),
            "price_vs_median_pct": pct_diff(latest.get("price"), median_price),
            "risk_reward_vs_median_pct": pct_diff(latest.get("risk_reward"), median_rr),
        }

    def record_and_summarize(self, ticker, snapshot):
        entries = self.record_snapshot(ticker, snapshot)
        return self.summarize(ticker, entries)


# Istanza globale storico
history_tracker = TickerHistory()


# ================== VALIDATION ==================
def validate_ticker(ticker):
    """Valida il ticker e verifica esistenza"""
    if not isinstance(ticker, str):
        raise ValueError("Ticker deve essere una stringa")

    ticker = ticker.upper().strip()

    if len(ticker) < 1 or len(ticker) > 10:
        raise ValueError("Ticker deve essere tra 1 e 10 caratteri")

    # Verifica esistenza ticker
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Se non ha dati fondamentali, probabilmente non esiste
        if not info or "symbol" not in info:
            raise ValueError(f"Ticker {ticker} non trovato o non valido")

        logging.info(f"✓ Ticker validato: {ticker}")
        return ticker
    except Exception as e:
        raise ValueError(f"Errore validazione ticker {ticker}: {e}")


# ================== MARKET HOURS ==================
def is_market_open():
    """Verifica se il mercato è aperto (NYSE/NASDAQ)"""
    now = datetime.now()

    # Weekend
    if now.weekday() >= 5:  # Sabato=5, Domenica=6
        return False, "Mercato chiuso (weekend)"

    # Orari mercato USA: 9:30 - 16:00 EST (15:30 - 22:00 CET)
    market_open = now.replace(hour=15, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=22, minute=0, second=0, microsecond=0)

    if market_open <= now <= market_close:
        return True, "Mercato aperto"
    else:
        return False, f"Mercato chiuso (orario: {now.strftime('%H:%M')})"


# ================== DATA FETCHING CON CACHE ==================
def fetch_historical_data(ticker, years=5):
    """Fetch dati storici con caching"""
    # Controlla cache (valida per 1 giorno per dati storici)
    cached_data = cache.load(ticker, "historical", max_age_minutes=1440)
    if cached_data:
        df = pd.DataFrame(cached_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{years}y")

        if len(df) < 50:
            raise ValueError("Dati storici insufficienti o non disponibili")

        df = df.rename(
            columns={
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
            }
        )
        df = df.reset_index()
        df = df.rename(columns={"Date": "timestamp"})
        df["timestamp"] = df["timestamp"].astype(str)

        # Salva in cache
        cache.save(ticker, "historical", df.to_dict("records"))

        logging.info(f"✓ Dati storici fetchati per {ticker}: {len(df)} righe")
        return df
    except Exception as e:
        logging.error(f"Errore in fetch_historical_data per {ticker}: {e}")
        raise


def fetch_current_snapshot(ticker):
    """Fetch snapshot con cache (5 minuti per dati real-time)"""
    cached_data = cache.load(ticker, "snapshot", max_age_minutes=5)
    if cached_data:
        return cached_data["price"]

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        if data.empty:
            raise ValueError("Nessun snapshot disponibile")

        current_price = float(data["Close"].iloc[-1])

        cache.save(ticker, "snapshot", {"price": current_price})

        logging.info(f"✓ Snapshot fetchato per {ticker}: ${current_price:.2f}")
        return current_price
    except Exception as e:
        logging.error(f"Errore in fetch_current_snapshot per {ticker}: {e}")
        raise


def fetch_financials(ticker):
    """Fetch financials con cache (1 giorno)"""
    cached_data = cache.load(ticker, "financials", max_age_minutes=1440)
    if cached_data:
        return cached_data["eps"], cached_data["revenue_growth"]

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        eps = info.get("trailingEps", 0)
        revenue_growth = info.get("revenueGrowth", None)

        if revenue_growth is not None:
            revenue_growth_yoy = revenue_growth * 100
        else:
            revenue_growth_yoy = "N/A"

        cache.save(
            ticker, "financials", {"eps": eps, "revenue_growth": revenue_growth_yoy}
        )

        logging.info(f"✓ Financials fetchati per {ticker}")
        return eps, revenue_growth_yoy
    except Exception as e:
        logging.error(f"Errore in fetch_financials per {ticker}: {e}")
        return 0, "N/A"


# ================== TECHNICAL INDICATORS ==================
def calculate_technical_indicators(df):
    """Calcola indicatori tecnici"""
    try:
        if len(df) < 50:
            raise ValueError("Dati insufficienti per indicatori tecnici")

        df["SMA50"] = df["close"].rolling(window=50).mean()
        df["SMA200"] = df["close"].rolling(window=200).mean()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        macd_status = (
            "Bullish" if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1] else "Bearish"
        )

        df["BB_Mid"] = df["close"].rolling(window=20).mean()
        df["BB_Std"] = df["close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Mid"] + (df["BB_Std"] * 2)
        df["BB_Lower"] = df["BB_Mid"] - (df["BB_Std"] * 2)
        bb_status = (
            "Overbought"
            if df["close"].iloc[-1] > df["BB_Upper"].iloc[-1]
            else "Oversold"
            if df["close"].iloc[-1] < df["BB_Lower"].iloc[-1]
            else "Neutral"
        )

        latest = df.iloc[-1]
        trend = "Up" if latest["close"] > latest["SMA50"] else "Down"
        rsi_status = (
            "Oversold"
            if latest["RSI"] < 30
            else "Overbought"
            if latest["RSI"] > 70
            else "Neutral"
        )

        logging.info("✓ Indicatori tecnici calcolati")
        return trend, rsi_status, macd_status, bb_status, df
    except Exception as e:
        logging.error(f"Errore in calculate_technical_indicators: {e}")
        raise


# ================== IMPROVED BUY SCORE ==================
def calculate_advanced_buy_score(
    df,
    current_price,
    eps,
    revenue_growth_yoy,
    trend,
    rsi_status,
    macd_status,
    bb_status,
):
    """
    Sistema di scoring migliorato con pesi dinamici e conferme multiple
    Score 0-100: considera contesto e conferme incrociate
    """
    score = 0
    reasons = []
    warnings = []

    # ============ 1. ANALISI TECNICA AVANZATA (45 punti) ============

    # RSI con contesto (12 punti)
    latest_rsi = df["RSI"].iloc[-1]
    rsi_prev = df["RSI"].iloc[-5]  # RSI 5 giorni fa

    if latest_rsi < 30:
        # Oversold MA con divergenza positiva?
        if latest_rsi > rsi_prev:
            score += 12
            reasons.append(
                f"✅ RSI oversold ({latest_rsi:.1f}) con divergenza positiva - forte segnale rialzista"
            )
        else:
            score += 6
            reasons.append(
                f"⚠️ RSI oversold ({latest_rsi:.1f}) ma ancora in calo - possibile ulteriore discesa"
            )
            warnings.append("RSI in oversold può continuare a scendere")
    elif 30 <= latest_rsi < 50:
        score += 9
        reasons.append(f"✅ RSI ({latest_rsi:.1f}) in zona favorevole per acquisto")
    elif 50 <= latest_rsi < 70:
        score += 6
        reasons.append(f"➖ RSI ({latest_rsi:.1f}) neutrale")
    else:
        score += 0
        reasons.append(
            f"❌ RSI overbought ({latest_rsi:.1f}) - alto rischio correzione"
        )
        warnings.append("RSI in territorio di ipercomprato - prudenza")

    # MACD con conferma trend (12 punti)
    macd_diff = df["MACD"].iloc[-1] - df["MACD_Signal"].iloc[-1]
    macd_prev_diff = df["MACD"].iloc[-2] - df["MACD_Signal"].iloc[-2]

    if macd_status == "Bullish":
        if macd_diff > macd_prev_diff:  # MACD in accelerazione
            score += 12
            reasons.append(f"✅ MACD bullish in accelerazione - forte momentum")
        else:
            score += 8
            reasons.append(f"✅ MACD bullish ma in decelerazione")
    else:
        if macd_diff < macd_prev_diff:  # MACD bearish in accelerazione
            score += 0
            warnings.append("MACD bearish e in deterioramento")
        else:
            score += 3
            reasons.append(f"⚠️ MACD bearish ma sta rallentando la discesa")

    # Bollinger Bands con volatilità (10 punti)
    bb_width = (df["BB_Upper"].iloc[-1] - df["BB_Lower"].iloc[-1]) / df["BB_Mid"].iloc[
        -1
    ]

    if bb_status == "Oversold":
        score += 10
        reasons.append(f"✅ Prezzo sotto banda inferiore - probabile rimbalzo tecnico")
    elif bb_status == "Neutral":
        if bb_width < 0.10:  # Bollinger squeeze
            score += 7
            reasons.append(f"✅ Bollinger squeeze - breakout imminente")
        else:
            score += 5
            reasons.append(f"➖ Prezzo in range normale")
    else:
        score += 0
        warnings.append("Prezzo sopra banda superiore - possibile pullback")

    # Trend con Golden/Death Cross (11 punti)
    if len(df) >= 200:
        sma50 = df["SMA50"].iloc[-1]
        sma200 = df["SMA200"].iloc[-1]

        if sma50 > sma200 and trend == "Up":
            score += 11
            reasons.append(f"✅ Golden Cross attivo - forte trend rialzista")
        elif sma50 < sma200 and trend == "Down":
            score += 0
            reasons.append(f"❌ Death Cross attivo - trend ribassista")
            warnings.append("Segnale di trend ribassista di lungo periodo")
        else:
            score += 6
            reasons.append(f"➖ Trend misto SMA50 vs SMA200")
    else:
        if trend == "Up":
            score += 7
            reasons.append(f"✅ Trend rialzista su SMA50")
        else:
            score += 0
            reasons.append(f"❌ Trend ribassista su SMA50")

    # ============ 2. ANALISI FONDAMENTALE (30 punti) ============

    # P/E Ratio contestualizzato (15 punti)
    pe_ratio = current_price / eps if eps > 0 else None

    if pe_ratio:
        if pe_ratio < 10:
            score += 15
            reasons.append(f"✅ P/E molto basso ({pe_ratio:.1f}) - potenziale value")
        elif pe_ratio < 15:
            score += 13
            reasons.append(f"✅ P/E attraente ({pe_ratio:.1f})")
        elif pe_ratio < 25:
            score += 9
            reasons.append(f"➖ P/E nella norma ({pe_ratio:.1f})")
        elif pe_ratio < 40:
            score += 4
            reasons.append(f"⚠️ P/E elevato ({pe_ratio:.1f})")
        else:
            score += 0
            reasons.append(
                f"❌ P/E molto alto ({pe_ratio:.1f}) - possibile sopravvalutazione"
            )
            warnings.append(f"P/E {pe_ratio:.1f}x indica valutazione molto elevata")
    else:
        if eps < 0:
            score += 0
            warnings.append("Azienda in perdita (EPS negativo)")
        else:
            score += 5
            reasons.append("⚠️ P/E non disponibile")

    # Crescita Revenue con qualità (15 punti)
    if revenue_growth_yoy != "N/A":
        if revenue_growth_yoy > 30:
            score += 15
            reasons.append(
                f"✅ Crescita revenue eccezionale ({revenue_growth_yoy:.1f}%)"
            )
        elif revenue_growth_yoy > 15:
            score += 12
            reasons.append(f"✅ Crescita revenue forte ({revenue_growth_yoy:.1f}%)")
        elif revenue_growth_yoy > 5:
            score += 8
            reasons.append(f"✅ Crescita revenue moderata ({revenue_growth_yoy:.1f}%)")
        elif revenue_growth_yoy > 0:
            score += 4
            reasons.append(f"➖ Crescita revenue debole ({revenue_growth_yoy:.1f}%)")
        else:
            score += 0
            reasons.append(f"❌ Revenue in calo ({revenue_growth_yoy:.1f}%)")
            warnings.append("Crescita negativa dei ricavi")
    else:
        score += 5
        reasons.append("⚠️ Dati crescita revenue non disponibili")

    # ============ 3. VOLATILITÀ E RISCHIO (15 punti) ============

    returns = df["close"].pct_change()
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe_approx = (returns.mean() * 252) / (returns.std() * np.sqrt(252))

    # Volatilità con Sharpe ratio (10 punti)
    if volatility < 20:
        score += 7
        reasons.append(f"✅ Bassa volatilità ({volatility:.1f}%) - rischio contenuto")
    elif volatility < 35:
        score += 5
        reasons.append(f"➖ Volatilità media ({volatility:.1f}%)")
    elif volatility < 50:
        score += 2
        reasons.append(f"⚠️ Alta volatilità ({volatility:.1f}%)")
    else:
        score += 0
        reasons.append(f"❌ Volatilità estrema ({volatility:.1f}%) - alto rischio")
        warnings.append("Volatilità molto elevata - solo per profili aggressivi")

    if sharpe_approx > 1:
        score += 3
        reasons.append(f"✅ Sharpe ratio positivo ({sharpe_approx:.2f})")

    # Supporto/Resistenza con volume (5 punti)
    sma50 = df["SMA50"].iloc[-1]
    distance_from_sma = ((current_price - sma50) / sma50) * 100

    if -2 < distance_from_sma < 2:
        score += 5
        reasons.append(f"✅ Sul livello chiave SMA50 - possibile rimbalzo")
    elif -10 < distance_from_sma < -2:
        score += 4
        reasons.append(f"✅ Sotto SMA50 ({distance_from_sma:.1f}%) - oversold tecnico")
    elif distance_from_sma > 10:
        score += 0
        warnings.append(f"Prezzo {distance_from_sma:.1f}% sopra SMA50 - esteso")
    else:
        score += 2

    # ============ 4. MOMENTUM E LIQUIDITÀ (10 punti) ============

    avg_volume_20 = df["volume"].tail(20).mean()
    recent_volume = df["volume"].tail(5).mean()
    volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1

    price_change_5d = (
        (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6]
    ) * 100

    if volume_ratio > 1.5 and price_change_5d > 0:
        score += 10
        reasons.append(
            f"✅ Volume in espansione ({volume_ratio:.1f}x) con prezzo in rialzo - accumulazione"
        )
    elif volume_ratio > 1.2:
        score += 6
        reasons.append(f"✅ Volume sopra media ({volume_ratio:.1f}x)")
    elif volume_ratio > 0.8:
        score += 4
        reasons.append(f"➖ Volume nella norma")
    else:
        score += 0
        reasons.append(f"⚠️ Volume basso ({volume_ratio:.1f}x) - scarso interesse")

    return score, reasons, warnings


def get_recommendation(score, warnings):
    """Converte score in raccomandazione con contesto"""
    if score >= 80:
        return (
            "🟢 STRONG BUY",
            "Eccellente opportunità. Indicatori tecnici e fondamentali allineati.",
        )
    elif score >= 65:
        return "🟢 BUY", "Buon momento per acquistare. Prevalenza di segnali positivi."
    elif score >= 50:
        return (
            "🟡 MODERATE BUY",
            "Situazione favorevole ma con qualche riserva. Valuta il tuo profilo di rischio.",
        )
    elif score >= 35:
        return "🟠 HOLD", "Segnali misti. Meglio attendere maggiore chiarezza."
    elif score >= 20:
        return (
            "🔴 WEAK SELL",
            "Prevalenza di segnali negativi. Considera di ridurre l'esposizione.",
        )
    else:
        return "🔴 STRONG SELL", "Segnali molto negativi. Evita o esci dalla posizione."


# ================== PROJECTIONS CON DISCLAIMER ==================
def make_projections_with_disclaimer(df):
    """
    Proiezioni conservative con disclaimer obbligatorio
    Usa media mobile invece di regressione lineare
    """
    try:
        if len(df) < 60:
            return {
                "disclaimer": "⚠️ ATTENZIONE: Dati insufficienti per proiezioni affidabili",
                "1_anno": "N/A",
                "2_anni": "N/A",
                "3_anni": "N/A",
            }

        # Usa crescita media storica invece di regressione lineare
        returns_1y = (
            df["close"].pct_change().tail(252).mean()
            if len(df) >= 252
            else df["close"].pct_change().tail(len(df) // 2).mean()
        )
        returns_2y = (
            df["close"].pct_change().tail(504).mean() if len(df) >= 504 else returns_1y
        )
        returns_3y = df["close"].pct_change().mean()

        current_price = df["close"].iloc[-1]

        # Proiezioni basate su return medio giornaliero
        proj_1y = current_price * (1 + returns_1y) ** 252
        proj_2y = current_price * (1 + returns_2y) ** 504
        proj_3y = current_price * (1 + returns_3y) ** 756

        disclaimer = (
            "⚠️ DISCLAIMER: Le proiezioni sono PURAMENTE INDICATIVE basate su trend storici. "
            "I mercati sono imprevedibili. NON usare come unica base per decisioni di investimento. "
            "Performance passate NON garantiscono risultati futuri."
        )

        return {
            "disclaimer": disclaimer,
            "1_anno": f"${proj_1y:.2f}",
            "2_anni": f"${proj_2y:.2f}",
            "3_anni": f"${proj_3y:.2f}",
            "nota": "Basate su rendimenti medi storici, non regressione lineare",
        }

    except Exception as e:
        logging.error(f"Errore in make_projections: {e}")
        return {
            "disclaimer": "⚠️ Impossibile calcolare proiezioni",
            "1_anno": "N/A",
            "2_anni": "N/A",
            "3_anni": "N/A",
        }


# ================== PRICE TARGETS ==================
def calculate_price_targets(df, current_price):
    """Calcola target price e stop loss"""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = ranges.max(axis=1).rolling(14).mean().iloc[-1]

    stop_loss = current_price - (2 * atr)

    bb_upper = df["BB_Upper"].iloc[-1]
    recent_high = df["high"].tail(50).max()

    target_1 = current_price * 1.05
    target_2 = min(bb_upper, recent_high)
    target_3 = current_price * 1.15

    risk_reward = (
        (target_2 - current_price) / (current_price - stop_loss)
        if current_price > stop_loss
        else 0
    )

    return {
        "Stop_Loss": f"${stop_loss:.2f}",
        "Target_1_conservative": f"${target_1:.2f} (+5%)",
        "Target_2_tecnico": f"${target_2:.2f}",
        "Target_3_ottimistico": f"${target_3:.2f} (+15%)",
        "Risk_Reward_Ratio": f"{risk_reward:.2f}",
        "ATR_14": f"${atr:.2f}",
    }


# ================== FORMATTAZIONE OUTPUT ==================
def format_report(report_data):
    """Formatta il report in modo leggibile"""

    def fmt_number(value, decimals=1, prefix="", suffix=""):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{prefix}{num:.{decimals}f}{suffix}"

    def fmt_pct(value):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return "N/A"
        sign = "+" if num >= 0 else ""
        return f"{sign}{num:.1f}%"

    def fmt_price(value):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"${num:.2f}"

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"  📊 ANALISI AZIONARIA: {report_data['Ticker']}")
    lines.append("=" * 70 + "\n")

    # Prezzo e raccomandazione
    lines.append(f"💵 PREZZO ATTUALE: ${report_data['Prezzo_Attuale']}")
    lines.append(f"🎯 {report_data['Raccomandazione']}")
    lines.append(f"📊 Buy Score: {report_data['Buy_Score']}/100")
    lines.append(f"💡 {report_data['Spiegazione']}\n")

    # Warnings
    if report_data.get("Warnings"):
        lines.append("⚠️  ATTENZIONI:")
        for warning in report_data["Warnings"]:
            lines.append(f"   • {warning}")
        lines.append("")

    # Indicatori tecnici
    lines.append("📈 INDICATORI TECNICI")
    lines.append("-" * 70)
    for key in ["Trend", "RSI_Status", "MACD_Status", "BB_Status"]:
        if key in report_data:
            lines.append(f"   {key.replace('_', ' ')}: {report_data[key]}")
    lines.append("")

    # Valutazione
    lines.append("💰 VALUTAZIONE FONDAMENTALE")
    lines.append("-" * 70)
    lines.append(f"   P/E Ratio: {report_data['PE_Ratio']}")
    lines.append(f"   Crescita Revenue YoY: {report_data['Revenue_Growth']}%")
    lines.append("")

    history = report_data.get("History")
    if history:
        lines.append("STORICO MEDIANO (PUNTEGGI/VALUTAZIONI)")
        lines.append("-" * 70)
        obs_line = f"   Osservazioni salvate: {history.get('observations', 0)}"
        if history.get("window_used"):
            obs_line += f" | Mediana su ultimi {history['window_used']} rilevazioni"
        lines.append(obs_line)
        lines.append(
            "   Buy Score mediano: "
            f"{fmt_number(history.get('median_buy_score'))} "
            f"(deviazione attuale: {fmt_pct(history.get('score_vs_median_pct'))})"
        )
        lines.append(
            "   Prezzo mediano: "
            f"{fmt_price(history.get('median_price'))} "
            f"(deviazione attuale: {fmt_pct(history.get('price_vs_median_pct'))})"
        )
        if history.get("median_pe") is not None:
            lines.append(f"   P/E mediano: {fmt_number(history.get('median_pe'))}")
        if history.get("median_revenue_growth") is not None:
            lines.append(
                "   Crescita revenue mediana: "
                f"{fmt_number(history.get('median_revenue_growth'))}%"
            )
        if history.get("median_risk_reward") is not None:
            lines.append(
                "   Risk/Reward mediano: "
                f"{fmt_number(history.get('median_risk_reward'))} "
                f"(deviazione attuale: {fmt_pct(history.get('risk_reward_vs_median_pct'))})"
            )
        lines.append("")

    # Price targets
    lines.append("🎯 LIVELLI DI PREZZO")
    lines.append("-" * 70)
    for key, value in report_data["Price_Targets"].items():
        lines.append(f"   {key.replace('_', ' ')}: {value}")
    lines.append("")

    # Proiezioni con disclaimer
    lines.append("🔮 PROIEZIONI (CON RISERVA)")
    lines.append("-" * 70)
    lines.append(f"   {report_data['Projections']['disclaimer']}")
    lines.append(f"   1 anno: {report_data['Projections']['1_anno']}")
    lines.append(f"   2 anni: {report_data['Projections']['2_anni']}")
    lines.append(f"   3 anni: {report_data['Projections']['3_anni']}")
    lines.append(f"   Nota: {report_data['Projections']['nota']}\n")

    # Motivi principali
    lines.append("📋 MOTIVI PRINCIPALI DELLA VALUTAZIONE")
    lines.append("-" * 70)
    for i, reason in enumerate(report_data["Top_Reasons"][:7], 1):
        lines.append(f"   {i}. {reason}")

    lines.append("\n" + "=" * 70)
    lines.append("⚠️  DISCLAIMER GENERALE")
    lines.append("=" * 70)
    lines.append("Questa analisi è SOLO A SCOPO INFORMATIVO e NON costituisce")
    lines.append(
        "consulenza finanziaria. Investi solo ciò che puoi permetterti di perdere."
    )
    lines.append("Consulta sempre un consulente finanziario professionista.")
    lines.append("=" * 70 + "\n")

    return "\n".join(lines)


# ================== MAIN ANALYSIS FUNCTION ==================
def analyze_company(ticker="AAPL"):
    """Funzione principale di analisi migliorata"""
    try:
        # Validazione
        ticker = validate_ticker(ticker)

        # Verifica orari mercato
        market_open, market_msg = is_market_open()
        logging.info(f"Status mercato: {market_msg}")

        # Fetch dati (con caching automatico)
        df = fetch_historical_data(ticker)
        trend, rsi_status, macd_status, bb_status, df = calculate_technical_indicators(
            df
        )
        current_price = fetch_current_snapshot(ticker)
        eps, revenue_growth_yoy = fetch_financials(ticker)
        pe_ratio = current_price / eps if eps > 0 else "N/A"

        # Calcola score avanzato
        buy_score, reasons, warnings = calculate_advanced_buy_score(
            df,
            current_price,
            eps,
            revenue_growth_yoy,
            trend,
            rsi_status,
            macd_status,
            bb_status,
        )

        recommendation, explanation = get_recommendation(buy_score, warnings)

        # Target price e stop loss
        price_targets = calculate_price_targets(df, current_price)

        # Proiezioni con disclaimer
        projections = make_projections_with_disclaimer(df)

        # Costruisci report strutturato
        report = {
            "Ticker": ticker,
            "Prezzo_Attuale": f"{current_price:.2f}",
            "Raccomandazione": recommendation,
            "Buy_Score": buy_score,
            "Spiegazione": explanation,
            "Warnings": warnings,
            "Market_Status": market_msg,
            "Trend": trend,
            "RSI_Status": rsi_status,
            "MACD_Status": macd_status,
            "BB_Status": bb_status,
            "PE_Ratio": f"{pe_ratio:.2f}" if pe_ratio != "N/A" else "N/A",
            "Revenue_Growth": f"{revenue_growth_yoy:.2f}"
            if revenue_growth_yoy != "N/A"
            else "N/A",
            "Price_Targets": price_targets,
            "Projections": projections,
            "Top_Reasons": reasons[:7],
        }

        # Storico: registra snapshot e calcola mediane per stabilita nel tempo
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "buy_score": buy_score,
            "price": current_price,
        }

        pe_numeric = safe_number(pe_ratio)
        if pe_numeric is not None:
            history_entry["pe_ratio"] = pe_numeric

        revenue_growth_numeric = safe_number(revenue_growth_yoy)
        if revenue_growth_numeric is not None:
            history_entry["revenue_growth"] = revenue_growth_numeric

        risk_reward_value = safe_number(price_targets.get("Risk_Reward_Ratio"))
        if risk_reward_value is not None:
            history_entry["risk_reward"] = risk_reward_value

        history_snapshot = history_tracker.record_and_summarize(ticker, history_entry)
        if history_snapshot:
            report["History"] = history_snapshot

        logging.info(f"✓ Analisi completata per {ticker}")
        return report

    except ValueError as ve:
        logging.error(f"Errore validazione per {ticker}: {ve}")
        return {"Errore": f"Validazione fallita: {str(ve)}"}
    except Exception as e:
        logging.error(f"Errore generale in analyze_company per {ticker}: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return {"Errore": f"Analisi fallita: {str(e)}"}


def _analyze_company_safe(ticker: str):
    """Wrapper sicuro per esecuzione in thread pool"""
    try:
        return analyze_company(ticker)
    except Exception as exc:
        logging.error(f"Errore analisi {ticker} (async): {exc}")
        return {"Errore": f"Analisi fallita: {exc}"}


async def _analyze_tickers_async(tickers, max_workers=8):
    """Esegue analyze_company in parallelo su più ticker"""
    loop = asyncio.get_event_loop()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, _analyze_company_safe, ticker)
            for ticker in tickers
        ]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
    return results


# ================== SCREENING MULTIPLO ==================
def screen_multiple_stocks(
    tickers, min_score=60, show_all=False, use_async=True, max_workers=8
):
    """
    Analizza multipli ticker e filtra quelli con score alto

    Args:
        tickers: Lista di ticker da analizzare
        min_score: Score minimo per segnalare (default 60)
        show_all: Se True, mostra tutti i risultati anche con score basso
        use_async: Se True, usa modalità parallela per liste grandi
        max_workers: Numero massimo di worker paralleli

    Returns:
        Lista di report ordinati per score
    """
    results = []

    print(f"\n{'=' * 70}")
    print(f"🔍 SCREENING DI {len(tickers)} AZIONI")
    print(f"📊 Soglia minima: {min_score}/100")
    print(f"{'=' * 70}\n")

    ran_async = False

    if use_async and len(tickers) >= max_workers:
        ran_async = True
        print(f"Modalità parallela attiva: fino a {max_workers} worker...")
        async_results = asyncio.run(_analyze_tickers_async(tickers, max_workers))
        for result in async_results:
            if not result or "Errore" in result:
                continue
            results.append(result)
    else:
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Analizzando {ticker}...", end=" ")

            try:
                result = analyze_company(ticker)

                if "Errore" in result:
                    print(f"❌ Errore: {result['Errore']}")
                    continue

                score = result["Buy_Score"]
                results.append(result)

                # Emoji basato su score
                if score >= 75:
                    emoji = "🟢🟢"
                elif score >= 60:
                    emoji = "🟢"
                elif score >= 45:
                    emoji = "🟡"
                elif score >= 30:
                    emoji = "🟠"
                else:
                    emoji = "🔴"

                print(f"{emoji} Score: {score}/100")

            except Exception as e:
                print(f"❌ Errore: {str(e)}")
                logging.error(f"Errore screening {ticker}: {e}")

    # Ordina per score (dal più alto al più basso)
    results.sort(key=lambda x: x["Buy_Score"], reverse=True)

    # Filtra in base a min_score se non show_all
    if not show_all:
        filtered = [r for r in results if r["Buy_Score"] >= min_score]
    else:
        filtered = results

    print(f"\n{'=' * 70}")
    print(f"✅ Analisi completata: {len(filtered)}/{len(results)} azioni sopra soglia")
    if ran_async:
        print("Modalità parallela completata.")
    print(f"{'=' * 70}\n")

    return filtered


def print_screening_summary(results):
    """Stampa summary table dei risultati"""
    if not results:
        print("❌ Nessuna azione trovata sopra la soglia\n")
        return

    print("\n" + "=" * 90)
    print(
        f"{'TICKER':<10} {'SCORE':<8} {'RACCOM.':<20} {'PREZZO':<12} {'P/E':<8} {'TREND':<8}"
    )
    print("=" * 90)

    for r in results:
        ticker = r["Ticker"]
        score = r["Buy_Score"]
        rec = r["Raccomandazione"].split()[1]  # Estrae solo BUY/SELL/HOLD
        price = f"${r['Prezzo_Attuale']}"
        pe = r["PE_Ratio"]
        trend = r["Trend"]

        # Colora score
        if score >= 75:
            score_str = f"🟢 {score}"
        elif score >= 60:
            score_str = f"🟢 {score}"
        elif score >= 45:
            score_str = f"🟡 {score}"
        else:
            score_str = f"🟠 {score}"

        print(f"{ticker:<10} {score_str:<8} {rec:<20} {price:<12} {pe:<8} {trend:<8}")

    print("=" * 90 + "\n")


def print_top_picks(results, top_n=3):
    """Stampa i top N pick con dettagli completi"""
    if not results:
        return

    print(f"\n🏆 TOP {min(top_n, len(results))} OPPORTUNITÀ\n")

    for i, result in enumerate(results[:top_n], 1):
        print(f"\n{'━' * 70}")
        print(f"#{i} - {result['Ticker']} - Score: {result['Buy_Score']}/100")
        print(f"{'━' * 70}")
        print(format_report(result))


def save_results_to_file(results, filename="screening_results.txt"):
    """Salva i risultati in un file di testo"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"SCREENING RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 90 + "\n\n")

            for result in results:
                f.write(format_report(result))
                f.write("\n\n")

        print(f"✅ Risultati salvati in: {filename}")
        return True
    except Exception as e:
        print(f"❌ Errore salvataggio file: {e}")
        return False


# ================== DISCOVERY: TROVARE NUOVE AZIONI ==================


def get_sp500_tickers():
    """Scarica lista completa S&P 500 da Wikipedia"""
    try:
        import pandas as pd
        import urllib.request

        # Aggiungi User-Agent per evitare 403
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response:
            tables = pd.read_html(response.read())

        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-").tolist()
        logging.info(f"✓ Scaricati {len(tickers)} ticker S&P 500")
        return tickers
    except Exception as e:
        logging.error(f"Errore download S&P 500: {e}")
        # Fallback: usa una lista statica ridotta
        logging.info("⚠️ Usando lista S&P 500 ridotta (top 50)")
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK-B",
            "UNH",
            "XOM",
            "JNJ",
            "JPM",
            "V",
            "PG",
            "LLY",
            "MA",
            "HD",
            "CVX",
            "MRK",
            "ABBV",
            "PEP",
            "KO",
            "AVGO",
            "COST",
            "WMT",
            "MCD",
            "CSCO",
            "TMO",
            "ACN",
            "ABT",
            "CRM",
            "ADBE",
            "NFLX",
            "DHR",
            "NKE",
            "DIS",
            "VZ",
            "WFC",
            "CMCSA",
            "NEE",
            "TXN",
            "UPS",
            "PM",
            "BMY",
            "RTX",
            "HON",
            "QCOM",
            "INTC",
            "IBM",
            "AMD",
        ]


def get_nasdaq100_tickers():
    """Lista Nasdaq 100 (top tech stocks)"""
    nasdaq100 = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AVGO",
        "COST",
        "NFLX",
        "TMUS",
        "CSCO",
        "ADBE",
        "PEP",
        "AMD",
        "QCOM",
        "TXN",
        "INTU",
        "CMCSA",
        "AMGN",
        "HON",
        "AMAT",
        "SBUX",
        "ISRG",
        "BKNG",
        "VRTX",
        "ADP",
        "GILD",
        "PANW",
        "ADI",
        "MU",
        "LRCX",
        "INTC",
        "PYPL",
        "REGN",
        "MDLZ",
        "KLAC",
        "SNPS",
        "CDNS",
        "MELI",
        "ASML",
        "CRWD",
        "MAR",
        "CSX",
        "ABNB",
        "FTNT",
        "NXPI",
        "ORLY",
        "WDAY",
        "DASH",
        "ROP",
        "ADSK",
        "MNST",
        "PCAR",
        "AEP",
        "CHTR",
        "CPRT",
        "PAYX",
        "ROST",
        "MRVL",
        "ODFL",
        "KDP",
        "DXCM",
        "FAST",
        "EA",
        "CTAS",
        "VRSK",
        "KHC",
        "CTSH",
        "EXC",
        "LULU",
        "GEHC",
        "XEL",
        "IDXX",
        "CCEP",
        "TEAM",
        "BKR",
        "ZS",
        "DDOG",
        "TTD",
        "FANG",
        "ANSS",
        "ON",
        "CDW",
        "CSGP",
        "BIIB",
        "GFS",
        "MDB",
        "WBD",
        "ILMN",
        "ARM",
        "MRNA",
        "DLTR",
        "WBA",
        "ALGN",
        "SMCI",
        "LCID",
        "RIVN",
        "ENPH",
    ]
    return nasdaq100


def get_dow30_tickers():
    """Dow Jones 30 (blue chips)"""
    dow30 = [
        "AAPL",
        "MSFT",
        "UNH",
        "GS",
        "HD",
        "CAT",
        "MCD",
        "AMGN",
        "V",
        "CRM",
        "BA",
        "HON",
        "IBM",
        "TRV",
        "AXP",
        "JPM",
        "JNJ",
        "WMT",
        "PG",
        "CVX",
        "MRK",
        "DIS",
        "NKE",
        "CSCO",
        "KO",
        "DOW",
        "VZ",
        "MMM",
        "AMZN",
        "INTC",
    ]
    return dow30


def discover_by_sector(sector):
    """Scopri azioni per settore"""
    sectors = {
        "tech": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "NVDA",
            "META",
            "AMD",
            "INTC",
            "QCOM",
            "AVGO",
            "ORCL",
            "CRM",
            "ADBE",
            "NOW",
            "PANW",
            "CRWD",
            "ZS",
            "DDOG",
            "NET",
            "SNOW",
            "PLTR",
        ],
        "finance": [
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            "BLK",
            "SCHW",
            "AXP",
            "SPGI",
            "CME",
            "ICE",
            "CB",
            "PGR",
            "TRV",
            "AFL",
            "ALL",
            "AIG",
            "MET",
            "PRU",
        ],
        "healthcare": [
            "UNH",
            "JNJ",
            "LLY",
            "ABBV",
            "MRK",
            "TMO",
            "ABT",
            "DHR",
            "PFE",
            "BMY",
            "AMGN",
            "GILD",
            "VRTX",
            "REGN",
            "CVS",
            "CI",
            "HUM",
            "ISRG",
            "SYK",
            "BSX",
        ],
        "consumer": [
            "AMZN",
            "TSLA",
            "HD",
            "WMT",
            "NKE",
            "MCD",
            "SBUX",
            "TGT",
            "LOW",
            "COST",
            "DIS",
            "BKNG",
            "MAR",
            "CMG",
            "YUM",
            "ROST",
            "ULTA",
            "DG",
            "DLTR",
            "LULU",
        ],
        "energy": [
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "EOG",
            "MPC",
            "PSX",
            "VLO",
            "OXY",
            "HAL",
            "BKR",
            "FANG",
            "DVN",
            "HES",
            "MRO",
            "APA",
            "CTRA",
            "OVV",
            "PXD",
            "EQT",
        ],
        "industrial": [
            "CAT",
            "BA",
            "HON",
            "UPS",
            "RTX",
            "LMT",
            "GE",
            "DE",
            "MMM",
            "UNP",
            "ETN",
            "ITW",
            "EMR",
            "GD",
            "NOC",
            "FDX",
            "CSX",
            "NSC",
            "WM",
            "RSG",
        ],
        "real_estate": [
            "AMT",
            "PLD",
            "CCI",
            "EQIX",
            "PSA",
            "WELL",
            "DLR",
            "O",
            "SPG",
            "VICI",
            "AVB",
            "EQR",
            "SBAC",
            "WY",
            "ARE",
            "INVH",
            "EXR",
            "MAA",
            "SUI",
            "UDR",
        ],
        "materials": [
            "LIN",
            "APD",
            "SHW",
            "FCX",
            "NEM",
            "ECL",
            "CTVA",
            "DOW",
            "DD",
            "NUE",
            "VMC",
            "MLM",
            "PPG",
            "IFF",
            "BALL",
            "AVY",
            "AMCR",
            "PKG",
            "IP",
            "SEE",
        ],
        "utilities": [
            "NEE",
            "DUK",
            "SO",
            "D",
            "AEP",
            "EXC",
            "SRE",
            "PEG",
            "XEL",
            "ED",
            "WEC",
            "ES",
            "AWK",
            "DTE",
            "PPL",
            "ETR",
            "FE",
            "AEE",
            "CMS",
            "CNP",
        ],
        "telecom": ["T", "VZ", "TMUS", "CHTR", "CMCSA"],
    }
    return sectors.get(sector.lower(), [])


def discover_by_market_cap(cap_type="large"):
    """Scopri azioni per capitalizzazione"""
    if cap_type == "mega":  # > $200B
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK-B",
            "LLY",
            "V",
        ]
    elif cap_type == "large":  # $10B - $200B
        return get_sp500_tickers()[:100]  # Top 100 S&P
    elif cap_type == "mid":  # $2B - $10B
        return [
            "COIN",
            "HOOD",
            "RBLX",
            "DASH",
            "ABNB",
            "BROS",
            "DKNG",
            "SOFI",
            "OPEN",
            "PLTR",
        ]
    elif cap_type == "small":  # < $2B
        return [
            "LCID",
            "RIVN",
            "FUBO",
            "SKLZ",
            "WISH",
            "ROOT",
            "CLOV",
            "SPCE",
            "WKHS",
            "RIDE",
        ]
    else:
        return []


def discover_trending_stocks():
    """Scopri azioni trending/popolari (devi implementare con API esterne)"""
    # Questa è una lista statica, ma potresti usare API come:
    # - Yahoo Finance Trending Tickers
    # - Reddit WallStreetBets API
    # - Twitter/X trending $CASHTAGS
    trending = [
        "NVDA",
        "TSLA",
        "PLTR",
        "AMD",
        "COIN",
        "HOOD",
        "SOFI",
        "RBLX",
        "NIO",
        "LCID",
        "RIVN",
        "DKNG",
        "DASH",
        "ABNB",
        "SNOW",
        "NET",
    ]
    return trending


def discover_by_performance(period="1mo", min_return=10):
    """
    Trova azioni con performance superiore a X% in un periodo
    NOTA: Questo è computazionalmente costoso - usa con liste piccole
    """
    print(f"\n🔍 Ricerca azioni con return > {min_return}% negli ultimi {period}...")

    # Usa una lista base (es. S&P 500)
    base_tickers = get_sp500_tickers()[:50]  # Limita a 50 per velocità

    strong_performers = []

    for ticker in base_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if len(hist) > 0:
                perf = (
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                    / hist["Close"].iloc[0]
                ) * 100

                if perf >= min_return:
                    strong_performers.append(
                        {"ticker": ticker, "return": round(perf, 2)}
                    )
                    print(f"  ✓ {ticker}: +{perf:.1f}%")
        except:
            continue

    # Ordina per performance
    strong_performers.sort(key=lambda x: x["return"], reverse=True)

    return [item["ticker"] for item in strong_performers]


def discover_by_criteria(
    min_pe=None, max_pe=None, min_revenue_growth=None, min_market_cap=None
):
    """
    Filtra azioni per criteri fondamentali
    NOTA: Molto lento su liste grandi - usa campioni
    """
    print(f"\n🔍 Ricerca azioni con criteri personalizzati...")

    base_tickers = get_sp500_tickers()[:100]  # Limita per velocità
    matches = []

    for ticker in base_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Filtri
            pe = info.get("trailingPE", 999)
            rev_growth = info.get("revenueGrowth", -1) * 100
            market_cap = info.get("marketCap", 0) / 1e9  # In billions

            # Controlla criteri
            if min_pe and pe < min_pe:
                continue
            if max_pe and pe > max_pe:
                continue
            if min_revenue_growth and rev_growth < min_revenue_growth:
                continue
            if min_market_cap and market_cap < min_market_cap:
                continue

            matches.append(ticker)
            print(
                f"  ✓ {ticker}: P/E={pe:.1f}, Rev Growth={rev_growth:.1f}%, MCap=${market_cap:.1f}B"
            )

        except:
            continue

    return matches


# ================== WATCHLISTS PRECONFIGURATE ==================
WATCHLISTS = {
    "tech_giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "DHR", "MRK"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "consumer": ["WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST"],
    "dividend": ["JNJ", "PG", "KO", "PEP", "MCD", "T", "VZ"],
    "growth": ["NVDA", "AVGO", "AMD", "PLTR", "SNOW", "NET", "DDOG"],
    "sp500_sample": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "JPM",
        "V",
        "UNH",
        "HD",
        "PG",
        "MA",
    ],
    # NUOVE WATCHLISTS PER DISCOVERY
    "sp500_full": "dynamic",  # Scarica dinamicamente
    "nasdaq100": "dynamic",
    "dow30": "dynamic",
    "trending": "dynamic",
}


def get_watchlist(name):
    """Ottiene watchlist statica o dinamica"""
    if name in WATCHLISTS:
        if WATCHLISTS[name] == "dynamic":
            if name == "sp500_full":
                return get_sp500_tickers()
            elif name == "nasdaq100":
                return get_nasdaq100_tickers()
            elif name == "dow30":
                return get_dow30_tickers()
            elif name == "trending":
                return discover_trending_stocks()
        else:
            return WATCHLISTS[name]
    return []


def list_watchlists():
    """Mostra le watchlist disponibili"""
    print("\n📋 WATCHLISTS DISPONIBILI:\n")

    print("🔹 LISTE STATICHE:")
    static = {k: v for k, v in WATCHLISTS.items() if v != "dynamic"}
    for name, tickers in static.items():
        print(
            f"  • {name:<15} ({len(tickers)} azioni): {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}"
        )

    print("\n🔹 LISTE DINAMICHE (scaricate live):")
    print(f"  • sp500_full      (500 azioni): Tutte le azioni S&P 500")
    print(f"  • nasdaq100       (100 azioni): Top tech stocks")
    print(f"  • dow30           (30 azioni): Blue chips Dow Jones")
    print(f"  • trending        (~15 azioni): Azioni trending/popolari")

    print("\n💡 Puoi anche usare i comandi di discovery!\n")


def discovery_menu():
    """Menu interattivo per scoprire azioni"""
    print("\n" + "=" * 70)
    print("  🔍 DISCOVERY: TROVA NUOVE AZIONI DA ANALIZZARE")
    print("=" * 70 + "\n")

    print("Scegli un metodo:")
    print("  1. Per SETTORE (tech, finance, healthcare, ecc.)")
    print("  2. Per CAPITALIZZAZIONE (mega, large, mid, small)")
    print("  3. Per PERFORMANCE (trova azioni che hanno guadagnato X%)")
    print("  4. Per CRITERI FONDAMENTALI (P/E, crescita revenue, ecc.)")
    print("  5. Liste PREDEFINITE (S&P 500, Nasdaq 100, Dow 30)")
    print("  6. Azioni TRENDING")
    print("  0. Torna indietro\n")

    choice = input("👉 Scelta: ").strip()

    discovered_tickers = []

    if choice == "1":
        print("\n📊 Settori disponibili:")
        sectors = [
            "tech",
            "finance",
            "healthcare",
            "consumer",
            "energy",
            "industrial",
            "real_estate",
            "materials",
            "utilities",
            "telecom",
        ]
        for i, s in enumerate(sectors, 1):
            print(f"  {i}. {s}")

        sector = input("\n👉 Scegli settore: ").strip().lower()
        discovered_tickers = discover_by_sector(sector)

        if discovered_tickers:
            print(f"\n✅ Trovate {len(discovered_tickers)} azioni nel settore {sector}")
            print(
                f"📋 {', '.join(discovered_tickers[:10])}{'...' if len(discovered_tickers) > 10 else ''}"
            )

    elif choice == "2":
        print("\n💰 Capitalizzazioni:")
        print("  1. MEGA CAP (> $200B)")
        print("  2. LARGE CAP ($10B - $200B)")
        print("  3. MID CAP ($2B - $10B)")
        print("  4. SMALL CAP (< $2B)")

        cap = input("\n👉 Scelta: ").strip()
        cap_map = {"1": "mega", "2": "large", "3": "mid", "4": "small"}

        if cap in cap_map:
            discovered_tickers = discover_by_market_cap(cap_map[cap])
            print(f"\n✅ Trovate {len(discovered_tickers)} azioni {cap_map[cap]} cap")

    elif choice == "3":
        period = input("\n📅 Periodo (1mo, 3mo, 6mo, 1y): ").strip() or "1mo"
        min_return = input("📈 Return minimo % (default 10): ").strip()
        min_return = int(min_return) if min_return.isdigit() else 10

        discovered_tickers = discover_by_performance(period, min_return)

        if discovered_tickers:
            print(
                f"\n✅ Trovate {len(discovered_tickers)} azioni con return > {min_return}%"
            )

    elif choice == "4":
        print("\n🔍 Filtra per criteri fondamentali")
        print("(Lascia vuoto per ignorare il criterio)\n")

        max_pe = input("P/E massimo (es. 25): ").strip()
        max_pe = float(max_pe) if max_pe else None

        min_growth = input("Crescita revenue minima % (es. 10): ").strip()
        min_growth = float(min_growth) if min_growth else None

        min_mcap = input("Market cap minima $B (es. 10): ").strip()
        min_mcap = float(min_mcap) if min_mcap else None

        discovered_tickers = discover_by_criteria(
            max_pe=max_pe, min_revenue_growth=min_growth, min_market_cap=min_mcap
        )

        if discovered_tickers:
            print(
                f"\n✅ Trovate {len(discovered_tickers)} azioni che soddisfano i criteri"
            )

    elif choice == "5":
        print("\n📋 Liste predefinite:")
        print("  1. S&P 500 (tutte le 500 azioni)")
        print("  2. Nasdaq 100")
        print("  3. Dow Jones 30")

        list_choice = input("\n👉 Scelta: ").strip()

        if list_choice == "1":
            discovered_tickers = get_sp500_tickers()
        elif list_choice == "2":
            discovered_tickers = get_nasdaq100_tickers()
        elif list_choice == "3":
            discovered_tickers = get_dow30_tickers()

        if discovered_tickers:
            print(f"\n✅ Caricate {len(discovered_tickers)} azioni")

    elif choice == "6":
        discovered_tickers = discover_trending_stocks()
        print(f"\n✅ Azioni trending: {', '.join(discovered_tickers)}")

    # Dopo aver scoperto, chiedi cosa fare
    if discovered_tickers:
        print("\n" + "─" * 70)
        print("Cosa vuoi fare con queste azioni?")
        print("  1. Analizza TUTTE (può richiedere tempo)")
        print("  2. Analizza solo le PRIME 10")
        print("  3. Analizza solo le PRIME 20")
        print("  4. Salva lista e torna al menu")
        print("  0. Annulla")

        action = input("\n👉 Scelta: ").strip()

        if action == "1":
            tickers_to_screen = discovered_tickers
        elif action == "2":
            tickers_to_screen = discovered_tickers[:10]
        elif action == "3":
            tickers_to_screen = discovered_tickers[:20]
        elif action == "4":
            filename = (
                input("\n💾 Nome file (default: discovered_stocks.txt): ").strip()
                or "discovered_stocks.txt"
            )
            with open(filename, "w") as f:
                f.write("\n".join(discovered_tickers))
            print(f"✅ Lista salvata in {filename}")
            return
        else:
            return

        # Esegui screening
        min_score = input("\n📊 Score minimo (default 60): ").strip()
        min_score = int(min_score) if min_score.isdigit() else 60

        results = screen_multiple_stocks(tickers_to_screen, min_score=min_score)
        print_screening_summary(results)

        if results:
            show_details = input("\n📋 Mostrare dettagli top 3? (s/n): ").lower()
            if show_details == "s":
                print_top_picks(results, top_n=3)

            save = input("\n💾 Salvare risultati? (s/n): ").lower()
            if save == "s":
                filename = f"screening_discovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                save_results_to_file(results, filename)


# ================== ESEMPIO DI UTILIZZO ==================
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("  🚀 STOCK ANALYZER - SCREENING TOOL")
    print("=" * 70)

    # Modalità 1: Analisi singola
    if len(sys.argv) == 2:
        ticker = sys.argv[1].upper()
        print(f"\n🔍 Modalità: Analisi singola per {ticker}\n")
        result = analyze_company(ticker)
        if "Errore" not in result:
            print(format_report(result))
        else:
            print(f"❌ {result['Errore']}")

    # Modalità 2: Screening watchlist
    elif len(sys.argv) == 3 and sys.argv[1] == "--watchlist":
        watchlist_name = sys.argv[2].lower()

        if watchlist_name == "list":
            list_watchlists()
        elif watchlist_name in WATCHLISTS:
            tickers = WATCHLISTS[watchlist_name]
            results = screen_multiple_stocks(tickers, min_score=60)
            print_screening_summary(results)
            print_top_picks(results, top_n=3)

            # Chiedi se salvare
            save = input("\n💾 Vuoi salvare i risultati in un file? (s/n): ").lower()
            if save == "s":
                filename = f"screening_{watchlist_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                save_results_to_file(results, filename)
        else:
            print(f"❌ Watchlist '{watchlist_name}' non trovata.")
            list_watchlists()

    # Modalità 3: Screening custom
    elif len(sys.argv) >= 3 and sys.argv[1] == "--screen":
        tickers = [t.upper() for t in sys.argv[2:]]
        print(f"\n🔍 Modalità: Screening personalizzato di {len(tickers)} azioni\n")
        results = screen_multiple_stocks(tickers, min_score=60)
        print_screening_summary(results)
        print_top_picks(results, top_n=3)

    # Default: Modalità interattiva
    else:
        print("\n📖 MODALITÀ D'USO:\n")
        print("1️⃣  Analisi singola:")
        print("   python stock_analyzer.py AAPL\n")

        print("2️⃣  Screening watchlist:")
        print("   python stock_analyzer.py --watchlist mag7")
        print(
            "   python stock_analyzer.py --watchlist list  (mostra tutte le watchlist)\n"
        )

        print("3️⃣  Screening personalizzato:")
        print("   python stock_analyzer.py --screen AAPL MSFT GOOGL NVDA\n")

        print("4️⃣  Modalità interattiva:")
        print()

        # Menu interattivo
        while True:
            print("\n" + "─" * 70)
            print("Scegli un'opzione:")
            print("  1. Analizza singolo ticker")
            print("  2. Screening watchlist")
            print("  3. Screening personalizzato")
            print("  4. 🔍 DISCOVERY - Trova nuove azioni")
            print("  5. Esci")
            print("─" * 70)

            choice = input("\n👉 Scelta: ").strip()

            if choice == "1":
                ticker = input("\n📊 Inserisci ticker (es. AAPL): ").upper().strip()
                result = analyze_company(ticker)
                if "Errore" not in result:
                    print(format_report(result))
                else:
                    print(f"\n❌ {result['Errore']}")

            elif choice == "2":
                list_watchlists()
                watchlist_name = input("\n📋 Scegli watchlist: ").lower().strip()

                # Gestisci watchlist dinamiche
                tickers = get_watchlist(watchlist_name)

                if tickers:
                    min_score = input("\n📊 Score minimo (default 60): ").strip()
                    min_score = int(min_score) if min_score.isdigit() else 60

                    results = screen_multiple_stocks(tickers, min_score=min_score)
                    print_screening_summary(results)

                    if results:
                        show_details = input(
                            "\n📋 Mostrare dettagli top 3? (s/n): "
                        ).lower()
                        if show_details == "s":
                            print_top_picks(results, top_n=3)
                else:
                    print(f"\n❌ Watchlist '{watchlist_name}' non trovata")

            elif choice == "3":
                tickers_input = input(
                    "\n📊 Inserisci ticker separati da spazio (es. AAPL MSFT GOOGL): "
                )
                tickers = [t.upper().strip() for t in tickers_input.split()]

                if tickers:
                    min_score = input("\n📊 Score minimo (default 60): ").strip()
                    min_score = int(min_score) if min_score.isdigit() else 60

                    results = screen_multiple_stocks(tickers, min_score=min_score)
                    print_screening_summary(results)

                    if results:
                        show_details = input(
                            "\n📋 Mostrare dettagli top 3? (s/n): "
                        ).lower()
                        if show_details == "s":
                            print_top_picks(results, top_n=3)
                else:
                    print("\n❌ Nessun ticker inserito")

            elif choice == "4":
                discovery_menu()

            elif choice == "5":
                print("\n👋 Arrivederci!\n")
                break

            else:
                print("\n❌ Scelta non valida")