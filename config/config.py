# config.py
import os
from datetime import datetime, timedelta

# Basic configuration
INITIAL_AMOUNT = 100000
TRANSACTION_COST_PCT = 0.001
HMAX = 100  # maximum shares to hold

# Crypto symbols (เริ่มจาก 1 symbol ก่อน)
CRYPTO_SYMBOLS = ["BTC-USD"]

# Date range
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years of data

# Technical indicators (ใช้ indicators เดียวกับใน main.py - รวม 12 ตัว)
INDICATORS = [
    # Moving Averages
    "sma_20",          # Simple Moving Average 20 periods
    "ema_20",          # Exponential Moving Average 20 periods
    
    # Momentum Oscillators
    "rsi_14",          # Relative Strength Index 14 periods
    
    # Trend Indicators
    "macd",            # MACD Line
    "macd_signal",     # MACD Signal Line
    "macd_hist",       # MACD Histogram
    
    # Volatility Indicators (Bollinger Bands)
    "bb_middle",       # Bollinger Bands Middle Line (SMA 20)
    "bb_std",          # Bollinger Bands Standard Deviation
    "bb_upper",        # Bollinger Bands Upper Band
    "bb_lower",        # Bollinger Bands Lower Band
    
    # Volume Indicators
    "volume_sma_20",   # Volume Simple Moving Average 20 periods
    "volume_ratio"     # Volume Ratio (current volume / volume SMA)
]

# Model parameters
MODEL_KWARGS = {
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
}

# ===========================================
# STANDARDIZED DIRECTORY STRUCTURE
# ===========================================
# All approaches (Native Python, Notebooks, Streamlit UI) use these directories

# Root data directory for raw data from exchanges
DATA_DIR = "data"

# Subdirectory for processed data ready for training
DATA_PREPARE_DIR = os.path.join(DATA_DIR, "data_prepare")

# Root models directory for trained agents
MODEL_DIR = "agents"

# Data sources configuration
DATA_SOURCES = {
    'yahoo_finance': True,  # Primary source via yfinance
    'ccxt': False,         # Optional CCXT integration
}

# Data pipeline stages
DATA_PIPELINE = {
    'raw_data': DATA_DIR,           # Raw CSV files from exchanges
    'processed_data': DATA_PREPARE_DIR,  # Data with indicators, cleaned, normalized
    'models': MODEL_DIR,            # Trained RL agents
}

# ===========================================
# CREATE DIRECTORIES
# ===========================================
# Ensure all required directories exist
for directory in [DATA_DIR, DATA_PREPARE_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)