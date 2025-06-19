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

# Technical indicators (เลือกแค่พื้นฐาน)
INDICATORS = [
    "macd",
    "rsi_30", 
    "cci_30",
    "dx_30"
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
MODEL_DIR = "models"

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