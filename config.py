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

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"

# Create directories if not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)