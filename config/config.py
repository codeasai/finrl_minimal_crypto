# config.py
import os
import torch
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

# Technical indicators (ปรับให้ตรงกับ feature data ที่มี - รวม 12 ตัว)
INDICATORS = [
    # Moving Averages
    "sma_20",          # Simple Moving Average 20 periods
    "ema_20",          # Exponential Moving Average 20 periods
    
    # Momentum Oscillators
    "rsi",             # Relative Strength Index (ใช้ rsi แทน rsi_14)
    
    # Trend Indicators
    "macd",            # MACD Line
    "macd_signal",     # MACD Signal Line
    "macd_histogram",  # MACD Histogram (ใช้ macd_histogram แทน macd_hist)
    
    # Volatility Indicators (Bollinger Bands)
    "bb_middle",       # Bollinger Bands Middle Line (SMA 20)
    "bb_width",        # Bollinger Bands Width (แทน bb_std)
    "bb_upper",        # Bollinger Bands Upper Band
    "bb_lower",        # Bollinger Bands Lower Band
    
    # Volume Indicators
    "volume_sma_20",   # Volume Simple Moving Average 20 periods
    "volume_ratio"     # Volume Ratio (current volume / volume SMA)
]

# ===========================================
# DEVICE CONFIGURATION - GPU REQUIRED
# ===========================================

def check_gpu_availability():
    """ตรวจสอบความพร้อมใช้งานของ GPU"""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    
    if torch.cuda.device_count() == 0:
        return False, "No CUDA devices found"
    
    try:
        # Test GPU memory allocation
        device = torch.device('cuda:0')
        test_tensor = torch.zeros(100, 100).to(device)
        del test_tensor
        torch.cuda.empty_cache()
        return True, f"GPU available: {torch.cuda.get_device_name(0)}"
    except Exception as e:
        return False, f"GPU test failed: {str(e)}"

def get_device_config():
    """Get device configuration with GPU requirement"""
    gpu_available, gpu_message = check_gpu_availability()
    
    if gpu_available:
        device = 'cuda'
        print(f"✅ {gpu_message}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = None  # Will cause training to fail
        print(f"❌ GPU Required: {gpu_message}")
        print("⚠️ Training will not be allowed without GPU")
    
    return device, gpu_available

# Initialize device configuration
DEVICE, GPU_AVAILABLE = get_device_config()

# Model parameters
MODEL_KWARGS = {
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'device': DEVICE if GPU_AVAILABLE else 'cuda',  # Force CUDA even if not available to trigger error
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