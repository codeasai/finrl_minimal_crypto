# config.py
# Configuration file for Crypto Trading RL Agent

# ==================== DATA CONFIGURATION ====================

# Cryptocurrency symbols to trade
CRYPTO_SYMBOLS = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum  
    'ADA-USD',   # Cardano
    'SOL-USD',   # Solana
    'DOT-USD'    # Polkadot
]

# Date range for data download
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# ==================== TRADING CONFIGURATION ====================

# Initial portfolio amount
INITIAL_AMOUNT = 100000  # $100,000

# Maximum holding for each asset
HMAX = 1000

# Transaction cost percentage
TRANSACTION_COST_PCT = 0.001  # 0.1%

# ===========================================
# STANDARDIZED DIRECTORY STRUCTURE
# ===========================================
# Use same directories as main config for consistency across all approaches

# Data directories (same as root config.py)
DATA_DIR = "../data"  # Relative to notebooks directory
DATA_PREPARE_DIR = "../data/data_prepare"

# Model directory (same as root config.py)
MODEL_DIR = "../models"  # Relative to notebooks directory

# ==================== TECHNICAL INDICATORS ====================

# Technical indicators to use (same as main config.py)
INDICATORS = [
    'sma_20',         # Simple Moving Average (20 days)
    'ema_20',         # Exponential Moving Average (20 days)
    'rsi_14',         # Relative Strength Index (14 days)
    'macd',           # MACD line
    'macd_signal',    # MACD signal line
    'macd_hist',      # MACD histogram
    'bb_middle',      # Bollinger Bands middle line
    'bb_std',         # Bollinger Bands standard deviation
    'bb_upper',       # Bollinger Bands upper line
    'bb_lower',       # Bollinger Bands lower line
    'volume_sma_20',  # Volume Simple Moving Average (20 days)
    'volume_ratio'    # Volume ratio
]

# Backward compatibility
TECHNICAL_INDICATORS = INDICATORS

# ==================== MODEL PARAMETERS ====================

# PPO (Proximal Policy Optimization) Parameters
PPO_PARAMS = {
    'learning_rate': 1e-4,
    'n_steps': 1024,
    'batch_size': 128,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'target_kl': 0.02,
    'verbose': 1
}

# A2C (Advantage Actor-Critic) Parameters
A2C_PARAMS = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.01,
    'vf_coef': 0.25,
    'max_grad_norm': 0.5,
    'verbose': 1
}

# DDPG (Deep Deterministic Policy Gradient) Parameters
DDPG_PARAMS = {
    'learning_rate': 1e-3,
    'buffer_size': 100000,
    'learning_starts': 100,
    'batch_size': 100,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'verbose': 1
}

# SAC (Soft Actor-Critic) Parameters
SAC_PARAMS = {
    'learning_rate': 3e-4,
    'buffer_size': 100000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 'auto',
    'verbose': 1
}

# ==================== TRAINING CONFIGURATION ====================

# Training parameters
TRAINING_TIMESTEPS = 100000    # Total training timesteps
EVAL_FREQ = 10000             # Evaluation frequency
N_EVAL_EPISODES = 5           # Number of evaluation episodes
SAVE_FREQ = 25000             # Model save frequency
LOG_INTERVAL = 100            # Logging interval

# ==================== RISK MANAGEMENT ====================

# Risk management parameters
MAX_POSITION_SIZE = 0.2       # Maximum 20% of portfolio per position
STOP_LOSS_PCT = 0.05          # 5% stop loss
TAKE_PROFIT_PCT = 0.15        # 15% take profit
MAX_DAILY_TRADES = 10         # Maximum trades per day
MAX_DRAWDOWN_PCT = 0.1        # Maximum 10% drawdown before stopping

# ==================== LIVE TRADING CONFIGURATION ====================

# Live trading parameters
PAPER_TRADING_BALANCE = 100000  # Paper trading initial balance
LIVE_TRADING_MIN_BALANCE = 1000 # Minimum balance for live trading
TRADING_INTERVAL_MINUTES = 60   # Trading decision interval (60 minutes)

# Safety limits
SAFETY_MAX_ORDER_SIZE = 1000    # Maximum $1000 per order
SAFETY_MAX_DAILY_LOSS = 0.02    # Maximum 2% daily loss
SAFETY_REQUIRE_CONFIRMATION = True  # Require confirmation for live trades

# ==================== API CONFIGURATION ====================

# Exchange API settings (to be filled by user)
EXCHANGE_API_KEY = ""           # Your exchange API key
EXCHANGE_API_SECRET = ""        # Your exchange API secret
EXCHANGE_TESTNET = True         # Use testnet by default

# ==================== NOTIFICATIONS ====================

# Notification settings
ENABLE_NOTIFICATIONS = True
NOTIFICATION_EMAIL = ""         # Email for notifications
NOTIFICATION_WEBHOOK = ""       # Webhook URL for notifications

# ==================== LOGGING ====================

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "crypto_trading_agent.log"
ENABLE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = "tensorboard_logs"

# ==================== VALIDATION ====================

def validate_config():
    """
    Validate configuration parameters
    """
    errors = []
    
    # Validate symbols
    if not CRYPTO_SYMBOLS:
        errors.append("CRYPTO_SYMBOLS cannot be empty")
    
    # Validate amounts
    if INITIAL_AMOUNT <= 0:
        errors.append("INITIAL_AMOUNT must be positive")
    
    if TRANSACTION_COST_PCT < 0 or TRANSACTION_COST_PCT > 0.1:
        errors.append("TRANSACTION_COST_PCT should be between 0 and 0.1")
    
    # Validate training parameters
    if TRAINING_TIMESTEPS <= 0:
        errors.append("TRAINING_TIMESTEPS must be positive")
    
    # Validate risk parameters
    if MAX_POSITION_SIZE <= 0 or MAX_POSITION_SIZE > 1:
        errors.append("MAX_POSITION_SIZE should be between 0 and 1")
    
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True

# ==================== HELPER FUNCTIONS ====================

def get_model_params(model_name):
    """
    Get model parameters by name
    """
    params_map = {
        'PPO': PPO_PARAMS,
        'A2C': A2C_PARAMS, 
        'DDPG': DDPG_PARAMS,
        'SAC': SAC_PARAMS
    }
    
    return params_map.get(model_name.upper(), PPO_PARAMS)

def print_config_summary():
    """
    Print configuration summary
    """
    print("📋 Configuration Summary:")
    print("=" * 50)
    print(f"Symbols: {CRYPTO_SYMBOLS}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Initial Amount: ${INITIAL_AMOUNT:,}")
    print(f"Transaction Cost: {TRANSACTION_COST_PCT*100:.2f}%")
    print(f"Training Timesteps: {TRAINING_TIMESTEPS:,}")
    print(f"Technical Indicators: {len(TECHNICAL_INDICATORS)} indicators")
    print(f"Risk Management: Max position {MAX_POSITION_SIZE*100:.0f}%, Stop loss {STOP_LOSS_PCT*100:.0f}%")
    print("=" * 50)

if __name__ == "__main__":
    # Validate and print config when run directly
    validate_config()
    print_config_summary()
