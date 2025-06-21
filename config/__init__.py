# config/__init__.py - Configuration Package
"""
Configuration package for finrl_minimal_crypto

Contains all configuration files:
- config.py: Main project configuration
- algorithm_configs.py: Algorithm-specific configurations
- sac_configs.py: SAC algorithm configurations
- rl_agent_configs.py: RL agent configurations
"""

# Import main configurations
from .config import *
from .algorithm_configs import AlgorithmConfigs, get_algorithm_config, list_algorithms, recommend_algorithm

# Make commonly used items available at package level
__all__ = [
    # From config.py
    'INITIAL_AMOUNT',
    'TRANSACTION_COST_PCT', 
    'HMAX',
    'CRYPTO_SYMBOLS',
    'START_DATE',
    'END_DATE',
    'INDICATORS',
    'DATA_DIR',
    'DATA_PREPARE_DIR',
    'MODEL_DIR',
    
    # From algorithm_configs.py
    'AlgorithmConfigs',
    'get_algorithm_config',
    'list_algorithms', 
    'recommend_algorithm'
] 