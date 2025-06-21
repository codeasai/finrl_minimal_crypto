# src/__init__.py - Source Package Initialization
"""
Source package สำหรับ finrl_minimal_crypto project

Packages:
- agents: RL agent implementations (SAC, PPO, etc.)
- environments: Trading environment implementations
- cli: Command-line interface components
- sac: SAC-specific implementations
- data_loader: Advanced Yahoo Finance data loader with caching and validation
- data_feature: Advanced technical indicators feature engineering
"""

# Import data utilities
from .data_loader import (
    YahooDataLoader,
    download_crypto_data,
    load_crypto_data,
    list_available_crypto_data,
    get_crypto_data_summary
)

from .data_feature import (
    CryptoFeatureProcessor,
    process_crypto_features,
    process_all_crypto_features,
    load_crypto_features,
    get_crypto_feature_summary
)

# Import agent components
from .agents import (
    CryptoSACAgent,
    create_crypto_sac_agent,
    load_crypto_sac_agent
)

# Import environment components
from .environments import (
    EnhancedCryptoTradingEnv,
    create_enhanced_environment
)

# Import CLI components
from .cli import (
    InteractiveCLI,
    AgentManager,
    DataManager
)

__version__ = "1.0.0"
__author__ = "finrl_minimal_crypto"

__all__ = [
    # Data Utilities
    'YahooDataLoader',
    'download_crypto_data',
    'load_crypto_data',
    'list_available_crypto_data',
    'get_crypto_data_summary',
    'CryptoFeatureProcessor',
    'process_crypto_features',
    'process_all_crypto_features',
    'load_crypto_features',
    'get_crypto_feature_summary',
    
    # Agent Components
    'CryptoSACAgent',
    'create_crypto_sac_agent',
    'load_crypto_sac_agent',
    
    # Environment Components
    'EnhancedCryptoTradingEnv',
    'create_enhanced_environment',
    
    # CLI Components
    'InteractiveCLI',
    'AgentManager',
    'DataManager'
] 