# src/sac/__init__.py - SAC Package
"""
SAC (Soft Actor-Critic) package for finrl_minimal_crypto

Contains all SAC-related implementations:
- sac.py: Basic SAC implementation
- improved_sac.py: Improved SAC with optimizations
- improved_sac_strategy.py: SAC strategy improvements
- backtest_sac.py: SAC backtesting utilities
- sac_metadata_manager.py: Metadata management for SAC agents
- enhanced_sac_trainer.py: Enhanced SAC training system
- integration_example.py: Integration examples
"""

# Import main SAC components
try:
    from .sac import CryptoTradingEnv, SAC_Agent
except ImportError:
    pass

try:
    from .improved_sac import ImprovedSACAgent
except ImportError:
    pass

try:
    from .sac_metadata_manager import SACMetadataManager
except ImportError:
    pass

try:
    from .enhanced_sac_trainer import EnhancedSACTrainer
except ImportError:
    pass

# Package information
__version__ = "1.0.0"
__author__ = "finrl_minimal_crypto"
__description__ = "SAC implementation for cryptocurrency trading"

# Available components
AVAILABLE_COMPONENTS = [
    'CryptoTradingEnv',
    'SAC_Agent', 
    'ImprovedSACAgent',
    'SACMetadataManager',
    'EnhancedSACTrainer'
] 