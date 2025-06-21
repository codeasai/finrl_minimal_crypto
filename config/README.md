# Configuration Directory

This directory contains all configuration files for the finrl_minimal_crypto project.

## Files

### Core Configuration
- **`config.py`** - Main project configuration
  - Trading parameters (initial amount, transaction costs, etc.)
  - Cryptocurrency symbols
  - Technical indicators
  - Directory structure
  - Date ranges

### Algorithm Configuration
- **`algorithm_configs.py`** - Multi-algorithm configuration system
  - SAC, PPO, DDPG, TD3, A2C configurations
  - Grade-based parameter optimization (N, D, C, B, A, S)
  - Algorithm recommendations and comparisons
  - Dynamic configuration generation

### SAC-Specific Configuration
- **`sac_configs.py`** - SAC algorithm specific configurations
  - Grade-based SAC parameters
  - Performance optimizations
  - Buffer sizes and learning rates

### RL Agent Configuration
- **`rl_agent_configs.py`** - General RL agent configurations
  - Agent creation parameters
  - Training configurations
  - Evaluation settings

## Usage

```python
# Import main configuration
from config.config import INITIAL_AMOUNT, CRYPTO_SYMBOLS, INDICATORS

# Import algorithm configurations
from config.algorithm_configs import get_algorithm_config, AlgorithmConfigs

# Get SAC Grade A configuration
config = get_algorithm_config('SAC', 'A')
print(f"Timesteps: {config['total_timesteps']:,}")
print(f"Buffer Size: {config['default_params']['buffer_size']:,}")

# Get algorithm recommendations
recommended = AlgorithmConfigs.recommend_algorithm('crypto_trading')
print(f"Recommended algorithm: {recommended}")
```

## Configuration Hierarchy

```
config/
├── config.py              # Base configuration
├── algorithm_configs.py    # Algorithm-specific configs
├── sac_configs.py         # SAC-specific configs
└── rl_agent_configs.py    # RL agent configs
```

All configuration files are designed to be modular and can be imported independently or together through the package interface. 