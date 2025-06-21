# 📦 Source Package (src/)

This directory contains the main source code packages for finrl_minimal_crypto project, organized following Python best practices.

## 📁 Package Structure

```
src/
├── 📁 agents/                    # RL Agent Implementations
│   ├── __init__.py
│   └── crypto_agent.py          # Unified SAC Agent with Grade System
│
├── 📁 cli/                       # Command-Line Interface
│   ├── __init__.py
│   └── interactive_cli.py       # Interactive CLI with Agent Management
│
├── 📁 environments/              # Trading Environments
│   ├── __init__.py
│   └── enhanced_crypto_env.py   # Enhanced Trading Environment
│
├── 📁 sac/                       # SAC-Specific Implementations
│   ├── __init__.py
│   ├── README.md
│   ├── sac.py                   # Basic SAC implementation
│   ├── improved_sac.py          # Enhanced SAC
│   ├── enhanced_sac_trainer.py  # Advanced training system
│   ├── sac_metadata_manager.py  # Metadata management
│   ├── backtest_sac.py          # Backtesting utilities
│   ├── improved_sac_strategy.py # Strategy improvements
│   └── integration_example.py   # Usage examples
│
├── 📄 data_feature.py            # Feature Engineering System
├── 📄 data_loader.py             # Data Loading Utilities
├── 📄 __init__.py                # Package initialization
└── 📄 README.md                  # This file
```

## 🎯 Package Overview

### 🤖 Agents Package (`agents/`)
Contains all reinforcement learning agent implementations:
- **Unified SAC Agent**: Grade-based configuration system (N, D, C, B, A, S)
- **Metadata Tracking**: Performance monitoring and persistence
- **Multi-Algorithm Support**: SAC, PPO, DDPG, TD3, A2C (extensible)

### 🎮 CLI Package (`cli/`)
Command-line interface components:
- **Interactive CLI**: Menu-driven interface for agent management
- **Agent Manager**: Multi-agent creation, loading, and comparison
- **Data Manager**: Cryptocurrency data loading and processing

### 🏗️ Environments Package (`environments/`)
Trading environment implementations:
- **Enhanced Environment**: Multi-component rewards, risk management
- **Grade Integration**: Environment features based on agent grade
- **Advanced Features**: Portfolio management, benchmark comparison

### 🧠 SAC Package (`sac/`)
SAC algorithm specific implementations:
- **Multiple SAC Variants**: Basic, improved, and enhanced versions
- **Metadata Management**: Training history and performance tracking
- **Advanced Training**: Callbacks, auto-save, comparison tools

## 📚 Usage Examples

### Basic Agent Creation
```python
from src.agents import create_crypto_sac_agent

# Create Grade A agent
agent = create_crypto_sac_agent(grade='A')
print(f"Agent: {agent.agent_id}")
print(f"Timesteps: {agent.config['total_timesteps']:,}")
```

### Interactive CLI
```python
from src.cli import InteractiveCLI

# Start interactive interface
cli = InteractiveCLI()
cli.main_menu()
```

### Enhanced Environment
```python
from src.environments import EnhancedCryptoTradingEnv
import pandas as pd

# Create enhanced environment
data = pd.read_csv('crypto_data.csv')
env = EnhancedCryptoTradingEnv(data)

# Get trading statistics
stats = env.get_trading_statistics()
print(f"Total Return: {stats['total_return']:.2f}%")
```

### Data Processing
```python
from src import download_crypto_data, process_crypto_features

# Download and process data
data = download_crypto_data(['BTC-USD'], '2023-01-01', '2024-01-01')
features = process_crypto_features(data, 'BTC-USD')
print(f"Features: {len(features.columns)}")
```

## 🔄 Integration with Main System

This package structure integrates seamlessly with:
- **`main.py`**: Unified entry point with command-line arguments
- **`config/`**: Configuration management system
- **`tests/`**: Comprehensive testing suite
- **`notebooks/`**: Jupyter notebook workflows
- **`ui/`**: Streamlit web interface

## 🎯 Benefits of This Structure

1. **Separation of Concerns**: Each package has a clear responsibility
2. **Modularity**: Components can be imported and used independently
3. **Extensibility**: Easy to add new agents, environments, or CLI features
4. **Maintainability**: Clear organization makes code easier to maintain
5. **Testing**: Package structure supports comprehensive unit testing
6. **Documentation**: Each package has its own documentation
7. **Python Best Practices**: Follows PEP 8 and standard Python conventions

## 🚀 Development Workflow

### Adding New Agent
1. Create new agent class in `src/agents/`
2. Update `src/agents/__init__.py` to export new agent
3. Add tests in `tests/test_agents.py`
4. Update main.py to support new agent type

### Adding New Environment
1. Create environment in `src/environments/`
2. Update `src/environments/__init__.py`
3. Add integration tests
4. Update CLI to support new environment

### Adding CLI Features
1. Extend `InteractiveCLI` class in `src/cli/`
2. Add new menu options and workflows
3. Test new features
4. Update documentation

## 📋 Import Guide

### From Root Directory
```python
# Import agents
from src.agents import CryptoSACAgent, create_crypto_sac_agent

# Import CLI
from src.cli import InteractiveCLI

# Import environments
from src.environments import EnhancedCryptoTradingEnv

# Import data utilities
from src import download_crypto_data, process_crypto_features
```

### From Package Level
```python
# Import specific components
from src.agents.crypto_agent import CryptoSACAgent
from src.cli.interactive_cli import AgentManager
from src.environments.enhanced_crypto_env import EnhancedCryptoTradingEnv
```

This organized structure supports the project's growth while maintaining clean, maintainable, and testable code. 