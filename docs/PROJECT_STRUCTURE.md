# Project Structure - finrl_minimal_crypto

This document describes the refactored project structure for better organization and maintainability.

## Directory Structure

```
finrl_minimal_crypto/
├── 📁 config/                    # Configuration files
│   ├── __init__.py
│   ├── README.md
│   ├── config.py                 # Main project configuration
│   ├── algorithm_configs.py      # Multi-algorithm configurations
│   ├── sac_configs.py           # SAC-specific configurations
│   └── rl_agent_configs.py      # RL agent configurations
│
├── 📁 src/                       # Source code packages
│   ├── __init__.py
│   ├── data_feature.py          # Feature engineering system
│   ├── data_loader.py           # Data loading utilities
│   └── 📁 sac/                  # SAC implementation package
│       ├── __init__.py
│       ├── README.md
│       ├── sac.py               # Basic SAC implementation
│       ├── improved_sac.py      # Enhanced SAC
│       ├── improved_sac_strategy.py
│       ├── backtest_sac.py      # Backtesting utilities
│       ├── sac_metadata_manager.py
│       ├── enhanced_sac_trainer.py
│       └── integration_example.py
│
├── 📁 tests/                     # Test files
│   ├── __init__.py
│   ├── README.md
│   ├── test_enhanced_agent_creation.py
│   ├── test_data_feature.py
│   ├── test_data_loader.py
│   ├── test_rl_agent_configs.py
│   ├── test_sac_results.py
│   ├── test_enhanced_environment.py
│   ├── test_enhanced_vs_original.py
│   └── test_enhanced_sac_system.py
│
├── 📁 docs/                      # Documentation
│   ├── NATIVE_PYTHON_README.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── TRAINING_HISTORY_GUIDE.md
│   ├── SAC_AGENT_RPG_GUIDE.md
│   ├── SAC_OPTIMIZATION_GUIDE.md
│   ├── STRATEGY_IMPROVEMENT_GUIDE.md
│   ├── MODEL_DEVELOPMENT_GUIDE.md
│   ├── FINRL_MODELS_GUIDE.md
│   ├── Claude.md
│   └── PROJECT_STRUCTURE.md     # This file
│
├── 📁 data/                      # Data storage
│   ├── raw/                     # Raw cryptocurrency data
│   ├── feature/                 # Processed feature data
│   └── data_prepare/            # Prepared data for training
│
├── 📁 models/                    # Trained models
│   ├── sac/                     # SAC model storage
│   └── [other model types]/
│
├── 📁 notebooks/                 # Jupyter notebooks
│   ├── 1_data_preparation.ipynb
│   ├── 2_agent_creation.ipynb
│   ├── 3_agent_training.ipynb
│   ├── 4_agent_evaluation.ipynb
│   ├── 5_trading_implementation.ipynb
│   └── [supporting files]
│
├── 📁 ui/                        # Streamlit web interface
│   ├── app.py
│   ├── pages/
│   ├── pipeline/
│   └── STREAMLIT_GUIDE.md
│
├── 📁 logs/                      # Log files
│
├── 📄 crypto_agent.py           # Unified agent interface
├── 📄 interactive_cli.py        # Interactive command-line interface
├── 📄 main_refactored.py        # Main entry point
├── 📄 main.py                   # Legacy main (kept for compatibility)
├── 📄 enhanced_crypto_env.py    # Enhanced trading environment
├── 📄 README.md                 # Project overview
├── 📄 INSTALL.md                # Installation instructions
├── 📄 install_talib.md          # TA-Lib installation guide
├── 📄 requirements.txt          # Python dependencies
├── 📄 requirements-dev.txt      # Development dependencies
├── 📄 environment.yml           # Conda environment
├── 📄 pyproject.toml           # Project metadata
└── 📄 .gitignore               # Git ignore patterns
```

## Key Improvements

### 1. Configuration Management
- **Before**: Config files scattered in root directory
- **After**: All configs organized in `config/` directory
- **Benefits**: 
  - Clear separation of configuration concerns
  - Easier to maintain and update
  - Package-based imports for better organization

### 2. Test Organization
- **Before**: Test files mixed with source code
- **After**: All tests in dedicated `tests/` directory
- **Benefits**:
  - Clear separation of test code
  - Easier to run test suites
  - Better test discovery and organization

### 3. SAC Implementation Consolidation
- **Before**: SAC files scattered across root and `models/sac/`
- **After**: Unified SAC package in `src/sac/`
- **Benefits**:
  - Eliminates code duplication
  - Single source of truth for SAC implementations
  - Better package organization and imports

### 4. Documentation Structure
- **Before**: Documentation files mixed with source code
- **After**: All docs in `docs/` directory (except README.md, INSTALL.md)
- **Benefits**:
  - Clean root directory
  - Organized documentation
  - Easier to find and maintain docs

## Import Changes

### Before Refactoring
```python
from config import *
from sac import SAC_Agent
from test_enhanced_agent_creation import *
```

### After Refactoring
```python
from config.config import *
from src.sac.sac import SAC_Agent
from tests.test_enhanced_agent_creation import *
```

## Package Structure

### Configuration Package (`config/`)
- Centralized configuration management
- Modular configuration files
- Package-level imports for convenience

### SAC Package (`src/sac/`)
- Complete SAC implementation suite
- Metadata management
- Enhanced training systems
- Integration examples

### Test Package (`tests/`)
- Comprehensive test coverage
- Organized by functionality
- Easy test discovery and execution

## Usage Examples

### Configuration Usage
```python
# Import specific configs
from config.config import INITIAL_AMOUNT, CRYPTO_SYMBOLS
from config.algorithm_configs import get_algorithm_config

# Get algorithm configuration
sac_config = get_algorithm_config('SAC', 'A')
```

### SAC Package Usage
```python
# Import SAC components
from src.sac.sac import SAC_Agent
from src.sac.enhanced_sac_trainer import EnhancedSACTrainer
from src.sac.sac_metadata_manager import SACMetadataManager

# Use enhanced SAC system
trainer = EnhancedSACTrainer(grade='A')
model = trainer.train(data, timesteps=1000000)
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python tests/test_enhanced_agent_creation.py

# Run tests with coverage
python -m pytest tests/ --cov=src --cov=config
```

## Migration Guide

If you have existing code that uses the old structure:

1. **Update imports**: Change `from config import *` to `from config.config import *`
2. **Update SAC imports**: Change `from sac import *` to `from src.sac.sac import *`
3. **Update test imports**: Change `from test_* import *` to `from tests.test_* import *`
4. **Update paths**: Update any hardcoded paths to reflect new structure

## Benefits of New Structure

1. **Better Organization**: Clear separation of concerns
2. **Easier Maintenance**: Related files grouped together
3. **Improved Testing**: Dedicated test directory with comprehensive coverage
4. **Cleaner Root**: Less clutter in root directory
5. **Package Management**: Proper Python package structure
6. **Documentation**: Organized documentation for better discoverability
7. **Scalability**: Structure supports future growth and new features

This refactored structure follows Python best practices and makes the project more maintainable and professional. 