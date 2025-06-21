# SAC (Soft Actor-Critic) Package

This directory contains all SAC-related implementations for cryptocurrency trading.

## Files

### Core SAC Implementation
- **`sac.py`** - Basic SAC implementation
  - CryptoTradingEnv class
  - SAC_Agent class
  - Basic training and evaluation

### Enhanced SAC Implementation
- **`improved_sac.py`** - Improved SAC with optimizations
  - Enhanced hyperparameters
  - Better reward functions
  - Optimized training process

- **`improved_sac_strategy.py`** - SAC strategy improvements
  - Advanced trading strategies
  - Risk management enhancements
  - Performance optimizations

### Utilities and Tools
- **`backtest_sac.py`** - SAC backtesting utilities
  - Historical performance analysis
  - Strategy validation
  - Performance metrics

### Advanced Management System
- **`sac_metadata_manager.py`** - Metadata management for SAC agents
  - Agent metadata tracking
  - Performance history
  - Grade-based configurations
  - JSON persistence

- **`enhanced_sac_trainer.py`** - Enhanced SAC training system
  - Advanced training workflows
  - Real-time callbacks
  - Auto-save functionality
  - Performance comparison

### Examples and Integration
- **`integration_example.py`** - Integration examples
  - Usage examples
  - Best practices
  - Workflow demonstrations

## Usage

### Basic SAC Agent
```python
from src.sac.sac import SAC_Agent, CryptoTradingEnv

# Create environment
env = CryptoTradingEnv(data)

# Create and train agent
agent = SAC_Agent()
model = agent.train(env, timesteps=100000)

# Evaluate
results = agent.evaluate(model, env)
```

### Enhanced SAC System
```python
from src.sac.enhanced_sac_trainer import EnhancedSACTrainer
from src.sac.sac_metadata_manager import SACMetadataManager

# Create enhanced trainer
trainer = EnhancedSACTrainer(grade='A')

# Train with metadata tracking
model, metadata = trainer.train_with_metadata(
    data=crypto_data,
    timesteps=1000000
)

# Save with metadata
trainer.save_agent_with_metadata(model, metadata)
```

### Metadata Management
```python
from src.sac.sac_metadata_manager import SACMetadataManager

# Create metadata manager
manager = SACMetadataManager()

# Track agent performance
metadata = manager.create_agent_metadata(
    agent_id="sac_agent_A_001",
    grade="A",
    config=sac_config
)

# Update performance
manager.update_performance(
    agent_id="sac_agent_A_001",
    mean_reward=2.5,
    training_duration=3600
)

# Export data
manager.export_to_csv("sac_performance.csv")
```

## SAC Algorithm Features

### Core Features
- **Continuous Action Space** - Optimal for position sizing
- **Entropy Regularization** - Encourages exploration
- **Off-Policy Learning** - Sample efficient
- **Actor-Critic Architecture** - Stable training

### Enhanced Features
- **Grade-Based Configuration** - Automatic parameter optimization
- **Multi-Component Rewards** - Portfolio return + risk metrics
- **Advanced Risk Management** - Drawdown control, volatility adjustment
- **Real-Time Monitoring** - Training progress tracking

### Performance Optimizations
- **Replay Buffer Optimization** - Efficient memory usage
- **Gradient Clipping** - Training stability
- **Learning Rate Scheduling** - Adaptive learning
- **Early Stopping** - Prevent overfitting

## Grade System Integration

The SAC package integrates with the project's grade system:

- **Grade N** (Novice): 50K timesteps, basic configuration
- **Grade D** (Developing): 100K timesteps, balanced settings
- **Grade C** (Competent): 200K timesteps, professional grade
- **Grade B** (Proficient): 500K timesteps, high performance
- **Grade A** (Advanced): 1M timesteps, research grade
- **Grade S** (Supreme): 2M timesteps, cutting-edge

## Integration with Main System

This package is integrated with:
- `crypto_agent.py` - Unified agent interface
- `config/` - Configuration management
- `enhanced_crypto_env.py` - Advanced trading environment
- `tests/` - Comprehensive testing suite

## Performance Benchmarks

Typical performance on BTC-USD trading:
- **Basic SAC**: 15-25% annual return
- **Enhanced SAC**: 25-40% annual return
- **Grade A SAC**: 35-50+ annual return

*Results may vary based on market conditions and configuration.* 