# Native Python SAC Agent - Refactored Implementation

> 🚀 **Unified SAC Agent Implementation with Grade System Integration**

## 📋 Overview

This is the refactored Native Python implementation of the Crypto SAC Agent, designed following the **Native Python First** development strategy. The new implementation provides a unified, modular, and feature-rich platform for cryptocurrency trading using Deep Reinforcement Learning.

## 🎯 Key Features

### ✨ **Unified Architecture**
- **Single Source of Truth**: One comprehensive SAC agent implementation
- **Grade System Integration**: N, D, C, B, A, S performance tiers
- **Modular Design**: Clean separation of concerns
- **Enhanced Environment**: Grade-based trading features

### 🎮 **Multiple Interface Options**
- **Interactive CLI**: Menu-driven interface for all operations
- **Command-Line Modes**: Direct training, testing, and comparison
- **Argument Parsing**: Flexible parameter configuration
- **Batch Operations**: Automated workflows

### 📊 **Advanced Features**
- **Metadata Tracking**: Real-time performance monitoring
- **Agent Management**: Multi-agent comparison and analysis
- **Data Pipeline**: Automated data loading and preprocessing
- **Performance Analytics**: Comprehensive evaluation metrics

## 🚀 Quick Start

### 1. Interactive Mode (Recommended)
```bash
# Start interactive CLI
python main_refactored.py

# Or use the interactive CLI directly
python interactive_cli.py
```

### 2. Direct Training
```bash
# Train a Grade C agent (default)
python main_refactored.py --mode train

# Train a Grade B agent with custom timesteps
python main_refactored.py --mode train --grade B --timesteps 300000

# Force download fresh data
python main_refactored.py --mode train --grade A --force-download
```

### 3. Testing Agents
```bash
# Test a specific agent
python main_refactored.py --mode test --agent-id sac_agent_20250101_120000

# Test with custom episodes
python main_refactored.py --mode test --agent-id ABC123 --episodes 20
```

### 4. Compare Agents
```bash
# Compare all saved agents
python main_refactored.py --mode compare

# View system information
python main_refactored.py --mode info
```

## 📁 File Structure

```
├── crypto_agent.py           # 🤖 Unified SAC Agent Implementation
├── interactive_cli.py        # 🎮 Interactive Command-Line Interface
├── main_refactored.py       # 🚀 Unified Entry Point
├── config.py                # ⚙️ Configuration Settings
├── sac_configs.py           # 📊 Grade-based Configurations
└── models/sac/              # 💾 Saved Agent Models
```

## 🎯 Grade System

| Grade | Description | Timesteps | Buffer Size | Use Case |
|-------|-------------|-----------|-------------|----------|
| **N** | Novice | 50K | 50K | Learning & Testing |
| **D** | Developing | 100K | 100K | Basic Trading |
| **C** | Competent | 200K | 250K | Professional Setup |
| **B** | Proficient | 500K | 500K | High Performance |
| **A** | Advanced | 1M | 1M | Research Grade |
| **S** | Supreme | 2M | 2M | State-of-the-art |

## 🎮 Interactive CLI Features

### 📋 Main Menu
1. **🆕 Create New Agent** - Grade selection with custom configuration
2. **📊 Load Existing Agent** - Browse and load saved agents
3. **📈 Load/Download Data** - Data management with technical indicators
4. **🏋️ Train Current Agent** - Training with real-time progress
5. **🧪 Test Current Agent** - Performance evaluation
6. **📊 View Agent Performance** - Detailed metrics and analysis
7. **🔍 Compare Agents** - Multi-agent comparison table
8. **💾 Save Current Agent** - Model persistence with metadata
9. **⚙️ Agent Settings** - System configuration and information

### 🔧 Agent Management
- **Multi-Agent Support**: Create, load, and manage multiple agents
- **Metadata Tracking**: Performance history and training statistics
- **Grade-based Configuration**: Automatic parameter optimization
- **Intelligent Comparison**: Performance ranking and analysis

### 📊 Data Management
- **Automatic Downloads**: Yahoo Finance integration
- **Technical Indicators**: 12+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Data Validation**: Comprehensive error handling
- **Caching System**: Efficient data reuse

## 🏗️ Core Components

### 🤖 CryptoSACAgent Class

```python
from crypto_agent import create_crypto_sac_agent

# Create agent with grade system
agent = create_crypto_sac_agent(grade='B')

# Load data and create environment
data = load_crypto_data()
train_env, test_env = agent.create_environment(data)

# Train agent
agent.train(timesteps=100000)

# Evaluate performance
results = agent.evaluate(n_episodes=10)

# Save agent
agent.save()
```

### 🎮 Interactive CLI Usage

```python
from interactive_cli import InteractiveCLI

# Start interactive mode
cli = InteractiveCLI()
cli.main_menu()
```

### 📊 Agent Management

```python
from interactive_cli import AgentManager

# Manage multiple agents
manager = AgentManager()
agents = manager.list_available_agents()
agent = manager.load_agent('agent_name')
```

## ⚙️ Configuration

### 🎯 Grade-based Configuration
Each grade automatically configures:
- **Timesteps**: Training duration
- **Buffer Size**: Experience replay capacity  
- **Batch Size**: Learning batch size
- **Learning Rate**: Optimization rate
- **Environment Features**: Trading complexity

### 🔧 Custom Configuration
```python
# Custom configuration example
custom_config = {
    'total_timesteps': 150000,
    'buffer_size': 200000,
    'learning_rate': 1e-4,
    'batch_size': 512
}

agent = create_crypto_sac_agent(grade='C', config=custom_config)
```

## 📊 Performance Tracking

### 📈 Metadata System
- **Training History**: Step-by-step progress tracking
- **Performance Metrics**: Comprehensive evaluation statistics
- **System Information**: Hardware and environment details
- **Comparison Analytics**: Multi-agent performance analysis

### 🎯 Evaluation Metrics
- **Mean Reward**: Average performance across episodes
- **Standard Deviation**: Performance consistency
- **Best/Worst Rewards**: Performance range
- **Episode Length**: Trading session duration
- **Sharpe Ratio**: Risk-adjusted returns (Grade A/S)

## 🚀 Advanced Usage

### 🔄 Batch Training
```bash
# Train multiple grades sequentially
for grade in N D C B A S; do
    python main_refactored.py --mode train --grade $grade
done
```

### 📊 Performance Analysis
```bash
# Compare all agents and export results
python main_refactored.py --mode compare > agent_comparison.txt
```

### 🎯 Automated Testing
```bash
# Test all agents with custom episodes
python -c "
from interactive_cli import AgentManager
manager = AgentManager()
for agent in manager.list_available_agents():
    print(f'Testing {agent[\"name\"]}...')
    # Add testing logic here
"
```

## 🔧 Troubleshooting

### ❌ Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct environment
   conda activate tfyf
   ```

2. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python main_refactored.py --mode info
   ```

3. **Data Download Failures**
   ```bash
   # Force fresh data download
   python main_refactored.py --mode train --force-download
   ```

4. **Agent Loading Issues**
   ```bash
   # List available agents
   python main_refactored.py --mode compare
   ```

### 🔍 Debug Mode
```bash
# Increase verbosity for debugging
python main_refactored.py --mode train --grade C -vvv

# Suppress output for automation
python main_refactored.py --mode train --grade C --quiet
```

## 🎯 Migration from Legacy

### 📋 Migration Steps
1. **Backup Existing Models**: Copy `models/` directory
2. **Update Imports**: Use new unified components
3. **Test Compatibility**: Verify agent loading
4. **Gradual Migration**: Move workflows incrementally

### 🔄 Legacy Support
- **Existing Models**: Compatible with new system
- **Configuration Files**: Automatic migration
- **Data Formats**: Backward compatibility maintained

## 🚀 Future Enhancements

### 📈 Planned Features
- **Ensemble Methods**: Multi-agent coordination
- **Live Trading**: Real-time market integration
- **Advanced Analytics**: Performance dashboards
- **API Integration**: RESTful service interface
- **Multi-Asset Support**: Portfolio optimization
- **Risk Management**: Advanced position sizing

### 🎯 Development Roadmap
- **Phase 3**: Advanced Integration (In Progress)
- **Phase 4**: Polish & Optimization
- **Phase 5**: Production Deployment
- **Phase 6**: Advanced Features

## 📚 Additional Resources

- **[SAC Optimization Guide](SAC_OPTIMIZATION_GUIDE.md)**: Performance tuning
- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)**: Technical details
- **[Training History Guide](TRAINING_HISTORY_GUIDE.md)**: Performance analysis
- **[SAC Agent RPG Guide](SAC_AGENT_RPG_GUIDE.md)**: Gamified learning

## 🤝 Contributing

1. **Follow Native Python First**: Prioritize this implementation
2. **Incremental Development**: Small, testable changes
3. **Comprehensive Testing**: Validate all functionality
4. **Documentation**: Update guides and examples

## 📝 License

This project follows the same license as the main finrl_minimal_crypto project.

---

**🎉 Happy Trading with SAC Agents! 🚀** 