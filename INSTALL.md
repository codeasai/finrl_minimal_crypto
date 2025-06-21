# 🛠️ Installation Guide - Native Python First

## 🎯 Native Python First Development

ตามกลยุทธ์ **Native Python First + Incremental Development** เราเน้นการติดตั้งที่เร็วและง่ายสำหรับการพัฒนาหลักใน `main.py` และ `sac.py`

## ⚡ Quick Start (Recommended)

### Option 1: Minimal Setup (Fastest)

```bash
# Clone repository
git clone <repository-url>
cd finrl_minimal_crypto

# Install core dependencies
pip install -r requirements.txt

# Test immediately
python sac.py  # Custom SAC agent (Primary)
python main.py # FinRL-based agent (Baseline)
```

### Option 2: Conda Environment (Stable)

```bash
# Clone repository
git clone <repository-url>
cd finrl_minimal_crypto

# Create and activate environment
conda create -n finrl_crypto python=3.9
conda activate finrl_crypto

# Install dependencies
pip install -r requirements.txt

# Test immediately
python sac.py
```

## 🎯 Development-Focused Installation

### 1. Core Development Platform

```bash
# Essential for Native Python development
pip install numpy==1.26.4
pip install pandas==2.0.3
pip install yfinance==0.2.61
pip install stable-baselines3==2.6.0
pip install gymnasium==1.1.1
pip install torch
pip install matplotlib

# Test core functionality
python -c "import stable_baselines3, gymnasium; print('✅ Core RL packages ready')"
```

### 2. FinRL Support (for main.py)

```bash
# For FinRL-based baseline
pip install finrl==0.3.7

# Test FinRL
python -c "import finrl; print('✅ FinRL ready')"
```

### 3. Development Tools (Optional)

```bash
# For enhanced development
pip install -r requirements-dev.txt

# Includes: jupyter, streamlit, plotly, seaborn
```

## 📊 System Requirements

### Minimum (for Development)
- **Python:** 3.9.x
- **RAM:** 4GB+
- **Storage:** 1GB+ free space
- **OS:** Windows/macOS/Linux

### Recommended (for Training)
- **Python:** 3.9.23 (tested)
- **RAM:** 8GB+
- **GPU:** CUDA-compatible (optional)
- **Storage:** 5GB+ free space

## 🔧 Core Dependencies (Tested Versions)

```txt
# Core RL Stack
stable-baselines3==2.6.0
gymnasium==1.1.1
torch>=1.12.0

# Data Processing
numpy==1.26.4
pandas==2.0.3
yfinance==0.2.61

# Visualization
matplotlib==3.9.4

# FinRL (for baseline)
finrl==0.3.7

# Optional Advanced Features
ta-lib  # Technical indicators
jupyter # Notebooks
streamlit # Web UI
```

## 🚀 Installation Workflow

### Step 1: Environment Setup

```bash
# Option A: Using venv (Simple)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Option B: Using conda (Recommended)
conda create -n finrl_crypto python=3.9
conda activate finrl_crypto
```

### Step 2: Core Installation

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install step by step for troubleshooting
pip install numpy==1.26.4 pandas==2.0.3
pip install stable-baselines3==2.6.0 gymnasium==1.1.1
pip install yfinance==0.2.61 matplotlib==3.9.4
pip install finrl==0.3.7
```

### Step 3: Verification

```bash
# Test Native Python development platform
python sac.py --timesteps 1000  # Quick test

# Test FinRL baseline
python main.py  # Full test

# Test core imports
python -c "
import stable_baselines3
import gymnasium  
import pandas
import numpy
import yfinance
print('✅ All core packages working!')
"
```

## 🎯 Development Environment Verification

### Quick Test Suite

```bash
# 1. Test SAC environment
python -c "
from sac import CryptoTradingEnv
import pandas as pd
print('✅ SAC environment ready')
"

# 2. Test data loading
python -c "
import yfinance as yf
df = yf.download('BTC-USD', period='5d')
print(f'✅ Data loading: {len(df)} rows')
"

# 3. Test RL training
python -c "
from stable_baselines3 import SAC
import gymnasium as gym
print('✅ RL training ready')
"
```

### Performance Test

```bash
# Quick performance test
python sac.py --timesteps 5000 --verbose

# Expected output:
# - Environment created successfully
# - SAC agent training started
# - Training completed
# - Performance metrics displayed
```

## 🔧 Optional: Advanced Features

### TA-Lib (Technical Indicators)

```bash
# Recommended: Use conda-forge
conda install ta-lib -c conda-forge

# Verify installation
python -c "import talib; print(f'✅ TA-Lib {talib.__version__}')"
```

### Jupyter Notebooks (for Phase 3)

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook notebooks/

# Test notebook integration
# Open 1_data_preparation.ipynb
```

### Streamlit UI (for Phase 3)

```bash
# Install Streamlit
pip install streamlit

# Start UI
cd ui
streamlit run app.py

# Access at http://localhost:8501
```

## 🐛 Common Issues & Quick Fixes

### Issue 1: numpy AttributeError

```bash
# Problem: numpy 2.x compatibility
# Solution: Use specific version
pip install numpy==1.26.4 --force-reinstall
```

### Issue 2: SAC Training Fails

```bash
# Problem: Environment or dependency issue
# Solution: Test step by step
python -c "from sac import CryptoTradingEnv; print('✅ Env OK')"
python -c "from stable_baselines3 import SAC; print('✅ SAC OK')"
```

### Issue 3: Data Loading Error

```bash
# Problem: yfinance API issue
# Solution: Test connection
python -c "
import yfinance as yf
try:
    df = yf.download('BTC-USD', period='1d')
    print(f'✅ Data: {len(df)} rows')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

### Issue 4: GPU/CUDA Issues

```bash
# Problem: CUDA compatibility
# Solution: Force CPU mode
python sac.py --device cpu

# Or install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 🎯 Development Tips

### Fast Iteration Setup

```bash
# Create alias for quick testing
alias sac-test="python sac.py --timesteps 5000"
alias sac-full="python sac.py --timesteps 50000"

# Quick development cycle
sac-test  # Test changes quickly
sac-full  # Full training when ready
```

### Environment Management

```bash
# Save current environment
pip freeze > requirements-current.txt

# Reset to clean state
pip install -r requirements.txt --force-reinstall

# Compare environments
diff requirements.txt requirements-current.txt
```

### Performance Monitoring

```bash
# Monitor resource usage during training
# Linux/macOS:
top -p $(pgrep -f "python sac.py")

# Windows:
# Use Task Manager or Resource Monitor
```

## 📋 Installation Checklist

### ✅ Core Development Ready
- [ ] Python 3.9.x installed
- [ ] Virtual environment created and activated
- [ ] Core dependencies installed (`pip install -r requirements.txt`)
- [ ] `python sac.py` runs successfully
- [ ] `python main.py` runs successfully

### ✅ Enhanced Development Ready (Optional)
- [ ] TA-Lib installed for advanced indicators
- [ ] Jupyter installed for notebook workflow
- [ ] Streamlit installed for web UI

### ✅ Production Ready (Future)
- [ ] All tests pass
- [ ] Performance benchmarks completed
- [ ] Documentation updated

## 🎯 Next Steps After Installation

1. **Start Development**: `python sac.py`
2. **Test Baseline**: `python main.py`
3. **Compare Performance**: `python test_enhanced_vs_original.py`
4. **Begin Incremental Improvements**: Edit reward function in `sac.py`

## 📞 Support

### Quick Diagnostics

```bash
# Generate system report
python -c "
import sys, platform, torch
print(f'Python: {sys.version}')
print(f'Platform: {platform.system()} {platform.release()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"
```

### Common Solutions

- **Import Errors**: `pip install -r requirements.txt --force-reinstall`
- **Performance Issues**: Use `--timesteps 5000` for testing
- **Memory Issues**: Close other applications, use CPU mode
- **Network Issues**: Check internet connection for data downloads

---

**🎯 Goal: Get `python sac.py` running in under 5 minutes!**
