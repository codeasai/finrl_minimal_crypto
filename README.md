# FinRL Minimal Crypto Trading Agents

โปรเจคนี้เป็น cryptocurrency trading agents ที่ใช้ Deep Reinforcement Learning (DRL) โดยเน้น **Native Python First Development** กับการพัฒนาแบบ incremental

## 🎯 Development Philosophy

### **Native Python First + Incremental Development**
- 🚀 **Core Platform**: `main.py` และ `sac.py` เป็นหลักในการพัฒนา
- 📈 **Incremental**: พัฒนาทีละน้อย test ทีละ step
- ⚡ **Quick Iteration**: Fast feedback loop สำหรับการทดลอง
- 🎯 **Focus**: หลีกเลี่ยง complex systems ที่ไม่จำเป็น

### **Development Flow**
```
Native Python → Test & Validate → Notebooks → Streamlit UI → Production
```

## 🚀 Getting Started (Recommended Path)

### 1. **Quick Start - Native Python** 
```bash
# Clone และติดตั้ง
git clone <repository-url>
cd finrl_minimal_crypto
pip install -r requirements.txt

# รันทันที - FinRL-based agent
python main.py

# หรือ SAC agent (Custom implementation)
python sac.py
```

### 2. **การพัฒนาแบบ Incremental**
```bash
# 1. ทดลองกับ Native Python ก่อน
python sac.py  # Test basic SAC

# 2. ปรับปรุงทีละน้อย
# - แก้ reward function
# - ปรับ hyperparameters  
# - เพิ่ม technical indicators

# 3. Test ทุกการเปลี่ยนแปลง
python sac.py  # Validate improvements

# 4. เมื่อ stable แล้วค่อยขยายไป platforms อื่น
```

## 📁 Project Structure (Development-Focused)

```
finrl_minimal_crypto/
├── 🎯 CORE DEVELOPMENT PLATFORM
│   ├── main.py                 # FinRL-based agent (stable)
│   ├── sac.py                  # SAC agent (active development)
│   ├── config.py               # Shared configuration
│   └── enhanced_crypto_env.py  # Custom environment
│
├── 🧪 DEVELOPMENT SUPPORT
│   ├── improved_sac.py         # Enhanced SAC implementation
│   ├── sac_configs.py          # Grade-based configurations
│   ├── test_*.py               # Testing scripts
│   └── backtest_sac.py         # Backtesting tools
│
├── 📊 DATA & MODELS
│   ├── data/                   # Cryptocurrency data
│   ├── models/                 # Trained models
│   └── logs/                   # Training logs
│
├── 📚 INTEGRATION PLATFORMS (Secondary)
│   ├── notebooks/              # Jupyter workflow
│   │   ├── 1_data_preparation.ipynb
│   │   ├── 2_agent_creation.ipynb
│   │   ├── 3_agent_training.ipynb
│   │   ├── 4_agent_evaluation.ipynb
│   │   └── 5_trading_implementation.ipynb
│   │
│   └── ui/                     # Streamlit interface
│       ├── app.py              # Dashboard
│       └── pages/              # UI pages
│
└── 📖 DOCUMENTATION
    ├── README.md               # This file
    ├── INSTALL.md              # Installation guide
    ├── SAC_OPTIMIZATION_GUIDE.md
    ├── IMPLEMENTATION_GUIDE.md
    └── STRATEGY_IMPROVEMENT_GUIDE.md
```

## 🛠️ Installation

### Quick Install (Recommended)
```bash
pip install -r requirements.txt
```

### Detailed Installation
📖 **ดูคำแนะนำการติดตั้งแบบละเอียดใน [INSTALL.md](INSTALL.md)**

## 🎯 Core Development Platforms

### 1. **main.py** - FinRL-based Agent
```bash
python main.py
```
- ✅ **Stable**: ใช้ FinRL framework
- 🔧 **Algorithm**: PPO (default) หรือ SAC
- 📊 **Features**: Technical indicators, GPU/CPU auto-detection
- 🎯 **Use Case**: Baseline comparison, stable reference

### 2. **sac.py** - Custom SAC Agent
```bash
python sac.py
```
- 🚀 **Active Development**: Custom implementation
- 🔧 **Algorithm**: SAC (Soft Actor-Critic)
- ⚡ **Performance**: Optimized for crypto trading
- 🎯 **Use Case**: Main development platform

## 📊 Algorithm Comparison

| Algorithm | Implementation | Status | Performance | Use Case |
|-----------|---------------|---------|-------------|----------|
| **PPO** | FinRL-based | ✅ Stable | Baseline | Reference |
| **SAC** | Custom | 🚀 Active | **Optimized** | **Primary** |

## 🔧 Configuration

แก้ไขไฟล์ `config.py`:

```python
# Core settings
CRYPTO_SYMBOLS = ['BTC-USD']
INITIAL_AMOUNT = 100000
TRANSACTION_COST_PCT = 0.001

# SAC-specific settings  
SAC_TIMESTEPS = 50000
SAC_BUFFER_SIZE = 100000
SAC_LEARNING_RATE = 3e-4

# Technical indicators (12 indicators)
INDICATORS = [
    'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal', 
    'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower', 
    'bb_std', 'volume_sma', 'volume_ratio'
]
```

## 📈 Development Workflow

### Phase 1: Native Python Development
```bash
# 1. Start with basic SAC
python sac.py

# 2. Incremental improvements
# - Modify reward function in sac.py
# - Adjust hyperparameters
# - Add new technical indicators

# 3. Test each change
python sac.py  # Validate improvement

# 4. Backtest performance
python backtest_sac.py
```

### Phase 2: Enhanced Features
```bash
# 1. Test enhanced environment
python test_enhanced_environment.py

# 2. Compare implementations
python test_enhanced_vs_original.py

# 3. Use improved SAC
python improved_sac.py
```

### Phase 3: Integration (When Native Python is stable)
```bash
# 1. Update notebooks
jupyter notebook notebooks/

# 2. Update Streamlit UI
cd ui && streamlit run app.py
```

## 🎯 Quick Development Tips

### ⚡ Fast Iteration
```bash
# Quick test with minimal timesteps
python sac.py --timesteps 5000

# Quick backtest
python backtest_sac.py --days 30
```

### 🔍 Debugging
```bash
# Enable verbose logging
python sac.py --verbose

# Test environment only
python test_enhanced_environment.py
```

### 📊 Performance Testing
```bash
# Compare SAC vs PPO
python test_enhanced_vs_original.py

# Detailed SAC analysis
python test_sac_results.py
```

## 📚 Documentation

### 🎯 Core Guides
- **[INSTALL.md](INSTALL.md)** - Installation & setup
- **[SAC_OPTIMIZATION_GUIDE.md](SAC_OPTIMIZATION_GUIDE.md)** - SAC optimization
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Implementation steps
- **[STRATEGY_IMPROVEMENT_GUIDE.md](STRATEGY_IMPROVEMENT_GUIDE.md)** - Strategy improvements

### 📖 Reference
- **[FINRL_MODELS_GUIDE.md](FINRL_MODELS_GUIDE.md)** - RL algorithms overview
- **[Claude.md](Claude.md)** - Development workflow with Claude

## 🐛 Troubleshooting

### Common Issues
- **Module Import Error**: ตรวจสอบ `pip install -r requirements.txt`
- **Data Loading**: ตรวจสอบ internet connection
- **GPU Issues**: Agent ทำงานทั้ง CPU/GPU อัตโนมัติ
- **Performance Issues**: ลด timesteps สำหรับการทดสอบ

### Quick Fixes
```bash
# Reset environment
pip install -r requirements.txt --force-reinstall

# Test basic functionality
python -c "import pandas, numpy, yfinance; print('✅ Core packages OK')"

# Verify SAC implementation
python test_enhanced_environment.py
```

## 🎯 Next Steps

1. **เริ่มต้น**: `python sac.py`
2. **ปรับปรุง**: แก้ reward function ใน `sac.py`
3. **ทดสอบ**: `python backtest_sac.py`
4. **เปรียบเทียบ**: `python test_enhanced_vs_original.py`
5. **ขยายผล**: อัพเดท notebooks และ UI

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/sac-improvement`)
3. **Test in Native Python first** (`python sac.py`)
4. Commit changes (`git commit -m 'Improve SAC reward function'`)
5. Push to branch (`git push origin feature/sac-improvement`)
6. Open Pull Request

---

**🎯 เริ่มต้นด้วย: `python sac.py` สำหรับการพัฒนาหลัก!** 