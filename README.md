# FinRL Minimal Crypto Trading Agents

โปรเจคนี้เป็น cryptocurrency trading agents ที่ใช้ Deep Reinforcement Learning (DRL) ผ่าน FinRL library มี 3 main features หลัก:

## 🚀 Main Features

### 1. **Basic Crypto Agent** (`main.py`)
- Agent พื้นฐานสำหรับ crypto trading
- ใช้ PPO algorithm
- Technical indicators พื้นฐาน (SMA, RSI, MACD, Bollinger Bands)
- เหมาะสำหรับผู้เริ่มต้น

### 2. **Simple Advanced Agent** (`simple_advanced_agent.py`)
- Advanced agent แบบง่าย ที่แก้ปัญหา AttributeError
- ใช้หลักการจาก main.py ที่ทำงานได้เรียบร้อย
- Technical indicators ครบครัน (11 indicators)
- ประสิทธิภาพดีกว่า basic agent

### 3. **Full Advanced Agent** (`advanced_crypto_agent.py`)
- Advanced agent แบบเต็มรูปแบบ
- Technical indicators มากกว่า 40 ตัว
- Cross-asset features และ market regime analysis
- Ensemble models และ advanced risk management

## 📁 Project Structure

```
finrl_minimal_crypto/
├── main.py                     # Basic crypto agent
├── simple_advanced_agent.py    # Simple advanced agent (แนะนำ)
├── advanced_crypto_agent.py    # Full advanced agent
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── models/                     # Trained models directory
│   ├── minimal_crypto_ppo.zip          # Model จาก main.py
│   ├── simple_advanced_crypto_ppo.zip  # Model จาก simple_advanced_agent.py
│   └── performance_analysis.png        # Performance charts
├── data/                       # Data directory
│   ├── crypto_data.csv                 # Data สำหรับ main.py
│   ├── advanced_crypto_data.csv        # Data สำหรับ advanced_crypto_agent.py
│   └── simple_advanced_crypto_data.csv # Data เพิ่มเติม
├── simple_data/               # Data สำหรับ simple_advanced_agent.py
│   └── simple_crypto_data.csv
├── notebooks/                 # Jupyter notebooks
│   ├── 1_data_preparation.ipynb
│   ├── 2_agent_creation.ipynb
│   ├── 3_agent_training.ipynb
│   ├── 4_agent_evaluation.ipynb
│   └── 5_trading_implementation.ipynb
└── ui/                        # Streamlit UI
    ├── app.py
    └── pipeline/
        ├── data_loader.py
        ├── train.py
        ├── evaluate.py
        └── agent_manager.py
```

## 🛠️ Installation

### Quick Install
```bash
git clone <repository-url>
cd finrl_minimal_crypto
pip install -r requirements.txt
```

### Detailed Installation
📖 **ดูคำแนะนำการติดตั้งแบบละเอียดใน [INSTALL.md](INSTALL.md)**

- ✅ Tested package versions
- 🐛 Common issues & solutions  
- 🔧 Development setup
- 📋 System requirements

## 🚀 Quick Start

### วิธีที่ 1: Basic Agent (เริ่มต้น)
```bash
python main.py
```

### วิธีที่ 2: Simple Advanced Agent (แนะนำ)
```bash
python simple_advanced_agent.py
```

### วิธีที่ 3: Full Advanced Agent (ขั้นสูง)
```bash
python advanced_crypto_agent.py
```

### วิธีที่ 4: Streamlit UI
```bash
cd ui
streamlit run app.py
```

### วิธีที่ 5: Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 📊 Performance Comparison

| Agent Type | Return | Sharpe Ratio | Complexity | Recommended |
|------------|--------|--------------|------------|-------------|
| Basic | ~0% | 0.002 | ⭐ | Beginners |
| Simple Advanced | ~10.81% | 0.653 | ⭐⭐ | **✅ Most Users** |
| Full Advanced | Variable | Variable | ⭐⭐⭐⭐⭐ | Researchers |

## 🔧 Configuration

แก้ไขไฟล์ `config.py`:

```python
# Crypto symbols to trade
CRYPTO_SYMBOLS = ['BTC-USD']

# Trading parameters
INITIAL_AMOUNT = 100000
HMAX = 100
TRANSACTION_COST_PCT = 0.001

# Training parameters
TOTAL_TIMESTEPS = 100000
```

## 📈 Features

### Technical Indicators
- **Basic:** SMA, RSI, MACD, Bollinger Bands
- **Advanced:** 40+ indicators including ADX, CCI, Stochastic, ATR
- **Cross-asset:** BTC correlation, Beta analysis
- **Market Regime:** Trend strength, Volatility regime

### Machine Learning
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Environment:** FinRL StockTradingEnv
- **Reward:** Portfolio return with risk adjustment
- **Features:** Technical indicators + market data

### Risk Management
- **Transaction Costs:** 0.1% default
- **Position Sizing:** Maximum 100 shares per asset
- **Drawdown Control:** Built-in risk metrics
- **Diversification:** Multi-asset portfolio

## 🐛 Troubleshooting

### ปัญหา AttributeError: 'numpy.float64' object has no attribute 'values'
**แก้ไข:** ใช้ `simple_advanced_agent.py` แทน `advanced_crypto_agent.py`

### ปัญหา TA-Lib installation
**แก้ไข:** ดูคำแนะนำใน `install_talib.md`

### ปัญหา GPU/CUDA
**แก้ไข:** Agents ทำงานได้ทั้ง CPU และ GPU

## 📚 Documentation

- **Notebooks:** ดูใน `notebooks/` directory
- **Advanced Features:** ดูใน `README_advanced_agent.md`
- **UI Guide:** ดูใน `ui/` directory

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Financial Reinforcement Learning
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL Algorithms
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical Analysis Library

---

**🎯 แนะนำให้เริ่มต้นด้วย `simple_advanced_agent.py` เพราะทำงานได้เสถียรและให้ผลลัพธ์ดี!** 