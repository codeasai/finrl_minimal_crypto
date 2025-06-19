# FinRL Minimal Crypto Trading Agents

โปรเจคนี้เป็น cryptocurrency trading agents ที่ใช้ Deep Reinforcement Learning (DRL) ผ่าน FinRL library มี 3 แนวทางหลักในการ implement:

## 🚀 Implementation Approaches

### 1. **Native Python** (`main.py`)
- Core crypto trading agent implementation
- ใช้ PPO algorithm จาก Stable Baselines3
- Technical indicators พื้นฐาน (SMA, RSI, MACD, Bollinger Bands)
- รัน command line โดยตรง
- เหมาะสำหรับการพัฒนาและ debugging

### 2. **Jupyter Notebooks** (`notebooks/`)
- Interactive development environment
- Step-by-step workflow ตั้งแต่ data preparation ถึง evaluation
- เหมาะสำหรับการทดลองและ research
- มี 5 notebooks หลัก: preparation, creation, training, evaluation, implementation

### 3. **Streamlit UI** (`ui/`)
- Web-based user interface
- ง่ายต่อการใช้งาน ไม่ต้องเขียนโค้ด
- Grade system การเทรน (N, D, C, B, A, S)
- เหมาะสำหรับ end users และ production deployment

## 📁 Project Structure

```
finrl_minimal_crypto/
├── main.py                     # Core crypto agent implementation
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── models/                     # Trained models directory
│   └── trained models (.zip files)
├── data/                       # Data directory
│   └── crypto_data.csv         # Cryptocurrency data
├── notebooks/                  # Jupyter notebooks workflow
│   ├── 1_data_preparation.ipynb
│   ├── 2_agent_creation.ipynb
│   ├── 3_agent_training.ipynb
│   ├── 4_agent_evaluation.ipynb
│   ├── 5_trading_implementation.ipynb
│   ├── verification_script.py  # System verification
│   ├── config.py              # Extended configuration
│   ├── agents/                # Agent configs
│   ├── models/                # Notebook models
│   ├── data/                  # Notebook data
│   └── processed_data/        # Processed datasets
└── ui/                        # Streamlit web interface
    ├── app.py                 # Main dashboard
    ├── pages/                 # UI pages
    │   ├── 1_Data_Loader.py
    │   ├── 2_Data_Prepare.py
    │   ├── 3_Train_Agent.py
    │   ├── 4_Test_Agent.py
    │   ├── 5_Evaluate_Performance.py
    │   └── 6_Manage_Agents.py
    ├── pipeline/              # Backend logic
    └── STREAMLIT_GUIDE.md     # UI documentation
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

### วิธีที่ 1: Native Python (Command Line)
```bash
python main.py
```

### วิธีที่ 2: Jupyter Notebooks (Interactive)
```bash
jupyter notebook notebooks/
# เริ่มจาก 1_data_preparation.ipynb
```

### วิธีที่ 3: Streamlit UI (Web Interface)
```bash
cd ui
streamlit run app.py
# เปิด browser ที่ http://localhost:8501
```

## 📊 Implementation Comparison

| Approach | Ease of Use | Flexibility | Recommended For |
|----------|------------|-------------|-----------------|
| **Native Python** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Developers, Debugging |
| **Jupyter Notebooks** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **✅ Research, Learning** |
| **Streamlit UI** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **✅ End Users, Production** |

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

### ปัญหา Module Import Error
**แก้ไข:** ตรวจสอบ dependencies ใน `requirements.txt` และ environment

### ปัญหา Data Loading
**แก้ไข:** ตรวจสอบ internet connection และ yfinance API status

### ปัญหา GPU/CUDA
**แก้ไข:** Agent ทำงานได้ทั้ง CPU และ GPU โดยอัตโนมัติ

### ปัญหา Streamlit Port
**แก้ไข:** ใช้ `streamlit run app.py --port 8502` ถ้า port 8501 ถูกใช้

## 📚 Documentation

- **Installation:** ดูใน `INSTALL.md` 
- **Notebooks:** ดูใน `notebooks/` directory
- **UI Guide:** ดูใน `ui/STREAMLIT_GUIDE.md`
- **Claude Memory:** ดูใน `Claude.md`

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

**🎯 แนะนำให้เริ่มต้นด้วย Jupyter Notebooks สำหรับการเรียนรู้ หรือ Streamlit UI สำหรับการใช้งานจริง!** 