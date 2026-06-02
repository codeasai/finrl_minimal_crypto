# FinRL Minimal Crypto Trading Agents

โปรเจคนี้เป็น cryptocurrency trading agents ที่ใช้ Deep Reinforcement Learning (DRL) ผ่าน FinRL library มี 3 main features หลัก:

## 🚀 Main Features

### 1. **Streamlit Web UI** (แนะนำ)
- Pipeline ครบ: Load → Prepare → Train → Test → Evaluate → Manage
- เทรน PPO agent จริงในเบราว์เซอร์ พร้อม progress tracker
- รองรับหลาย exchange (Binance, Bybit, OKX) ผ่าน CCXT
- เลือก Training Grade (N–S) ตามเวลาและทรัพยากรที่มี
- บันทึก metadata (exchange, grade, symbols) ควบคู่ model

### 2. **Basic Crypto Agent** (`main.py`)
- Agent พื้นฐาน ดาวน์โหลดข้อมูลจาก yfinance อัตโนมัติ
- ใช้ PPO algorithm พร้อม 12 technical indicators
- เหมาะสำหรับทดสอบ pipeline หรือผู้เริ่มต้น

### 3. **Simple Advanced Agent** (`simple_advanced_agent.py`)
- Multi-symbol, รองรับ TA-Lib (fallback manual ถ้าไม่มี)
- Technical indicators ครบครัน (~30 indicators)
- ประสิทธิภาพดีกว่า basic agent

## 📁 Project Structure

```
finrl_minimal_crypto/
├── main.py                     # Basic crypto agent (CLI, yfinance)
├── simple_advanced_agent.py    # Advanced agent (CLI, multi-symbol)
├── advanced_crypto_agent.py    # Full advanced agent (CLI, TA-Lib)
├── config.py                   # Configuration (symbols, amounts, directories)
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── data/                       # Raw + prepared data
│   ├── *.csv                            # Raw OHLCV จาก Data Loader
│   └── data_prepare/
│       └── *.csv                        # ข้อมูลที่เพิ่ม indicators แล้ว (ใช้เทรน)
├── models/                     # Trained models
│   ├── minimal_crypto_<name>.zip        # Model file (Stable Baselines3)
│   └── minimal_crypto_<name>.json       # Metadata sidecar (exchange, grade, steps)
├── notebooks/                  # Jupyter notebooks
│   ├── 1_data_preparation.ipynb
│   ├── 2_agent_creation.ipynb
│   ├── 3_agent_training.ipynb
│   └── 4_agent_evaluation.ipynb
└── ui/                         # Streamlit web UI
    ├── app.py                           # Entry point
    └── pages/
        ├── 1_Data_Loader.py             # โหลดข้อมูลจาก exchange (CCXT)
        ├── 2_Data_Prepare.py            # เพิ่ม indicators
        ├── 3_Train_Agent.py             # เทรน RL agent
        ├── 4_Test_Agent.py              # Backtest
        ├── 5_Evaluate_Performance.py    # วิเคราะห์ผล
        └── 6_Manage_Agents.py          # จัดการ models
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

### วิธีแนะนำ: Streamlit UI (ครบ pipeline ในหน้าเดียว)

```bash
streamlit run ui/app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501` แล้วทำตามขั้นตอน:

| ขั้นตอน | หน้า | คำอธิบาย |
|---------|------|-----------|
| 1 | 📊 Data Loader | โหลดข้อมูล OHLCV จาก Binance / Bybit / OKX ผ่าน CCXT |
| 2 | 🔧 Data Prepare | เพิ่ม technical indicators (SMA, RSI, MACD, Bollinger Bands ฯลฯ) |
| 3 | 🎯 Train Agent | เลือก Grade และกด **Create Agent** — เทรน PPO จริง |
| 4 | 🧪 Test Agent | Backtest agent บนข้อมูล test set |
| 5 | 📈 Evaluate Performance | วิเคราะห์ผลและเปรียบเทียบกับ Buy & Hold |
| 6 | ⚙️ Manage Agents | ดู/ลบ models ที่เทรนแล้ว |

#### Training Grades

| Grade | Steps | ใช้เวลาโดยประมาณ (CPU) |
|-------|-------|------------------------|
| N | 5,000 | ~2 นาที |
| D | 10,000 | ~5 นาที |
| C | 20,000 | ~10 นาที |
| B | 30,000 | ~15 นาที |
| A | 50,000 | ~25 นาที |
| S | 100,000 | ~50 นาที |

---

### วิธีที่ 2: Command Line (Python scripts)

```bash
# Basic agent (BTC-USD, ดาวน์โหลดข้อมูลจาก yfinance)
python main.py

# Advanced agent (multi-symbol, TA-Lib optional)
python simple_advanced_agent.py
```

### วิธีที่ 3: Jupyter Notebooks

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

แก้ไขไฟล์ [`config.py`](config.py) สำหรับ CLI scripts:

```python
CRYPTO_SYMBOLS = ['BTC-USD']   # สัญลักษณ์ที่ใช้ใน main.py
INITIAL_AMOUNT = 100000        # เงินทุนเริ่มต้น (USD)
HMAX = 100                     # จำนวน shares สูงสุดต่อ asset
TRANSACTION_COST_PCT = 0.001   # ค่า fee 0.1%
```

> **หมายเหตุ:** เมื่อใช้ Streamlit UI หน้า Train Agent จะอ่าน symbols จากไฟล์ข้อมูลโดยตรง ไม่จำเป็นต้องแก้ config

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

### Train Agent: "ขาด indicators ที่จำเป็น"
ไฟล์ข้อมูลที่เลือกยังไม่ผ่านหน้า Data Prepare หรือ export มาจาก `data/` แทนที่จะเป็น `data/data_prepare/`  
**แก้ไข:** ไปหน้า Data Prepare → เพิ่ม indicators → กด Export → กลับมาเลือกไฟล์ใหม่ในหน้า Train Agent

### Train Agent: "ไม่พบคอลัมน์ 'tic' หรือ 'symbol'"
ไฟล์ข้อมูลไม่ได้สร้างจาก Data Loader ของ UI นี้  
**แก้ไข:** ใช้หน้า Data Loader เพื่อโหลดข้อมูลใหม่

### AttributeError: 'numpy.float64' object has no attribute 'values'
**แก้ไข:** ใช้ `simple_advanced_agent.py` แทน `advanced_crypto_agent.py`

### TA-Lib installation
**แก้ไข:** ดูคำแนะนำใน [`install_talib.md`](install_talib.md)

### GPU/CUDA ไม่ถูกใช้งาน
**แก้ไข:** ระบบจะใช้ CPU อัตโนมัติถ้าไม่พบ GPU — ทำงานได้ทั้งสองแบบ

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

**🎯 แนะนำให้เริ่มต้นด้วย Streamlit UI (`streamlit run ui/app.py`) เพราะครบทุก pipeline และเทรน agent ได้เลยโดยไม่ต้องเขียน code!**