# 🚀 Advanced Crypto Trading Agent

## 📋 ภาพรวม

ไฟล์ `advanced_crypto_agent.py` เป็น trading agent ขั้นสูงที่พัฒนาจาก `main.py` โดยเพิ่มความสามารถและปรับปรุงหลายด้านเพื่อให้มีประสิทธิภาพการเทรดที่ดีขึ้น

## 🔄 ความแตกต่างจาก main.py

### 1. 📊 ข้อมูลและ Technical Indicators ขั้นสูง

#### main.py:
- ใช้ indicators พื้นฐาน 12 ตัว (SMA, EMA, RSI, MACD, Bollinger Bands, Volume)
- การ normalize แบบ standard
- ข้อมูล crypto 3-4 ตัว

#### advanced_crypto_agent.py:
- ใช้ **TALib** สำหรับคำนวณ indicators กว่า **40+ ตัว**
- เพิ่ม **Pattern Recognition** (Doji, Hammer, Engulfing)
- **Support/Resistance Levels** อัตโนมัติ
- **Cross-asset Features** (correlation, beta กับ BTC)
- **Market Regime Detection** (trend strength, volatility regime)
- **Outlier Detection** ด้วย Isolation Forest
- **RobustScaler** แทน StandardScaler
- รองรับ crypto มากขึ้น (BTC, ETH, ADA, DOT, SOL, MATIC)

### 2. 🏛️ Trading Environment ขั้นสูง

#### main.py:
- ใช้ StockTradingEnv มาตรฐาน
- Reward function พื้นฐาน (return-based)

#### advanced_crypto_agent.py:
- **AdvancedTradingEnv** ที่กำหนดเอง
- **Multi-component Reward Function:**
  - Return-based reward
  - Risk-adjusted reward (Sharpe-like)
  - Drawdown penalty (penalty เมื่อ drawdown > 10%)
  - Diversification reward
  - Transaction cost consideration
- แบ่งข้อมูล Train/Validation/Test (70/15/15)

### 3. 🤖 Model Training ขั้นสูง

#### main.py:
- เทรน PPO model เดียว
- Hyperparameters คงที่

#### advanced_crypto_agent.py:
- **Ensemble Training** - เทรน 3 models พร้อมกัน:
  - PPO Conservative (เน้นความปลอดภัย)
  - PPO Aggressive (เน้นผลตอบแทน)
  - PPO Balanced (สมดุล)
- **Validation-based Model Selection**
- **Hyperparameter Optimization** สำหรับแต่ละ model
- เทรน 200,000 timesteps (เพิ่มจาก 100,000)

### 4. 📈 การวิเคราะห์ผลลัพธ์ขั้นสูง

#### main.py:
- วิเคราะห์พื้นฐาน (return, comparison กับ BTC)
- กราฟ 2 แผง

#### advanced_crypto_agent.py:
- **Advanced Performance Metrics:**
  - Sharpe Ratio (annualized)
  - Maximum Drawdown
  - Volatility (annualized)
  - Calmar Ratio
  - Alpha vs BTC และ Equal-weight portfolio
- **กราฟ 4 แผง:**
  - Portfolio value over time
  - Drawdown chart
  - Rolling Sharpe ratio
  - Return comparison chart
- บันทึก summary เป็น JSON

## 🛠️ ความสามารถใหม่

### 1. 🔍 Advanced Feature Engineering
```python
# Pattern Recognition
'cdl_doji', 'cdl_hammer', 'cdl_engulfing'

# Support/Resistance
'resistance', 'support', 'distance_to_resistance', 'distance_to_support'

# Price Action
'price_change', 'high_low_ratio', 'body_size', 'upper_shadow', 'lower_shadow'

# Market Regime
'trend_strength', 'volatility_regime'

# Cross-asset
'btc_correlation', 'btc_beta'
```

### 2. 💰 Multi-Component Reward System
```python
advanced_reward = (
    portfolio_return * 100 +      # Return reward
    risk_adjusted_reward +        # Sharpe-like reward  
    drawdown_penalty +            # Risk penalty
    diversification_reward +      # Portfolio diversification
    transaction_penalty           # Trading frequency penalty
)
```

### 3. 🏆 Ensemble Strategy
- เทรนหลาย models พร้อมกัน
- เลือก best model จาก validation performance
- สามารถสร้าง weighted ensemble prediction

### 4. 📊 Risk Management
- Drawdown monitoring และ penalty
- Volatility-based position sizing
- Diversification incentives
- Transaction cost optimization

## 🚀 วิธีการใช้งาน

### วิธีที่ 1: Setup อัตโนมัติ (แนะนำ)
```bash
python setup_advanced_agent.py
```
Script นี้จะติดตั้ง dependencies ทั้งหมดโดยอัตโนมัติ

### วิธีที่ 2: ติดตั้งด้วย requirements file
```bash
pip install -r requirements_advanced.txt
```

### วิธีที่ 3: ติดตั้งแยกเป็นชิ้นๆ

**Core packages (จำเป็น):**
```bash
pip install pandas numpy matplotlib yfinance torch scikit-learn finrl
```

**Technical indicators (เลือก 1 วิธี):**
```bash
# วิธีที่ 1: pandas_ta (แนะนำ - ติดตั้งง่าย)
pip install pandas_ta

# วิธีที่ 2: TA-Lib (สำหรับ advanced users)
# Windows:
pip install TA-Lib-Binary
# หรือ: conda install -c conda-forge ta-lib

# Linux/Mac:
pip install TA-Lib
```

**หมายเหตุ:** 
- Agent จะทำงานได้แม้ไม่มี pandas_ta โดยใช้การคำนวณ manual แทน
- ถ้า talib ติดตั้งไม่ได้ ให้ใช้ pandas_ta หรือให้ agent คำนวณเอง

### รันโปรแกรม:
```bash
python advanced_crypto_agent.py
```

### การแก้ไขปัญหาการติดตั้ง:

**1. ปัญหา talib ใน Windows:**
```bash
# ลองวิธีนี้
pip install --only-binary=all TA-Lib-Binary

# หรือ
conda install -c conda-forge ta-lib

# หรือใช้ pandas_ta แทน
pip install pandas_ta
```

**2. ปัญหา torch:**
```bash
# สำหรับ CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# สำหรับ GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 ผลลัพธ์ที่คาดหวัง

Advanced agent ควรให้ผลลัพธ์ที่ดีกว่า main.py ใน:

1. **Risk-Adjusted Returns** - Sharpe ratio ที่สูงขึ้น
2. **Drawdown Control** - Maximum drawdown ที่ต่ำลง
3. **Stability** - Volatility ที่ควบคุมได้ดีขึ้น
4. **Diversification** - การกระจายความเสี่ยงที่ดีขึ้น
5. **Market Adaptation** - ปรับตัวได้ดีกับ market conditions ต่างๆ

## 🔧 การปรับแต่งเพิ่มเติม

สามารถปรับแต่ง parameters ต่างๆ ได้:

1. **Model Parameters** - ใน `train_ensemble_agents()`
2. **Reward Weights** - ใน `AdvancedTradingEnv.step()`
3. **Indicators** - ใน `add_advanced_technical_indicators()`
4. **Asset Universe** - ใน `download_crypto_data_advanced()`

## 📁 ไฟล์ที่สร้างขึ้น

```
advanced_models/
├── advanced_crypto_ppo_conservative.zip
├── advanced_crypto_ppo_aggressive.zip  
├── advanced_crypto_ppo_balanced.zip
├── advanced_performance_analysis.png
└── advanced_agent_summary.json
```

## ⚠️ ข้อควรระวัง

1. **Memory Usage** - ใช้ memory มากขึ้นเนื่องจาก features เยอะ
2. **Training Time** - ใช้เวลาเทรนนานขึ้น (3 models)
3. **Dependencies** - ต้องติดตั้ง TALib และ scikit-learn
4. **Data Requirements** - ต้องการข้อมูลย้อนหลังมากขึ้น

## 🎯 สรุป

Advanced Crypto Agent เป็นการพัฒนาต่อยอดจาก main.py ที่เพิ่ม:
- ความซับซ้อนของ features (40+ indicators)
- ระบบ risk management ที่ดีขึ้น
- Ensemble learning approach
- การวิเคราะห์ผลลัพธ์ที่ครอบคลุมขึ้น

ทำให้ agent มีศักยภาพในการสร้างผลตอบแทนที่ดีขึ้นและควบคุมความเสี่ยงได้ดีขึ้น! 