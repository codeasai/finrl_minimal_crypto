# การติดตั้ง TA-Lib สำหรับ Advanced Crypto Agent

## สำหรับ Windows

### วิธีที่ 1: ใช้ conda (แนะนำ)
```bash
# เข้า conda environment
conda activate tfyf

# ติดตั้ง TA-Lib จาก conda-forge
conda install -c conda-forge ta-lib
```

### วิธีที่ 2: ใช้ pip กับ pre-compiled wheel
```bash
# ติดตั้งจาก wheel file
pip install --find-links https://www.lfd.uci.edu/~gohlke/pythonlibs/ TA-Lib
```

### วิธีที่ 3: ดาวน์โหลด wheel file โดยตรง
1. ไปที่ https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. ดาวน์โหลด wheel file ที่เหมาะกับ Python version ของคุณ
   - สำหรับ Python 3.12: `TA_Lib‑0.4.28‑cp312‑cp312‑win_amd64.whl`
   - สำหรับ Python 3.11: `TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl`
3. ติดตั้งด้วยคำสั่ง:
```bash
pip install TA_Lib‑0.4.28‑cp312‑cp312‑win_amd64.whl
```

## สำหรับ macOS

### ใช้ Homebrew
```bash
# ติดตั้ง TA-Lib C library
brew install ta-lib

# ติดตั้ง Python wrapper
pip install TA-Lib
```

### ใช้ conda
```bash
conda install -c conda-forge ta-lib
```

## สำหรับ Linux (Ubuntu/Debian)

### ติดตั้ง dependencies
```bash
# ติดตั้ง TA-Lib C library
sudo apt-get update
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

### ติดตั้ง Python wrapper
```bash
pip install TA-Lib
```

## การตรวจสอบการติดตั้ง

หลังจากติดตั้งเสร็จแล้ว ให้ทดสอบด้วยคำสั่ง:

```python
import talib
print("TA-Lib version:", talib.__version__)
print("✅ TA-Lib ติดตั้งสำเร็จ!")
```

## หากการติดตั้งไม่สำเร็จ

ถ้าไม่สามารถติดตั้ง TA-Lib ได้ สคริปต์จะใช้ manual calculations แทน โดยอัตโนมัติ

สคริปต์จะแสดงข้อความ:
- `✅ Using TA-Lib for technical indicators` - ถ้าติดตั้งสำเร็จ
- `⚠️ TA-Lib not found, using manual calculations` - ถ้าไม่พบ TA-Lib

## การใช้งาน

หลังจากติดตั้ง TA-Lib เสร็จแล้ว ให้รันสคริปต์:

```bash
python advanced_crypto_agent.py
```

สคริปต์จะใช้ TA-Lib เพื่อคำนวณ technical indicators ต่อไปนี้:
- Moving Averages (SMA, EMA)
- Momentum indicators (RSI, Stochastic, Williams %R)
- MACD
- Bollinger Bands
- Volume indicators (AD, OBV)
- Volatility indicators (ATR, NATR)
- Trend indicators (ADX, CCI)
- Candlestick patterns (Doji, Hammer, Engulfing)

TA-Lib จะให้ผลลัพธ์ที่แม่นยำและเร็วกว่า manual calculations มาก! 