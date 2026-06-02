import streamlit as st
from pathlib import Path
import sys
import os

st.set_page_config(
    page_title="Help & Guide",
    page_icon="📖",
    layout="wide"
)

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

st.title("📖 คู่มือการใช้งาน — Crypto RL Trading System")

st.markdown("""
ระบบนี้ช่วยให้คุณสร้าง RL (Reinforcement Learning) agent สำหรับ cryptocurrency trading
โดยไม่ต้องเขียน code — ทำได้ทุกขั้นตอนผ่าน UI นี้
""")

# ─── Pipeline Overview ────────────────────────────────────────────────────────
st.markdown("---")
st.header("🗺️ Pipeline ภาพรวม")

col1, col2, col3, col4, col5, col6 = st.columns(6)
steps = [
    ("📊", "1", "Data\nLoader", "โหลดข้อมูล OHLCV"),
    ("🔧", "2", "Data\nPrepare", "เพิ่ม indicators"),
    ("🎯", "3", "Train\nAgent", "เทรน RL agent"),
    ("🧪", "4", "Test\nAgent", "รัน backtest"),
    ("📈", "5", "Evaluate", "วิเคราะห์ผล"),
    ("⚙️", "6", "Manage", "จัดการ models"),
]
for col, (icon, num, name, desc) in zip(
        [col1, col2, col3, col4, col5, col6], steps):
    col.markdown(f"""
<div style="text-align:center; padding:10px; border:1px solid #ddd; border-radius:8px;">
<div style="font-size:2em">{icon}</div>
<div style="font-weight:bold">ขั้นที่ {num}</div>
<div style="font-size:0.9em">{name}</div>
<div style="color:gray; font-size:0.8em">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ─── Step-by-step guide ───────────────────────────────────────────────────────
st.markdown("---")
st.header("📋 คำแนะนำทีละขั้นตอน")

with st.expander("📊 ขั้นที่ 1 — Data Loader", expanded=True):
    st.markdown("""
### เป้าหมาย
โหลดข้อมูลราคา OHLCV (Open, High, Low, Close, Volume) จาก exchange จริง

### วิธีใช้
1. เลือก **Exchange** (Binance / Bybit / OKX)
2. ใส่ **สกุลเงิน** ที่ต้องการ เช่น `BTC/USDT,ETH/USDT`
3. เลือก **Timeframe** — แนะนำ `1d` สำหรับเทรน RL agent
4. ตั้งช่วงวันที่ — แนะนำอย่างน้อย **6 เดือน** (ยิ่งนานยิ่งดี)
5. กด **เริ่มโหลดข้อมูล**

### ผลลัพธ์
ไฟล์ CSV จะถูกบันทึกใน `data/` เช่น `BTC_USDT-1d-20240101-20241231.csv`

### เคล็ดลับ
- ใช้ `1d` (1 วัน) สำหรับ RL training — ข้อมูลที่ละเอียดเกินไป (5m, 15m) ทำให้เทรนนานมากโดยไม่จำเป็น
- โหลดอย่างน้อย **365 วัน** เพื่อให้มีข้อมูล test set เพียงพอ
- หลาย symbol (เช่น BTC + ETH) ช่วยให้ agent เรียนรู้การ diversify portfolio ได้

### Troubleshooting
| ปัญหา | แนวทางแก้ไข |
|-------|-------------|
| Connection error | ตรวจสอบอินเทอร์เน็ต หรือลอง exchange อื่น |
| ข้อมูลน้อยกว่า 100 แถว | ขยายช่วงวันที่ให้กว้างขึ้น |
| Symbol not found | ตรวจสอบรูปแบบ เช่น `BTC/USDT` (ต้องมี `/`) |
""")

with st.expander("🔧 ขั้นที่ 2 — Data Prepare"):
    st.markdown("""
### เป้าหมาย
เพิ่ม Technical Indicators ให้กับข้อมูล OHLCV เพื่อให้ RL agent มี features สำหรับตัดสินใจ

### วิธีใช้
1. เลือกไฟล์จาก **ขั้นที่ 1** (ไฟล์ใน `data/`)
2. ดูตัวอย่างข้อมูลในแท็บ Preview
3. กด **เพิ่ม Technical Indicators**
4. กด **Export** เพื่อบันทึกไฟล์ที่พร้อมเทรน

### Indicators ที่เพิ่ม
| กลุ่ม | Indicators |
|-------|-----------|
| Trend | SMA 5/10/20/50, EMA 12/20/26 |
| Momentum | RSI 14/21, MACD, MACD Signal, MACD Hist |
| Volatility | Bollinger Bands (upper/middle/lower/std), ATR, NATR |
| Volume | Volume SMA 20, Volume Ratio, OBV, A/D |
| Oscillator | Stochastic K/D, Williams %R, CCI, ADX |

### ผลลัพธ์
ไฟล์ที่ Export จะบันทึกใน `data/data_prepare/` — **ใช้ไฟล์นี้ในการเทรน**

### เคล็ดลับ
- ถ้า **TA-Lib** ติดตั้งอยู่ จะได้ indicators ที่แม่นยำกว่า
- ถ้าไม่มี TA-Lib ระบบใช้ manual calculation แทน — ทำงานได้เหมือนกัน
- ต้องการอย่างน้อย **50 แถว** สำหรับคำนวณ indicators ได้ครบ

### Troubleshooting
| ปัญหา | แนวทางแก้ไข |
|-------|-------------|
| ไม่พบ TA-Lib | ติดตั้งด้วย `conda install ta-lib -c conda-forge` |
| Indicators เป็น NaN ทั้งหมด | ข้อมูลน้อยเกินไป — โหลดข้อมูลเพิ่มในขั้นที่ 1 |
""")

with st.expander("🎯 ขั้นที่ 3 — Train Agent"):
    st.markdown("""
### เป้าหมาย
เทรน PPO (Proximal Policy Optimization) agent ให้เรียนรู้กลยุทธ์การซื้อขาย

### วิธีใช้
1. เลือก **Training Mode**: Create Agent (ใหม่) หรือ Continue Training (ต่อจากเดิม)
2. เลือก **Exchange** — ส่งผลต่อ fee structure ที่ใช้ใน simulation
3. เลือก **Grade** ตามเวลาที่มี
4. เลือก **ไฟล์ข้อมูล** จาก `data/data_prepare/` (ไฟล์ที่ผ่านขั้นที่ 2 แล้ว)
5. กด **Create Agent**

### Training Grades
| Grade | Steps | เวลา (CPU) | เหมาะกับ |
|-------|-------|-----------|---------|
| N | 5,000 | ~2 นาที | ทดสอบว่า pipeline ทำงานได้ |
| D | 10,000 | ~5 นาที | Development / debug |
| C | 20,000 | ~10 นาที | ผลเบื้องต้น |
| B | 30,000 | ~15 นาที | ผลพอใช้ |
| A | 50,000 | ~25 นาที | ผลดี |
| S | 100,000 | ~50 นาที | ผลดีที่สุด |

### ผลลัพธ์
- `models/minimal_crypto_<name>.zip` — ตัว model
- `models/minimal_crypto_<name>.json` — metadata (exchange, grade, symbols, steps)

### ชื่อ Agent
ระบบตั้งชื่ออัตโนมัติในรูปแบบ: `{Symbol}-{ModelType}-{ClassName}`
เช่น `BTC_USDT-PPO-Warcryer`

### เคล็ดลับ
- **แนะนำ Grade B ขึ้นไป** สำหรับผลที่ใช้งานได้จริง
- GPU ช่วยให้เทรนเร็วขึ้น ~3-5x — ตรวจสอบจาก GPU Status ด้านบน
- ถ้าต้องการ multi-symbol ต้องโหลดข้อมูลหลาย symbol ในขั้นที่ 1 แล้วรวมไฟล์ก่อน

### Troubleshooting
| ปัญหา | แนวทางแก้ไข |
|-------|-------------|
| "ขาด indicators ที่จำเป็น" | ใช้ไฟล์จาก `data/data_prepare/` ไม่ใช่ `data/` |
| "ไม่พบคอลัมน์ tic" | ไฟล์ต้องมาจากหน้า Data Loader ของระบบนี้ |
| CUDA out of memory | ลด Batch Size ใน Advanced Options |
| เทรนช้ามาก | ใช้ Grade N หรือ D ก่อน แล้วค่อย Continue Training |
""")

with st.expander("🧪 ขั้นที่ 4 — Test Agent (Backtest)"):
    st.markdown("""
### เป้าหมาย
ทดสอบ agent บน **test set (20% หลัง)** ของข้อมูล ซึ่ง agent ไม่เคยเห็นระหว่างเทรน

### วิธีใช้
1. เลือก **Model** ที่ต้องการทดสอบ (เรียงตามล่าสุด)
2. เลือก **ไฟล์ข้อมูล** — ควรเป็นไฟล์เดียวกับที่ใช้เทรน (ระบบแนะนำให้อัตโนมัติ)
3. กด **Run Backtest**

### ผลลัพธ์ที่แสดง
- Total Return ของ agent บน test set
- กราฟ portfolio value
- ไฟล์ CSV บันทึกใน `models/backtest_results/`
- ผลถูกส่งต่อไปหน้า Evaluate Performance โดยอัตโนมัติ

### สิ่งที่ควรรู้
- Test set = 20% หลังของข้อมูลที่เลือก — ตรงกับที่ใช้ตอนเทรน
- ถ้า backtest แล้วผลแย่ อาจเพราะ Grade ต่ำเกินไป หรือข้อมูลน้อยเกินไป
- ผล backtest ใน session จะหายถ้า reload หน้า — กด Run Backtest ใหม่ หรือโหลดจากไฟล์ CSV

### Troubleshooting
| ปัญหา | แนวทางแก้ไข |
|-------|-------------|
| Error ตอนโหลด model | ไฟล์ .zip อาจเสียหาย — เทรนใหม่ |
| "ขาด indicators" | เลือกไฟล์จาก data_prepare/ และผ่านขั้นที่ 2 |
| ผล backtest ดูผิดปกติ | ตรวจสอบว่าเลือกไฟล์ตรงกับที่ใช้เทรน |
""")

with st.expander("📈 ขั้นที่ 5 — Evaluate Performance"):
    st.markdown("""
### เป้าหมาย
วิเคราะห์ผลการเทรดของ agent อย่างละเอียด และเปรียบเทียบกับ Buy & Hold

### วิธีใช้
- ถ้าเพิ่งรัน backtest จากขั้นที่ 4 — ผลจะโหลดอัตโนมัติจาก session
- ถ้า reload หน้า — เลือก backtest file จาก dropdown

### Metrics ที่แสดง

| Metric | ความหมาย | เป้าหมาย |
|--------|----------|---------|
| Total Return | ผลตอบแทนรวม | > 0% |
| Sharpe Ratio | Return ต่อหน่วยความเสี่ยง | > 1.0 = ดี |
| Sortino Ratio | เหมือน Sharpe แต่นับแค่ downside | > 1.0 = ดี |
| Max Drawdown | ขาดทุนสูงสุดจาก peak | ยิ่งน้อยยิ่งดี (ใกล้ 0%) |
| Calmar Ratio | Return / Max Drawdown | > 1.0 = ดี |
| Win Rate | % วันที่ portfolio เพิ่มขึ้น | > 50% |
| Volatility | ความผันผวนรายปี | ขึ้นกับ risk tolerance |

### กราฟ
- **Portfolio vs Buy & Hold** — เปรียบเทียบ agent กับการถือ crypto ไว้เฉยๆ
- **Drawdown** — เห็นช่วงที่ขาดทุนหนักสุด
- **B&H Return per Symbol** — Alpha ที่ agent ทำได้เทียบกับแต่ละ symbol

### Export
ดาวน์โหลด Account Value และ Actions เป็น CSV สำหรับวิเคราะห์เพิ่มเติมใน Excel

### ตีความผล
- **Sharpe > 1** = ดี, **> 2** = ดีมาก
- ถ้า agent แพ้ Buy & Hold — ลองเพิ่ม grade หรือข้อมูลให้มากขึ้น
- Max Drawdown > 30% = ความเสี่ยงสูง
""")

with st.expander("⚙️ ขั้นที่ 6 — Manage Agents"):
    st.markdown("""
### เป้าหมาย
ดูรายละเอียด model ทั้งหมด และลบ model ที่ไม่ต้องการ

### สิ่งที่แสดง
- ชื่อไฟล์, ขนาด, วันที่แก้ไขล่าสุด
- Exchange, Grade, Steps, Symbols (ถ้ามี metadata sidecar)
- สถิติรวม: จำนวน model, ขนาดรวม

### การลบ model
- กดปุ่ม 🗑️ ข้างชื่อไฟล์ เพื่อลบทีละไฟล์
- เปิด "โซนอันตราย" เพื่อลบทั้งหมดพร้อมกัน
""")

# ─── FAQ ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("❓ คำถามที่พบบ่อย")

faq = [
    ("ต้องใช้ GPU ไหม?",
     "ไม่จำเป็น — ระบบทำงานได้บน CPU แต่ใช้เวลานานกว่า GPU ประมาณ 3–5 เท่า "
     "สำหรับ Grade N–C CPU เพียงพอ Grade A–S แนะนำ GPU"),
    ("ทำไม agent ถึงแพ้ Buy & Hold?",
     "เป็นเรื่องปกติสำหรับ Grade ต่ำ (N–C) ที่เทรนน้อยเกินไป "
     "ลอง Grade B–S และข้อมูลอย่างน้อย 1 ปี นอกจากนี้ crypto market มี trend ชัดเจน "
     "ทำให้ B&H มักชนะในตลาด bull run"),
    ("ไฟล์ข้อมูลควรมาจากไหน?",
     "ใช้ **Data Loader** โหลดจาก exchange แล้วผ่าน **Data Prepare** "
     "เสมอ — อย่าใช้ไฟล์ภายนอกโดยตรงเพราะ column format อาจต่างกัน"),
    ("Continue Training กับ Create Agent ต่างกันอย่างไร?",
     "**Create Agent** = สร้าง model ใหม่ตั้งแต่ต้น  \n"
     "**Continue Training** = เทรนต่อจาก model เดิม เพิ่ม steps โดยไม่ลืมสิ่งที่เรียนไปแล้ว "
     "(ประหยัดเวลาถ้าต้องการ Grade สูงขึ้น)"),
    ("backtest ต่างจาก live trading อย่างไร?",
     "backtest ใช้ข้อมูลในอดีตที่รู้ผลแล้ว — ผลดีใน backtest ไม่ได้รับประกันผลใน live trading "
     "โดยเฉพาะ slippage, liquidity, และ market impact ที่ backtest ไม่จำลอง"),
    ("ผล session หายหลัง reload หน้าทำอย่างไร?",
     "ไปหน้า **Test Agent** แล้วกด Run Backtest อีกครั้ง หรือที่หน้า **Evaluate Performance** "
     "เลือกไฟล์ CSV จาก dropdown (ระบบบันทึก CSV ไว้ใน `models/backtest_results/` อัตโนมัติ)"),
]

for q, a in faq:
    with st.expander(f"**{q}**"):
        st.markdown(a)

# ─── Quick reference ──────────────────────────────────────────────────────────
st.markdown("---")
st.header("📂 โครงสร้างไฟล์สำคัญ")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**ข้อมูล:**
```
data/
├── *.csv                   ← Raw OHLCV จาก Data Loader
└── data_prepare/
    └── *.csv               ← พร้อมเทรน (ใช้ในขั้นที่ 3)
```
""")
with col2:
    st.markdown("""
**Models:**
```
models/
├── minimal_crypto_*.zip    ← ตัว model
├── minimal_crypto_*.json   ← metadata
└── backtest_results/
    ├── *_account_value.csv ← ผล backtest
    └── *_actions.csv       ← การซื้อขาย
```
""")

# ─── System status ────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🖥️ System Status")

c1, c2, c3, c4 = st.columns(4)

data_dir = root_path / "data"
models_dir = root_path / "models"

n_raw   = len(list(data_dir.glob("*.csv"))) if data_dir.exists() else 0
n_prep  = len(list((data_dir / "data_prepare").glob("*.csv"))) if (data_dir / "data_prepare").exists() else 0
n_model = len(list(models_dir.glob("minimal_crypto_*.zip"))) if models_dir.exists() else 0
n_bt    = len(list((models_dir / "backtest_results").glob("*_account_value.csv"))) if (models_dir / "backtest_results").exists() else 0

c1.metric("Raw Data Files",      n_raw)
c2.metric("Prepared Data Files", n_prep)
c3.metric("Trained Models",      n_model)
c4.metric("Backtest Results",    n_bt)

try:
    import torch
    gpu = f"✅ {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "⚠️ CPU only"
except Exception:
    gpu = "Unknown"
st.caption(f"GPU: {gpu}")
