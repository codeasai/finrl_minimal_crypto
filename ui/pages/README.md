# 📁 Pages Directory - Streamlit Multi-Page App

โฟลเดอร์นี้ประกอบด้วยหน้าต่างๆ ของ Streamlit Multi-Page Application

## 📋 รายการหน้า

### 1. 📊 Data Loader
**ไฟล์:** `1_Data_Loader.py`
- โหลดข้อมูล cryptocurrency จาก exchanges ต่างๆ
- รองรับ Binance, Bybit, OKX
- กำหนดช่วงเวลาและคู่เทรด
- บันทึกข้อมูลในรูปแบบ CSV

### 2. 🔧 Data Prepare  
**ไฟล์:** `2_Data_Prepare.py`
- เตรียมข้อมูลสำหรับการเทรน
- เพิ่ม Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Normalize ข้อมูลด้วย Min-Max และ Z-score
- ตรวจสอบคุณภาพข้อมูล (NaN, infinity, negative prices)
- บันทึกข้อมูลที่ประมวลผลแล้ว

### 3. 🎯 Train Agent
**ไฟล์:** `3_Train_Agent.py`  
- เทรน RL agent ด้วย PPO algorithm
- รองรับ Grade system (N, D, C, B, A, S)
- ตั้งค่า hyperparameters
- สร้าง agent ใหม่หรือเทรนต่อจากโมเดลเดิม
- รองรับ GPU acceleration

### 4. 🧪 Test Agent
**ไฟล์:** `4_Test_Agent.py`
- ทดสอบประสิทธิภาพของ trained agent
- Real-time trading simulation
- แสดงผลการเทรดแบบ step-by-step

### 5. 📈 Evaluate Performance
**ไฟล์:** `5_Evaluate_Performance.py`
- วิเคราะห์ผลการเทรดอย่างละเอียด
- เปรียบเทียบกับ Buy & Hold strategy
- คำนวณ performance metrics:
  - Sharpe Ratio
  - Maximum Drawdown
  - Volatility
  - Win Rate
  - Sortino Ratio
  - Calmar Ratio
- สร้างกราฟเปรียบเทียบ
- Export ผลลัพธ์เป็น CSV

### 6. ⚙️ Manage Agents
**ไฟล์:** `6_Manage_Agents.py`
- ดูรายการ agents ทั้งหมด
- ลบ agents ที่ไม่ต้องการ
- จัดการไฟล์โมเดล
- แสดงข้อมูลโมเดล (ขนาด, วันที่สร้าง, etc.)

## 🔗 Dependencies

แต่ละหน้ายังคงใช้โมดูลจากโฟลเดอร์ `pipeline/`:
- `data_loader.py` - ฟังก์ชันโหลดข้อมูล
- `data_prepare.py` & `data_prepare_ui.py` - ฟังก์ชันเตรียมข้อมูล
- `train.py` - ฟังก์ชันเทรน agent
- `test.py` - ฟังก์ชันทดสอบ agent
- `evaluate.py` - ฟังก์ชันประเมินผล
- `agent_manager.py` - ฟังก์ชันจัดการ agents

## 🏗️ Architecture

```
ui/
├── app.py (Main Dashboard)
├── pages/
│   ├── 1_Data_Loader.py
│   ├── 2_Data_Prepare.py  
│   ├── 3_Train_Agent.py
│   ├── 4_Test_Agent.py
│   ├── 5_Evaluate_Performance.py
│   └── 6_Manage_Agents.py
└── pipeline/ (Backend modules)
    ├── data_loader.py
    ├── data_prepare.py
    ├── data_prepare_ui.py
    ├── train.py
    ├── test.py
    ├── evaluate.py
    └── agent_manager.py
```

## 🔄 Navigation

Streamlit อ่านไฟล์ในโฟลเดอร์ `pages/` อัตโนมัติและสร้าง navigation menu จาก:
1. หมายเลขลำดับ (1_, 2_, 3_, ...)
2. ชื่อหน้า (Data_Loader, Data_Prepare, ...)

## 📝 Naming Convention

ไฟล์ในโฟลเดอร์ pages ใช้รูปแบบ:
```
{order}_{page_name}.py
```

ตัวอย่าง:
- `1_Data_Loader.py`
- `2_Data_Prepare.py`
- `3_Train_Agent.py`

## 🚀 การเพิ่มหน้าใหม่

เพื่อเพิ่มหน้าใหม่:

1. สร้างไฟล์ใหม่ในโฟลเดอร์ `pages/` 
2. ใช้ naming convention ที่กำหนด
3. เขียน Streamlit code ในไฟล์
4. Import ฟังก์ชันจาก `pipeline/` หากจำเป็น

ตัวอย่าง:
```python
# 7_Portfolio_Analysis.py
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

# Import from pipeline
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))
from portfolio_analysis import portfolio_analysis_ui

st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="💰",
    layout="wide"
)

# Main UI
portfolio_analysis_ui()
```

## ⚡ Performance Tips

- ใช้ `@st.cache_data` สำหรับข้อมูลที่ไม่เปลี่ยนบ่อย
- ใช้ `@st.cache_resource` สำหรับ model loading
- หลีกเลี่ยงการ import module ที่ไม่จำเป็นในหน้าหลัก
- ใช้ `st.spinner()` สำหรับ operations ที่ใช้เวลานาน

---
**🧠 Crypto RL Agent Dashboard** - Streamlit Multi-Page App Structure 