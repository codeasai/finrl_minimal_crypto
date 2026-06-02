import streamlit as st
import os
from pathlib import Path
import sys

st.set_page_config(
    page_title="Crypto RL Agent Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to Python path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

st.title("🧠 Crypto RL Agent Dashboard")

st.markdown("""
## ยินดีต้อนรับสู่ Cryptocurrency RL Trading System

ระบบนี้ช่วยให้คุณสร้างและทดสอบ RL trading agent ได้ครบ pipeline โดยไม่ต้องเขียน code
""")

st.info("📖 **มือใหม่?** ไปที่หน้า **Help** ในเมนูซ้ายเพื่อดูคำแนะนำทีละขั้นตอน")

st.markdown("""
### 🚀 Pipeline (ทำตามลำดับ)

| ขั้น | หน้า | หน้าที่ |
|-----|------|--------|
| 1 | 📊 Data Loader | โหลด OHLCV จาก Binance/Bybit/OKX |
| 2 | 🔧 Data Prepare | เพิ่ม Technical Indicators |
| 3 | 🎯 Train Agent | เทรน PPO agent |
| 4 | 🧪 Test Agent | รัน Backtest |
| 5 | 📈 Evaluate Performance | วิเคราะห์ผลและเปรียบเทียบ B&H |
| 6 | ⚙️ Manage Agents | ดู/ลบ models |

### 📋 System Overview
""")

# แสดงสถานะระบบ
col1, col2, col3 = st.columns(3)

with col1:
    # ตรวจสอบโฟลเดอร์ data
    data_dir = root_path / "data"
    if data_dir.exists():
        data_files = len([f for f in data_dir.glob("*.csv")])
        st.metric("📊 Data Files", data_files)
    else:
        st.metric("📊 Data Files", 0)

with col2:
    # ตรวจสอบโฟลเดอร์ models
    models_dir = root_path / "models"
    if models_dir.exists():
        model_files = len([f for f in models_dir.glob("*.zip")])
        st.metric("🤖 Trained Models", model_files)
    else:
        st.metric("🤖 Trained Models", 0)

with col3:
    # แสดงสถานะ GPU
    try:
        import torch
        if torch.cuda.is_available():
            st.metric("🎮 GPU Status", "Available")
        else:
            st.metric("🎮 GPU Status", "CPU Only")
    except:
        st.metric("🎮 GPU Status", "Unknown")

st.markdown("---")

# คำแนะนำการใช้งาน
with st.expander("💡 คำแนะนำการใช้งาน", expanded=False):
    st.markdown("""
    ### 📝 ขั้นตอนการใช้งานระบบ
    
    #### 1. การเตรียมข้อมูล
    - ไปที่หน้า **"📊 Data Loader"** เพื่อโหลดข้อมูลจาก exchanges
    - ใช้หน้า **"🔧 Data Prepare"** เพื่อเตรียมข้อมูลและเพิ่ม technical indicators
    
    #### 2. การเทรน Agent
    - ไปที่หน้า **"🎯 Train Agent"** เพื่อเทรน RL agent
    - เลือก Grade การเทรน (N, D, C, B, A, S) ตามความต้องการ
    - ตั้งค่า parameters ต่างๆ หรือใช้ค่า default
    
    #### 3. การทดสอบและประเมินผล
    - ใช้หน้า **"🧪 Test Agent"** เพื่อทดสอบ agent
    - ไปที่หน้า **"📈 Evaluate Performance"** เพื่อวิเคราะห์ผลการเทรด
    
    #### 4. การจัดการ
    - ใช้หน้า **"⚙️ Manage Agents"** เพื่อจัดการ agents ที่มีอยู่
    
    ### ⚠️ ข้อควรระวัง
    - ตรวจสอบว่ามี GPU สำหรับการเทรนที่มี steps มาก
    - สำรองข้อมูลสำคัญก่อนการเทรนหรือทดสอบ
    - ใช้ข้อมูลที่มีคุณภาพดีเพื่อผลลัพธ์ที่ดีที่สุด
    """)

# แสดงข้อมูลเพิ่มเติม
with st.expander("🔧 System Information", expanded=False):
    st.markdown(f"""
    **Project Structure:**
    - Root Path: `{root_path}`
    - Data Directory: `{data_dir}`
    - Models Directory: `{models_dir}`
    
    **Python Environment:**
    - Python Version: {sys.version.split()[0]}
    - Streamlit Version: {st.__version__}
    """)
    
    # แสดงสถานะโฟลเดอร์
    st.markdown("**Directory Status:**")
    
    directories = [
        ("data", "📊 Data files"),
        ("models", "🤖 Trained models"),
        ("ui/pages", "📄 Application pages"),
        ("notebooks", "📓 Jupyter notebooks")
    ]
    
    for dir_name, description in directories:
        dir_path = root_path / dir_name
        if dir_path.exists():
            st.success(f"✅ {description}: `{dir_path}`")
        else:
            st.warning(f"⚠️ {description}: `{dir_path}` (not found)")

st.markdown("---")
st.markdown("**🧠 Crypto RL Agent Dashboard** - Powered by FinRL & Streamlit")
