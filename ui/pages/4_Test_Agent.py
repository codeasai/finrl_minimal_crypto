import streamlit as st
import sys
from pathlib import Path

st.set_page_config(
    page_title="Test Agent",
    page_icon="🧪",
    layout="wide"
)

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

# Main UI
def test_agent_ui():
    """UI สำหรับการทดสอบ agent"""
    st.header("🧪 Test Agent")
    
    st.info("""
    ⚠️ **หมายเหตุ:** การทำงานของ Test Agent ต้องการไฟล์ core functions จาก pipeline directory 
    ที่ถูกลบออกแล้ว
    
    หากต้องการทดสอบ Agent ให้:
    1. ใช้ Jupyter Notebooks ใน `notebooks/4_agent_evaluation.ipynb`
    2. หรือใช้ไฟล์ Python หลัก `main.py`, `simple_advanced_agent.py`
    3. หรือใช้หน้า Evaluate Performance ที่ยังใช้งานได้
    """)
    
    st.markdown("---")
    
    st.subheader("🔄 ทางเลือกสำหรับการทดสอบ Agent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Jupyter Notebooks:**
        
        - `notebooks/4_agent_evaluation.ipynb` - ประเมินผล agent
        - `notebooks/5_trading_implementation.ipynb` - การเทรดจริง
        
        **วิธีใช้:**
        ```bash
        jupyter notebook notebooks/
        ```
        """)
    
    with col2:
        st.info("""
        **🎯 Streamlit Pages:**
        
        - ไปที่หน้า **"Evaluate Performance"** 
        - ใช้สำหรับวิเคราะห์ผลการเทรดอย่างละเอียด
        - เปรียบเทียบกับ Buy & Hold strategy
        - Export ผลลัพธ์เป็น CSV
        """)
    
    st.markdown("---")
    
    st.subheader("📋 ข้อมูลโมเดลที่มีอยู่")
    
    # ตรวจสอบโมเดลที่มีอยู่
    models_dir = root_path / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.zip"))
        if model_files:
            st.success(f"พบโมเดล {len(model_files)} ไฟล์:")
            for model_file in model_files:
                st.write(f"📁 {model_file.name}")
        else:
            st.warning("ไม่พบไฟล์โมเดล กรุณาเทรน agent ก่อน")
    else:
        st.warning("ไม่พบโฟลเดอร์ models")

# เรียกใช้ UI
test_agent_ui() 