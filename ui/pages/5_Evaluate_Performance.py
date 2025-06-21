import streamlit as st
import sys
from pathlib import Path

st.set_page_config(
    page_title="Evaluate Performance",
    page_icon="📈",
    layout="wide"
)

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

# Main UI
def evaluate_agent_ui():
    """UI สำหรับการประเมินผล agent"""
    st.header("📈 Evaluate Performance")
    
    st.info("""
    ⚠️ **หมายเหตุ:** การทำงานของ Evaluate Performance ต้องการไฟล์ core functions จาก pipeline directory 
    ที่ถูกลบออกแล้ว
    
    หากต้องการประเมินผล Agent ให้:
    1. ใช้ Jupyter Notebooks ใน `notebooks/4_agent_evaluation.ipynb`
    2. ใช้ไฟล์ Python หลัก `main.py`, `simple_advanced_agent.py`
    3. รอการพัฒนาเวอร์ชั่นใหม่ที่ integrate ทุกอย่างเข้าด้วยกัน
    """)
    
    st.markdown("---")
    
    st.subheader("📊 ทางเลือกสำหรับการประเมินผล")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📓 Jupyter Notebooks (แนะนำ):**
        
        - `notebooks/4_agent_evaluation.ipynb` - ประเมินผลโดยละเอียด
        - มี visualization ครบถ้วน
        - สามารถเปรียบเทียบหลายโมเดล
        - Export ผลลัพธ์ได้
        
        **วิธีใช้:**
        ```bash
        jupyter notebook notebooks/4_agent_evaluation.ipynb
        ```
        """)
    
    with col2:
        st.info("""
        **🐍 Python Scripts:**
        
        - `main.py` - รันและประเมินผลแบบง่าย
        - `simple_advanced_agent.py` - ประเมินผลแบบขั้นสูง
        - มี performance metrics พื้นฐาน
        
        **วิธีใช้:**
        ```bash
        python main.py
        python simple_advanced_agent.py
        ```
        """)
    
    st.markdown("---")
    
    st.subheader("📋 ไฟล์ที่เกี่ยวข้อง")
    
    # ตรวจสอบไฟล์ที่เกี่ยวข้อง
    files_to_check = [
        ("agents", "โมเดลที่เทรนแล้ว"),
        ("data", "ข้อมูลสำหรับการประเมิน"),
        ("notebooks", "Jupyter notebooks")
    ]
    
    for folder, description in files_to_check:
        folder_path = root_path / folder
        if folder_path.exists():
            files = list(folder_path.glob("*"))
            if files:
                st.success(f"✅ {description}: พบ {len(files)} ไฟล์")
                if folder == "agents":
                    model_files = [f for f in files if f.suffix == '.zip']
                    for model_file in model_files[:5]:  # แสดงแค่ 5 ไฟล์แรก
                        st.write(f"   📁 {model_file.name}")
                    if len(model_files) > 5:
                        st.write(f"   ... และอีก {len(model_files) - 5} ไฟล์")
            else:
                st.warning(f"⚠️ {description}: โฟลเดอร์ว่าง")
        else:
            st.error(f"❌ {description}: ไม่พบโฟลเดอร์")
    
    st.markdown("---")
    
    st.subheader("💡 คำแนะนำ")
    
    st.markdown("""
    **สำหรับการประเมินผลที่สมบูรณ์:**
    
    1. **เริ่มต้น:** ใช้ `notebooks/4_agent_evaluation.ipynb`
    2. **ง่ายๆ:** รัน `python main.py` แล้วดูผลลัพธ์
    3. **ขั้นสูง:** ใช้ `simple_advanced_agent.py` สำหรับ metrics ละเอียด
    4. **การเปรียบเทียบ:** ใช้ notebook เพื่อเปรียบเทียบหลายโมเดล
    
    **Performance Metrics ที่สำคัญ:**
    - Total Return (%)
    - Sharpe Ratio  
    - Maximum Drawdown
    - Win Rate
    - Volatility
    """)

# เรียกใช้ UI
evaluate_agent_ui() 