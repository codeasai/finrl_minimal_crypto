import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Manage Agents",
    page_icon="⚙️",
    layout="wide"
)

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

def get_model_info(model_path):
    """ดึงข้อมูลโมเดล"""
    try:
        if os.path.exists(model_path):
            stats = os.stat(model_path)
            return {
                "size": f"{stats.st_size/1024/1024:.2f} MB",
                "modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "exists": True
            }
        else:
            return {"exists": False}
    except Exception as e:
        return {"exists": False, "error": str(e)}

def manage_agents_ui():
    """UI สำหรับการจัดการ agents"""
    st.header("⚙️ Manage Agents")
    
    # ตรวจสอบโฟลเดอร์ agents
agents_dir = root_path / "agents"

if not agents_dir.exists():
    st.warning("⚠️ ไม่พบโฟลเดอร์ agents")
    st.info("สร้างโฟลเดอร์ agents ใหม่...")
    agents_dir.mkdir(exist_ok=True)
    st.success("✅ สร้างโฟลเดอร์ agents เรียบร้อย")
    
    # ค้นหาไฟล์โมเดล
    model_files = list(agents_dir.glob("*.zip"))
    
    if not model_files:
        st.info("""
        📋 **ไม่พบโมเดลในระบบ**
        
        เพื่อสร้างโมเดลใหม่ ให้:
        1. ใช้ไฟล์ `main.py` หรือ `simple_advanced_agent.py`
        2. หรือใช้ Jupyter Notebooks ใน `notebooks/`
        3. หรือใช้หน้า Train Agent (เมื่อฟีเจอร์กลับมา)
        """)
        return
    
    st.subheader(f"📁 พบโมเดล {len(model_files)} ไฟล์")
    
    # แสดงรายการโมเดล
    for i, model_file in enumerate(model_files):
        with st.expander(f"📁 {model_file.name}", expanded=False):
            info = get_model_info(str(model_file))
            
            if info["exists"]:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**ชื่อไฟล์:** {model_file.name}")
                    st.write(f"**ขนาด:** {info['size']}")
                    st.write(f"**แก้ไขล่าสุด:** {info['modified']}")
                
                with col2:
                    if st.button("📋 คัดลอกชื่อ", key=f"copy_{i}"):
                        st.code(model_file.name)
                
                with col3:
                    if st.button("🗑️ ลบไฟล์", key=f"delete_{i}", type="secondary"):
                        try:
                            os.remove(str(model_file))
                            st.success(f"✅ ลบไฟล์ {model_file.name} เรียบร้อย")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ ไม่สามารถลบไฟล์ได้: {str(e)}")
            else:
                st.error(f"❌ ไม่สามารถอ่านข้อมูลไฟล์ได้: {info.get('error', 'Unknown error')}")
    
    st.markdown("---")
    
    st.subheader("📊 สถิติโมเดล")
    
    # คำนวณสถิติ
    total_size = sum(os.path.getsize(str(f)) for f in model_files) / 1024 / 1024  # MB
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("จำนวนโมเดล", len(model_files))
    
    with col2:
        st.metric("ขนาดรวม", f"{total_size:.2f} MB")
    
    with col3:
        if model_files:
            latest_file = max(model_files, key=lambda f: os.path.getmtime(str(f)))
            latest_time = datetime.fromtimestamp(os.path.getmtime(str(latest_file)))
            st.metric("โมเดลล่าสุด", latest_time.strftime("%m/%d %H:%M"))
    
    st.markdown("---")
    
    st.subheader("🛠️ การจัดการขั้นสูง")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🔄 การสำรองข้อมูล:**
        
        แนะนำให้สำรองโฟลเดอร์ `models/` เป็นประจำ
        
        ```bash
        # สำรองข้อมูล
        cp -r models/ backup_models/
        
        # หรือใช้ zip
        zip -r models_backup.zip models/
        ```
        """)
    
    with col2:
        st.info("""
        **📈 การประเมินผล:**
        
        - ใช้ `notebooks/4_agent_evaluation.ipynb`
        - หรือหน้า "Evaluate Performance"
        - เปรียบเทียบประสิทธิภาพระหว่างโมเดล
        
        **🎯 การใช้งาน:**
        - โหลดโมเดลด้วย `stable_baselines3`
        - ทดสอบกับข้อมูลใหม่
        """)
    
    # ปุ่มล้างทั้งหมด (อันตราย)
    st.markdown("---")
    st.subheader("⚠️ โซนอันตราย")
    
    if st.checkbox("เปิดใช้งานการลบทั้งหมด"):
        if st.button("🗑️ ลบโมเดลทั้งหมด", type="primary"):
            if st.button("⚠️ ยืนยันการลบ", type="secondary"):
                try:
                    deleted_count = 0
                    for model_file in model_files:
                        os.remove(str(model_file))
                        deleted_count += 1
                    st.success(f"✅ ลบโมเดลทั้งหมด {deleted_count} ไฟล์เรียบร้อย")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

# เรียกใช้ UI
manage_agents_ui() 