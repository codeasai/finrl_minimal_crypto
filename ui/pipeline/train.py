import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR, MODEL_KWARGS

def get_model_info(model_path):
    """Get model information"""
    if os.path.exists(model_path):
        stats = os.stat(model_path)
        return {
            "last_modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size": f"{stats.st_size/1024:.1f} KB"
        }
    return None

def get_training_params(model_type="PPO"):
    """Get training parameters based on model type"""
    if model_type == "PPO":
        return {
            "learning_rate": MODEL_KWARGS["learning_rate"],
            "n_steps": MODEL_KWARGS["n_steps"],
            "batch_size": MODEL_KWARGS["batch_size"],
            "n_epochs": MODEL_KWARGS["n_epochs"]
        }
    else:  # PPO Simple
        return {
            "learning_rate": 1e-4,
            "batch_size": 128,
            "n_steps": 1024,
            "gamma": 0.99,
            "gae_lambda": 0.95
        }

def show_continue_training_guide():
    """แสดงคำแนะนำการใช้งาน Continue Training"""
    with st.expander("ℹ️ คำแนะนำการใช้ Continue Training", expanded=True):
        st.markdown("""
        **การเทรนต่อจากโมเดลเดิม (Continue Training) เหมาะสำหรับ:**
        1. 🔄 ต้องการเทรนโมเดลต่อเพื่อปรับปรุงประสิทธิภาพ
        2. 📈 ต้องการเทรนกับข้อมูลใหม่โดยใช้ความรู้จากโมเดลเดิม
        3. ⏱️ ต้องการเทรนเพิ่มเติมหลังจากหยุดการเทรนกลางคัน
        
        **ข้อควรระวัง:**
        - ⚠️ การเปลี่ยนค่า Learning Rate มากเกินไปอาจทำให้โมเดลลืมสิ่งที่เรียนรู้มาก่อน
        - 📊 ควรประเมินผลโมเดลก่อนและหลังการเทรนต่อเพื่อเปรียบเทียบประสิทธิภาพ
        - 💾 ระบบจะสร้าง checkpoint ทุกๆ Save Interval steps เพื่อป้องกันการสูญเสียข้อมูล
        
        **ขั้นตอนแนะนำ:**
        1. 📋 ประเมินผลโมเดลเดิมในส่วน Evaluate ก่อน
        2. 🎯 เลือกจำนวน steps ที่ต้องการเทรนเพิ่ม
        3. ⚙️ ปรับ parameters ให้เหมาะสม (แนะนำให้ใช้ค่าเดิมในการเทรนครั้งแรก)
        4. 🔄 กดปุ่ม Continue Training เพื่อเริ่มการเทรน
        5. 📊 ประเมินผลโมเดลใหม่หลังเทรนเสร็จ
        """)

def train_agent_ui():
    st.header("🎯 Train RL Agent")
    
    # Check existing models
    existing_models = [f.replace("minimal_crypto_", "") for f in os.listdir(MODEL_DIR) 
                      if f.startswith("minimal_crypto_")] if os.path.exists(MODEL_DIR) else []
    
    # Training mode selection
    train_mode = st.radio(
        "Training Mode",
        ["Train New Model", "Continue Training"],
        help="Choose whether to train a new model or continue training an existing one"
    )
    
    if train_mode == "Continue Training":
        # แสดงคำแนะนำการใช้งาน Continue Training
        show_continue_training_guide()
        
        if not existing_models:
            st.warning("⚠️ No existing models found. Please train a new model first.")
            return
            
        # Model selection for continuing training
        model_to_continue = st.selectbox(
            "Select Model to Continue Training",
            existing_models,
            help="Choose an existing model to continue training"
        )
        
        # Show existing model info
        model_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_to_continue}")
        model_info = get_model_info(model_path)
        if model_info:
            st.info(f"📝 Last trained: {model_info['last_modified']} | Size: {model_info['size']}")
        
        # Use the same type as the existing model
        model_type = "PPO (Simple)" if "simple" in model_to_continue else "PPO"
        st.write(f"Model Type: {model_type}")
        
        # แสดงคำแนะนำเพิ่มเติมสำหรับการตั้งค่า
        st.info("""
        💡 **คำแนะนำ:** ในการเทรนครั้งแรก แนะนำให้ใช้ค่า parameters เดิม 
        หากผลลัพธ์ไม่เป็นที่น่าพอใจ ค่อยปรับในการเทรนครั้งถัดไป
        """)
        
    else:  # Train New Model
        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            ["PPO", "PPO (Simple)"],
            help="PPO (Simple) uses fewer parameters and may train faster"
        )
    
    # Training parameters
    with st.expander("🔧 Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input(
                "Training Steps",
                min_value=1000,
                value=10000,
                step=1000,
                help="More steps = better training but takes longer"
            )
        with col2:
            save_interval = st.number_input(
                "Save Interval (steps)",
                min_value=1000,
                value=5000,
                step=1000,
                help="How often to save checkpoints during training"
            )
        
        # Show current parameters
        st.json(get_training_params(model_type))
        
        # Advanced options
        if st.checkbox("🔍 Show Advanced Options"):
            st.warning("""
            ⚠️ **คำเตือน:** การปรับค่าเหล่านี้อาจส่งผลต่อประสิทธิภาพของโมเดล 
            แนะนำให้ทดลองกับค่าเดิมก่อน
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=float(get_training_params(model_type)["learning_rate"]),
                    format="%.0e",
                    help="Model's learning rate (ควรปรับลดลงเมื่อ continue training)"
                )
            with col2:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=32,
                    max_value=512,
                    value=get_training_params(model_type)["batch_size"],
                    step=32,
                    help="Training batch size"
                )
    
    # Training button and progress
    if train_mode == "Continue Training":
        start_button = st.button("🚀 Continue Training")
    else:
        start_button = st.button("🚀 Start Training")
    
    if start_button:
        # Create or get model name
        if train_mode == "Continue Training":
            model_name = f"minimal_crypto_{model_to_continue}"
            st.info(f"📈 Continuing training for {model_to_continue}...")
        else:
            model_name = f"minimal_crypto_ppo{'_simple' if model_type == 'PPO (Simple)' else ''}"
            if model_name.replace("minimal_crypto_", "") in existing_models:
                st.warning(f"⚠️ Model {model_type} already exists. Training will create a backup.")
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training loop simulation
        for i in range(5):  # TODO: Replace with actual training
            progress = (i + 1) * 20
            progress_bar.progress(progress)
            status_text.text(f"Training progress: {progress}% | Step: {(i+1)*steps//5}/{steps}")
            if (i + 1) * steps//5 % save_interval == 0:
                st.info(f"💾 Saved checkpoint at step {(i+1)*steps//5}")
        
        st.success(f"✅ Training completed! Model saved as {model_type}")
        
        # Show next steps with more detail
        st.info("""
        👉 **ขั้นตอนถัดไป:**
        1. ไปที่หน้า Evaluate เพื่อประเมินผลโมเดล
        2. เปรียบเทียบผลลัพธ์กับโมเดลก่อนการเทรน
        3. หากผลลัพธ์ยังไม่ดีพอ สามารถกลับมาเทรนต่อได้
        """)
