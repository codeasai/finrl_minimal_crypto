import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import random
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

# เพิ่ม import สำหรับตรวจสอบ GPU
import torch
from torch.cuda import is_available as cuda_available
from torch.cuda import device_count as cuda_device_count
from torch.cuda import get_device_name as cuda_get_device_name

st.set_page_config(
    page_title="Train Agent",
    page_icon="🎯",
    layout="wide"
)

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR, MODEL_KWARGS, DATA_DIR, INITIAL_AMOUNT, TRANSACTION_COST_PCT, HMAX

# FinRL imports
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv

# เพิ่ม import สำหรับ PPO
from stable_baselines3 import PPO

# Lineage 2 class names
LINEAGE2_CLASSES = [
    "Warcryer", "Overlord", "Warlord", "Gladiator", "Warlock", "Sorcerer",
    "Necromancer", "Warlord", "Gladiator", "Warlock", "Sorcerer", "Necromancer",
    "Bishop", "Prophet", "ElvenElder", "ShillienElder", "Cardinal", "Hierophant",
    "EvaSaint", "ShillienSaint", "Dominator", "Doomcryer", "GrandKhavatari",
    "FortuneSeeker", "Maestro", "Doombringer", "SoulHound", "Trickster",
    "Inspector", "Judicator", "SigelKnight", "TyrrWarrior", "OthellRogue",
    "YulArcher", "FeohWizard", "IssEnchanter", "WynnSummoner", "AeoreHealer"
]

def get_model_info(model_path):
    """Get model information"""
    if os.path.exists(model_path):
        stats = os.stat(model_path)
        return {
            "last_modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size": f"{stats.st_size/1024:.1f} KB"
        }
    return None

def get_training_params(model_type="PPO", exchange="binance", grade="N"):
    """Get training parameters based on model type, exchange and grade"""
    base_params = {
        "PPO": {
            "learning_rate": 5e-5,
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "max_grad_norm": 0.3,
            "ent_coef": 0.005,
            "vf_coef": 0.3,
            "target_kl": 0.01
        },
        "PPO (Simple)": {
            "learning_rate": 5e-5,
            "batch_size": 128,
            "n_steps": 1024,
            "gamma": 0.99,
            "gae_lambda": 0.95
        }
    }
    
    # เพิ่มพารามิเตอร์เฉพาะ exchange
    exchange_params = {
        "binance": {
            "min_trade_amount": 0.001,
            "fee": 0.001,
            "max_leverage": 20
        },
        "bybit": {
            "min_trade_amount": 0.001,
            "fee": 0.0006,
            "max_leverage": 100
        },
        "okx": {
            "min_trade_amount": 0.001,
            "fee": 0.0008,
            "max_leverage": 125
        }
    }

    # กำหนดค่าตาม GRADE ที่ปรับปรุงใหม่
    grade_params = {
        "N": {
            "steps": 5000,
            "n_steps": 256,
            "batch_size": 32,
            "n_epochs": 2,
            "learning_rate": 1e-5,
            "show_advanced": False
        },
        "D": {
            "steps": 10000,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 3,
            "learning_rate": 2e-5,
            "show_advanced": False
        },
        "C": {
            "steps": 20000,
            "n_steps": 768,
            "batch_size": 96,
            "n_epochs": 4,
            "learning_rate": 3e-5,
            "show_advanced": True
        },
        "B": {
            "steps": 30000,
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 5,
            "learning_rate": 4e-5,
            "show_advanced": True
        },
        "A": {
            "steps": 50000,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 8,
            "learning_rate": 5e-5,
            "show_advanced": True
        },
        "S": {
            "steps": 100000,
            "n_steps": 4096,
            "batch_size": 512,
            "n_epochs": 10,
            "learning_rate": 1e-4,
            "show_advanced": True
        }
    }
    
    params = base_params[model_type].copy()
    params.update(exchange_params[exchange])
    params.update(grade_params[grade])
    return params

def show_gpu_info():
    """แสดงข้อมูล GPU ในส่วน UI"""
    gpu_info = check_gpu_availability()
    
    if gpu_info["available"]:
        st.success(f"✅ GPU พร้อมใช้งาน: พบ {gpu_info['count']} GPU")
        
        for i in range(gpu_info["count"]):
            with st.expander(f"GPU {i+1}: {gpu_info['devices'][i]}", expanded=True):
                memory = gpu_info["memory"][i]
                st.markdown(f"""
                - 🎮 ชื่อ: {gpu_info['devices'][i]}
                - 💾 หน่วยความจำที่ใช้: {memory['allocated']} MB
                - 📦 หน่วยความจำที่จอง: {memory['reserved']} MB
                """)
    else:
        st.warning("""
        ⚠️ ไม่พบ GPU ที่สามารถใช้งานได้
        - การเทรนจะใช้ CPU ซึ่งอาจใช้เวลานานกว่า
        - แนะนำให้ใช้ GPU สำหรับการเทรนที่มี steps มาก
        """)

def check_gpu_availability():
    """ตรวจสอบการใช้งาน GPU"""
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "memory": []
    }
    
    if cuda_available():
        gpu_info["available"] = True
        gpu_info["count"] = cuda_device_count()
        
        for i in range(gpu_info["count"]):
            device_name = cuda_get_device_name(i)
            gpu_info["devices"].append(device_name)
            
            # ตรวจสอบหน่วยความจำ GPU
            try:
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
                gpu_info["memory"].append({
                    "allocated": round(memory_allocated, 2),
                    "reserved": round(memory_reserved, 2)
                })
            except:
                gpu_info["memory"].append({
                    "allocated": "N/A",
                    "reserved": "N/A"
                })
    
    return gpu_info

def show_training_parameters(model_type, exchange, grade):
    """แสดงและรับค่าพารามิเตอร์การเทรน"""
    params = get_training_params(model_type, exchange, grade)
    
    with st.expander("🔧 Training Parameters", expanded=True):
        # Exchange Information
        st.markdown(f"""
        ### 📊 Exchange Information
        - Exchange: {exchange.upper()}
        - Minimum Trade Amount: {params['min_trade_amount']} BTC
        - Trading Fee: {params['fee']*100}%
        - Maximum Leverage: {params['max_leverage']}x
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input(
                "Training Steps",
                min_value=1000,
                value=params['steps'],
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
        
        # Advanced options
        if params['show_advanced']:
            if st.checkbox("🔍 Show Advanced Options"):
                st.warning("""
                ⚠️ **คำเตือน:** การปรับค่าเหล่านี้อาจส่งผลต่อประสิทธิภาพของโมเดล 
                แนะนำให้ทดลองกับค่าเดิมก่อน
                """)
                
                st.markdown("### 🎯 Model Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=1e-6,
                        max_value=1e-2,
                        value=params['learning_rate'],
                        format="%.0e",
                        help="Model's learning rate"
                    )
                    n_steps = st.number_input(
                        "Steps per Update",
                        min_value=512,
                        max_value=4096,
                        value=params['n_steps'],
                        step=512,
                        help="จำนวน steps ต่อการอัพเดท"
                    )
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=32,
                        max_value=512,
                        value=params['batch_size'],
                        step=32,
                        help="Training batch size"
                    )
                with col2:
                    n_epochs = st.number_input(
                        "Number of Epochs",
                        min_value=1,
                        max_value=20,
                        value=params['n_epochs'],
                        step=1,
                        help="จำนวน epochs ต่อการอัพเดท"
                    )
                    gamma = st.number_input(
                        "Gamma (Discount Factor)",
                        min_value=0.8,
                        max_value=0.999,
                        value=params['gamma'],
                        step=0.001,
                        help="Discount factor for future rewards"
                    )
                    gae_lambda = st.number_input(
                        "GAE Lambda",
                        min_value=0.8,
                        max_value=0.999,
                        value=params['gae_lambda'],
                        step=0.001,
                        help="GAE parameter"
                    )
                
                st.markdown("### ⚖️ Environment Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    reward_scaling = st.slider(
                        "Reward Scaling",
                        min_value=1e-5,
                        max_value=1e-2,
                        value=1e-4,
                        format="%.0e",
                        help="ปรับความสำคัญของ reward"
                    )
                    clip_range = st.slider(
                        "Clip Range",
                        min_value=0.1,
                        max_value=0.5,
                        value=params['clip_range'],
                        step=0.1,
                        help="PPO clip range"
                    )
                with col2:
                    ent_coef = st.slider(
                        "Entropy Coefficient",
                        min_value=0.0,
                        max_value=0.1,
                        value=params['ent_coef'],
                        step=0.001,
                        help="Entropy coefficient for exploration"
                    )
                    vf_coef = st.slider(
                        "Value Function Coefficient",
                        min_value=0.1,
                        max_value=1.0,
                        value=params['vf_coef'],
                        step=0.1,
                        help="Value function coefficient"
                    )
                
                print_verbosity = st.selectbox(
                    "Print Verbosity",
                    [1, 2],
                    index=1,
                    help="ระดับความละเอียดของการแสดงผล (2 = แสดงผลละเอียด)"
                )
    
    return {
        "steps": steps,
        "save_interval": save_interval,
        "learning_rate": learning_rate if 'learning_rate' in locals() else params['learning_rate'],
        "n_steps": n_steps if 'n_steps' in locals() else params['n_steps'],
        "batch_size": batch_size if 'batch_size' in locals() else params['batch_size'],
        "n_epochs": n_epochs if 'n_epochs' in locals() else params['n_epochs'],
        "gamma": gamma if 'gamma' in locals() else params['gamma'],
        "gae_lambda": gae_lambda if 'gae_lambda' in locals() else params['gae_lambda'],
        "clip_range": clip_range if 'clip_range' in locals() else params['clip_range'],
        "ent_coef": ent_coef if 'ent_coef' in locals() else params['ent_coef'],
        "vf_coef": vf_coef if 'vf_coef' in locals() else params['vf_coef'],
        "reward_scaling": reward_scaling if 'reward_scaling' in locals() else 1e-4,
        "print_verbosity": print_verbosity if 'print_verbosity' in locals() else 2
    }

def simple_train_agent_ui():
    """UI แบบง่ายสำหรับการเทรน agent"""
    st.header("🎯 Train RL Agent")
    
    # แสดงข้อมูล GPU
    show_gpu_info()
    
    # Check existing models
    existing_models = []
    if os.path.exists(MODEL_DIR):
        existing_models = [f.replace("minimal_crypto_", "").replace(".zip", "") 
                          for f in os.listdir(MODEL_DIR) 
                          if f.startswith("minimal_crypto_") and f.endswith(".zip")]
    
    # Training mode selection
    train_mode = st.radio(
        "Training Mode",
        ["Create Agent", "Continue Training"],
        help="Choose whether to create a new agent or continue training an existing one"
    )
    
    # Exchange selection
    exchange = st.selectbox(
        "Select Exchange",
        ["binance", "bybit", "okx"],
        help="Select the exchange to train the agent for"
    )

    # Grade selection
    grade = st.selectbox(
        "Training Grade",
        ["N", "D", "C", "B", "A", "S"],
        help="""Select training grade:
        N = Basic test (minimal parameters)
        D = Development grade
        C = Standard grade
        B = Advanced grade
        A = Professional grade (50,000 steps)
        S = Master grade (100,000 steps)"""
    )

    # Data file selection
    data_files = []
    if os.path.exists(DATA_DIR):
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not data_files:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูลในโฟลเดอร์ data กรุณาโหลดข้อมูลก่อน")
        return

    selected_data_file = st.selectbox(
        "เลือกไฟล์ข้อมูลสำหรับการเทรน",
        data_files,
        help="เลือกไฟล์ข้อมูลที่ต้องการใช้ในการเทรน"
    )

    # แสดงข้อมูลของไฟล์ที่เลือก
    if selected_data_file:
        file_path = os.path.join(DATA_DIR, selected_data_file)
        try:
            df = pd.read_csv(file_path)
            st.info(f"""
            📊 ข้อมูลไฟล์ที่เลือก:
            - 📄 ชื่อไฟล์: {selected_data_file}
            - 📅 จำนวนข้อมูล: {len(df):,} แถว
            - 💱 สกุลเงิน: {', '.join(df['symbol'].unique()) if 'symbol' in df.columns else 'N/A'}
            - ⏱️ ช่วงเวลา: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'} ถึง {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}
            """)
        except Exception as e:
            st.error(f"⚠️ เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")
            return
    
    if train_mode == "Continue Training":
        if not existing_models:
            st.warning("⚠️ No existing models found. Please create a new agent first.")
            return
            
        # Model selection for continuing training
        model_to_continue = st.selectbox(
            "Select Model to Continue Training",
            existing_models,
            help="Choose an existing model to continue training"
        )
        
        # Show existing model info
        model_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_to_continue}.zip")
        model_info = get_model_info(model_path)
        if model_info:
            st.info(f"📝 Last trained: {model_info['last_modified']} | Size: {model_info['size']}")
        
        # Use the same type as the existing model
        model_type = "PPO (Simple)" if "simple" in model_to_continue else "PPO"
        st.write(f"Model Type: {model_type}")
        
    else:  # Create Agent
        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            ["PPO", "PPO (Simple)"],
            help="PPO (Simple) uses fewer parameters and may train faster"
        )
    
    # Get training parameters
    params = show_training_parameters(model_type, exchange, grade)
    
    # Training button with simplified functionality
    if train_mode == "Continue Training":
        start_button = st.button("🚀 Continue Training")
        if start_button:
            st.info(f"""
            📋 **การเทรนต่อ:** {model_to_continue}
            - Model Type: {model_type}
            - Exchange: {exchange.upper()}
            - Grade: {grade}
            - Steps: {params['steps']:,}
            
            ⚠️ **หมายเหตุ:** ฟีเจอร์การเทรนอยู่ระหว่างการพัฒนา
            กรุณาใช้ทางเลือกข้างล่างแทน
            """)
    else:
        start_button = st.button("🚀 Create Agent")
        if start_button:
            # สร้างชื่อ agent ใหม่
            symbol = df['symbol'].unique()[0] if 'symbol' in df.columns else "BTC_USDT"
            agent_name = f"{symbol.replace('/', '_')}-{model_type}-{random.choice(LINEAGE2_CLASSES)}"
            
            st.info(f"""
            📋 **การสร้าง Agent ใหม่:** {agent_name}
            - Model Type: {model_type}
            - Exchange: {exchange.upper()}
            - Grade: {grade}
            - Steps: {params['steps']:,}
            - Data: {selected_data_file}
            
            ⚠️ **หมายเหตุ:** ฟีเจอร์การเทรนอยู่ระหว่างการพัฒนา
            กรุณาใช้ทางเลือกข้างล่างแทน
            """)
    
    st.markdown("---")
    
    st.subheader("🔄 ทางเลือกสำหรับการเทรน Agent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📁 ใช้ไฟล์ Python หลัก:**
        
        ```bash
        # เทรน agent แบบง่าย
        python main.py
        
        # เทรน agent แบบขั้นสูง  
        python simple_advanced_agent.py
        ```
        """)
    
    with col2:
        st.info("""
        **📊 ใช้ Jupyter Notebooks:**
        
        - `notebooks/1_data_preparation.ipynb`
        - `notebooks/2_agent_creation.ipynb`
        - `notebooks/3_agent_training.ipynb`
        - `notebooks/4_agent_evaluation.ipynb`
        """)

# เรียกใช้ UI
simple_train_agent_ui() 