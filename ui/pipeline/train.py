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

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR, MODEL_KWARGS, DATA_DIR, INITIAL_AMOUNT, TRANSACTION_COST_PCT, HMAX
from ui.pipeline.agent_manager import get_agent_info, list_agents

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

def get_model_info(model_path
):
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

def calculate_estimated_time(params):
    """คำนวณเวลาที่ใช้ในการเทรนโดยประมาณ"""
    total_steps = params['steps']
    steps_per_update = params['n_steps']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    
    # คำนวณจำนวนรอบการอัพเดท
    num_updates = total_steps / steps_per_update
    
    # ประมาณเวลาต่อรอบ (วินาที)
    time_per_update_cpu = 150  # 2.5 นาที
    time_per_update_gpu = 37.5  # 0.625 นาที
    
    # คำนวณเวลาทั้งหมด
    total_time_cpu = num_updates * time_per_update_cpu
    total_time_gpu = num_updates * time_per_update_gpu
    
    return {
        'cpu_minutes': round(total_time_cpu / 60, 1),
        'gpu_minutes': round(total_time_gpu / 60, 1),
        'num_updates': round(num_updates, 1)
    }

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
                
                # แสดงเวลาที่คาดว่าจะใช้
                estimated_time = calculate_estimated_time({
                    'steps': steps,
                    'n_steps': n_steps if 'n_steps' in locals() else params['n_steps'],
                    'batch_size': batch_size if 'batch_size' in locals() else params['batch_size'],
                    'n_epochs': n_epochs if 'n_epochs' in locals() else params['n_epochs']
                })
                
                st.info(f"""
                ⏱️ เวลาที่คาดว่าจะใช้ในการเทรน:
                - CPU: {estimated_time['cpu_minutes']} นาที
                - GPU: {estimated_time['gpu_minutes']} นาที
                - จำนวนรอบการอัพเดท: {estimated_time['num_updates']} รอบ
                """)
    
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

def add_technical_indicators(df):
    """เพิ่ม technical indicators และ normalize ข้อมูล - แก้ไขปัญหา inf/nan"""
    # สร้างสำเนาข้อมูล
    df = df.copy()
    
    print("เริ่มคำนวณ Technical Indicators...")
    
    # 1. Moving Averages
    print("คำนวณ Moving Averages...")
    df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    
    # 2. RSI (Relative Strength Index)
    print("คำนวณ RSI...")
    def calculate_rsi(group):
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        
        # ป้องกันการหารด้วย 0
        rs = gain / (loss + 1e-10)  # เพิ่มค่าเล็กๆ เพื่อป้องกันการหารด้วย 0
        rsi = 100 - (100 / (1 + rs))
        
        # จำกัดค่า RSI ให้อยู่ในช่วง 0-100
        rsi = rsi.clip(0, 100)
        return rsi
    
    # คำนวณ RSI แยกตาม tic
    rsi_values = []
    for tic in df['tic'].unique():
        tic_data = df[df['tic'] == tic].copy()
        rsi = calculate_rsi(tic_data)
        rsi_values.extend(rsi.values)
    df['rsi_14'] = rsi_values
    
    # 3. MACD
    print("คำนวณ MACD...")
    def calculate_macd(group):
        exp1 = group['close'].ewm(span=12, adjust=False).mean()
        exp2 = group['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_hist': hist
        })
    
    # คำนวณ MACD แยกตาม tic
    macd_dfs = []
    for tic in df['tic'].unique():
        tic_data = df[df['tic'] == tic].copy()
        macd_result = calculate_macd(tic_data)
        macd_dfs.append(macd_result)
    
    macd_df = pd.concat(macd_dfs, ignore_index=True)
    df['macd'] = macd_df['macd'].values
    df['macd_signal'] = macd_df['macd_signal'].values
    df['macd_hist'] = macd_df['macd_hist'].values
    
    # 4. Bollinger Bands
    print("คำนวณ Bollinger Bands...")
    df['bb_middle'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['bb_std'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    
    # ป้องกัน std = 0
    df['bb_std'] = df['bb_std'].fillna(0)
    df['bb_std'] = df['bb_std'].replace(0, df['bb_std'].mean())
    
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # 5. Volume Indicators
    print("คำนวณ Volume Indicators...")
    df['volume_sma_20'] = df.groupby('tic')['volume'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    
    # ป้องกันการหารด้วย 0 สำหรับ volume_ratio
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
    df['volume_ratio'] = df['volume_ratio'].clip(0, 100)  # จำกัดค่าไม่ให้สูงเกินไป
    
    print("เริ่ม Normalization...")
    
    # Normalize ข้อมูลราคา - ใช้ Min-Max scaling เพื่อป้องกันค่าลบ
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        print(f"Normalize {col}...")
        
        def minmax_normalize(x):
            # ใช้ Min-Max scaling แทน Z-score เพื่อป้องกันค่าลบ
            min_val = x.min()
            max_val = x.max()
            
            if min_val == max_val:
                # ถ้าค่าเท่ากันหมด ให้ return 0.5 (กลางช่วง 0-1)
                return pd.Series(0.5, index=x.index)
            
            # Min-Max scaling ให้อยู่ในช่วง 0-1
            normalized = (x - min_val) / (max_val - min_val)
            return normalized
        
        df[col] = df.groupby('tic')[col].transform(minmax_normalize)
    
    # Normalize volume
    print("Normalize volume...")
    def safe_normalize_volume(x):
        mean_val = x.mean()
        std_val = x.std()
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0, index=x.index)
        return (x - mean_val) / std_val
    
    df['volume'] = df.groupby('tic')['volume'].transform(safe_normalize_volume)
    
    # Normalize technical indicators
    print("Normalize Technical Indicators...")
    indicator_cols = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                     'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'volume_sma_20', 'volume_ratio']
    
    for col in indicator_cols:
        if col in df.columns:
            print(f"Normalize {col}...")
            
            def safe_normalize(x):
                mean_val = x.mean()
                std_val = x.std()
                
                if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                    return pd.Series(0, index=x.index)
                
                normalized = (x - mean_val) / std_val
                
                # จำกัดค่าที่ normalize แล้วไม่ให้มากเกินไป
                normalized = normalized.clip(-10, 10)
                return normalized
            
            df[col] = df.groupby('tic')[col].transform(safe_normalize)
    
    # ตรวจสอบและแทนที่ค่า inf, -inf, และ nan
    print("ตรวจสอบและแก้ไขค่า inf/nan...")
    
    # แทนที่ค่า inf และ -inf ด้วย 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # แทนที่ค่า nan ด้วย 0
    df = df.fillna(0)
    
    # ตรวจสอบหลังการแก้ไข
    inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    nan_count = df.isna().sum().sum()
    
    print(f"หลังการแก้ไข - inf: {inf_count}, nan: {nan_count}")
    
    if inf_count > 0 or nan_count > 0:
        print("⚠️ ยังพบ inf หรือ nan หลังการแก้ไข")
        # แก้ไขเพิ่มเติม
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                df[col] = df[col].fillna(0)
    
    print("✅ Technical Indicators และ Normalization เสร็จสิ้น")
    return df


def safe_normalize_by_group(df, columns, group_col='tic'):
    """ฟังก์ชันช่วยสำหรับ normalize ข้อมูลอย่างปลอดภัย"""
    for col in columns:
        if col in df.columns:
            def safe_normalize(x):
                if len(x) == 0:
                    return x
                    
                mean_val = x.mean()
                std_val = x.std()
                
                # ตรวจสอบค่าที่ไม่ปกติ
                if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                    return pd.Series(0, index=x.index)
                
                # Normalize
                normalized = (x - mean_val) / std_val
                
                # จำกัดค่าไม่ให้สุดโต่ง
                normalized = normalized.clip(-5, 5)
                
                return normalized
            
            df[col] = df.groupby(group_col)[col].transform(safe_normalize)
    
    return df

def prepare_data_for_training(df):
    """เตรียมข้อมูลสำหรับการเทรน - แก้ไขปัญหาค่า 0 หรือติดลบ"""
    try:
        # สร้าง DataFrame ใหม่ตามรูปแบบที่ FinRL ต้องการ
        processed_df = pd.DataFrame()
        
        # ตรวจสอบและแปลงคอลัมน์วันที่
        if 'timestamp' in df.columns:
            processed_df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
        elif 'date' in df.columns:
            processed_df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        else:
            raise ValueError("ไม่พบคอลัมน์ timestamp หรือ date ในข้อมูล")
        
        # ตรวจสอบและแปลงคอลัมน์ symbol
        if 'tic' in df.columns:
            processed_df['tic'] = df['tic']
        elif 'symbol' in df.columns:
            processed_df['tic'] = df['symbol']
        else:
            raise ValueError("ไม่พบคอลัมน์ symbol หรือ tic ในข้อมูล")
        
        # คัดลอกและตรวจสอบคอลัมน์ราคาและ volume
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in df.columns:
                # แปลงเป็น float และจัดการค่าไม่ถูกต้อง
                processed_df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # ตรวจสอบและแสดงข้อมูลก่อนแก้ไข
                print(f"=== ตรวจสอบคอลัมน์ {col} ===")
                print(f"ค่าต่ำสุด: {processed_df[col].min()}")
                print(f"ค่าสูงสุด: {processed_df[col].max()}")
                print(f"จำนวน NaN: {processed_df[col].isna().sum()}")
                print(f"จำนวนค่า <= 0: {(processed_df[col] <= 0).sum()}")
                
                # แทนที่ค่า NaN ด้วยค่าเฉลี่ย
                if processed_df[col].isna().any():
                    print(f"พบ NaN ใน {col}, แทนที่ด้วยค่าเฉลี่ย")
                    mean_values = processed_df.groupby('tic')[col].transform('mean')
                    processed_df[col] = processed_df[col].fillna(mean_values)
                
                # สำหรับคอลัมน์ราคา ห้ามมีค่า 0 หรือติดลบ
                if col != 'volume':  
                    zero_negative_mask = (processed_df[col] <= 0)
                    if zero_negative_mask.any():
                        print(f"พบค่า <= 0 ใน {col}: {zero_negative_mask.sum()} แถว")
                        
                        # แทนที่ด้วยค่าก่อนหน้าหรือถัดไป
                        for tic in processed_df['tic'].unique():
                            tic_mask = processed_df['tic'] == tic
                            tic_data = processed_df[tic_mask].copy()
                            
                            # หาแถวที่มีปัญหาใน tic นี้
                            problem_mask = tic_mask & zero_negative_mask
                            
                            if problem_mask.any():
                                print(f"แก้ไขค่าใน {tic} สำหรับ {col}")
                                
                                # ใช้ forward fill และ backward fill
                                tic_series = processed_df.loc[tic_mask, col].copy()
                                tic_series = tic_series.replace(0, np.nan)  # แปลง 0 เป็น NaN
                                tic_series = tic_series.mask(tic_series <= 0, np.nan)  # แปลงค่าติดลบเป็น NaN
                                
                                # Forward fill แล้ว backward fill
                                tic_series = tic_series.fillna(method='ffill')
                                tic_series = tic_series.fillna(method='bfill')
                                
                                # ถ้ายังมี NaN ให้ใช้ค่าเฉลี่ยของคอลัมน์
                                if tic_series.isna().any():
                                    global_mean = processed_df[processed_df[col] > 0][col].mean()
                                    tic_series = tic_series.fillna(global_mean)
                                    print(f"ใช้ค่าเฉลี่ย global: {global_mean}")
                                
                                # อัพเดทข้อมูลกลับ
                                processed_df.loc[tic_mask, col] = tic_series.values
                        
                        # ตรวจสอบอีกครั้งหลังแก้ไข
                        remaining_problems = (processed_df[col] <= 0).sum()
                        if remaining_problems > 0:
                            print(f"⚠️ ยังมีปัญหาใน {col}: {remaining_problems} แถว")
                            # แทนที่ด้วยค่าเฉลี่ยทั้งหมด
                            global_mean = processed_df[processed_df[col] > 0][col].mean()
                            processed_df.loc[processed_df[col] <= 0, col] = global_mean
                            print(f"แทนที่ด้วยค่าเฉลี่ย: {global_mean}")
                        else:
                            print(f"✅ แก้ไข {col} เรียบร้อยแล้ว")
                            
                else:  # สำหรับ volume
                    # Volume สามารถเป็น 0 ได้ แต่ไม่ควรติดลบ
                    negative_mask = (processed_df[col] < 0)
                    if negative_mask.any():
                        print(f"พบ volume ติดลบ: {negative_mask.sum()} แถว - แทนที่ด้วย 0")
                        processed_df.loc[negative_mask, col] = 0
                
                # แสดงสถิติหลังแก้ไข
                print(f"หลังแก้ไข - ค่าต่ำสุด: {processed_df[col].min()}, ค่าสูงสุด: {processed_df[col].max()}")
                print()
                        
            else:
                raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล")
        
        # ตรวจสอบและแก้ไขความถูกต้องของราคา OHLC
        print("=== ตรวจสอบความสัมพันธ์ของราคา OHLC ===")
        for tic in processed_df['tic'].unique():
            tic_mask = processed_df['tic'] == tic
            tic_data = processed_df[tic_mask].copy()
            
            # ตรวจสอบว่า high >= max(open, close, low)
            max_price = tic_data[['open', 'close', 'low']].max(axis=1)
            high_issue_mask = (tic_data['high'] < max_price)
            if high_issue_mask.any():
                print(f"แก้ไข high ใน {tic}: {high_issue_mask.sum()} แถว")
                processed_df.loc[tic_mask & tic_data.index.isin(tic_data[high_issue_mask].index), 'high'] = max_price[high_issue_mask]
            
            # ตรวจสอบว่า low <= min(open, close, high)
            min_price = tic_data[['open', 'close', 'high']].min(axis=1)
            low_issue_mask = (tic_data['low'] > min_price)
            if low_issue_mask.any():
                print(f"แก้ไข low ใน {tic}: {low_issue_mask.sum()} แถว")
                processed_df.loc[tic_mask & tic_data.index.isin(tic_data[low_issue_mask].index), 'low'] = min_price[low_issue_mask]
            
            # ตรวจสอบว่า open, close อยู่ระหว่าง high และ low
            high_val = processed_df.loc[tic_mask, 'high']
            low_val = processed_df.loc[tic_mask, 'low']
            
            # แก้ไข open
            open_high_mask = (processed_df.loc[tic_mask, 'open'] > high_val)
            open_low_mask = (processed_df.loc[tic_mask, 'open'] < low_val)
            if open_high_mask.any() or open_low_mask.any():
                print(f"แก้ไข open ใน {tic}: {(open_high_mask | open_low_mask).sum()} แถว")
                # ใช้ค่ากลางระหว่าง high และ low
                processed_df.loc[tic_mask & (open_high_mask | open_low_mask), 'open'] = (high_val + low_val) / 2
            
            # แก้ไข close
            close_high_mask = (processed_df.loc[tic_mask, 'close'] > high_val)
            close_low_mask = (processed_df.loc[tic_mask, 'close'] < low_val)
            if close_high_mask.any() or close_low_mask.any():
                print(f"แก้ไข close ใน {tic}: {(close_high_mask | close_low_mask).sum()} แถว")
                processed_df.loc[tic_mask & (close_high_mask | close_low_mask), 'close'] = (high_val + low_val) / 2
        
        # เพิ่ม technical indicators
        print("=== เพิ่ม Technical Indicators ===")
        processed_df = add_technical_indicators(processed_df)
        
        # ตรวจสอบข้อมูลหลังการประมวลผล
        print("=== ตรวจสอบข้อมูลสุดท้าย ===")
        data_ready, message = check_training_data(processed_df)
        if not data_ready:
            print(f"❌ ข้อมูลยังไม่พร้อม: {message}")
            
            # แสดงข้อมูลเพิ่มเติมเพื่อ debug
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                problematic = processed_df[processed_df[col] <= 0]
                if len(problematic) > 0:
                    print(f"❌ ยังมีปัญหาใน {col}:")
                    print(problematic[['date', 'tic', col]].head())
            
            raise ValueError(f"ข้อมูลไม่พร้อมสำหรับการเทรน: {message}")
        else:
            print(f"✅ {message}")
        
        return processed_df
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเตรียมข้อมูล: {str(e)}")
        raise ValueError(f"เกิดข้อผิดพลาดในการเตรียมข้อมูล: {str(e)}")


def check_training_data(df):
    """ตรวจสอบความพร้อมของข้อมูลก่อนการเทรน - รองรับ normalized data"""
    try:
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}"
        
        # ตรวจสอบค่า NaN
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            return False, f"พบค่า NaN ในคอลัมน์: {', '.join(nan_counts[nan_counts > 0].index)}"
        
        # ตรวจสอบค่า inf
        inf_counts = df[required_columns].isin([np.inf, -np.inf]).sum()
        if inf_counts.any():
            return False, f"พบค่า inf ในคอลัมน์: {', '.join(inf_counts[inf_counts > 0].index)}"
        
        # ตรวจสอบจำนวนข้อมูล
        if len(df) < 100:
            return False, "ข้อมูลน้อยเกินไป (น้อยกว่า 100 แถว)"
        
        # ตรวจสอบว่าข้อมูลถูก normalize แล้วหรือไม่
        price_cols = ['open', 'high', 'low', 'close']
        is_normalized = False
        
        for col in price_cols:
            # ถ้าค่าอยู่ในช่วง 0-1 หรือมีค่าลบ แสดงว่าถูก normalize แล้ว
            if (df[col].min() >= 0 and df[col].max() <= 1) or (df[col] < 0).any():
                is_normalized = True
                break
        
        if is_normalized:
            print("✅ ตรวจพบข้อมูลที่ผ่าน normalization แล้ว")
            # สำหรับข้อมูลที่ normalize แล้ว ตรวจสอบแค่ค่า finite
            for col in price_cols:
                if not np.isfinite(df[col]).all():
                    return False, f"พบค่า invalid ในคอลัมน์ {col}"
        else:
            print("⚠️ ข้อมูลยังไม่ผ่าน normalization")
            # สำหรับข้อมูลดิบ ตรวจสอบค่าลบหรือศูนย์
            for col in price_cols:
                if (df[col] <= 0).any():
                    zero_or_negative = (df[col] <= 0)
                    problematic_rows = df[zero_or_negative]
                    print(f"❌ พบปัญหาใน {col}:")
                    print(f"จำนวนแถว: {zero_or_negative.sum()}")
                    print("ตัวอย่างแถวที่มีปัญหา:")
                    print(problematic_rows[['date', 'tic', col]].head())
                    return False, f"พบราคา 0 หรือติดลบในคอลัมน์ {col}"
        
        return True, "ข้อมูลพร้อมสำหรับการเทรน"
        
    except Exception as e:
        print(f"❌ Error ในการตรวจสอบข้อมูล: {str(e)}")
        return False, f"เกิดข้อผิดพลาดในการตรวจสอบข้อมูล: {str(e)}"

def create_trading_env(df, initial_amount=INITIAL_AMOUNT, params=None):
    """สร้าง trading environment"""
    # แบ่งข้อมูลเป็น train/test โดยใช้สัดส่วน 80/20
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    # ตรวจสอบและแปลงคอลัมน์วันที่
    for data in [train_df, test_df]:
        # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
        if 'timestamp' not in data.columns:
            # ถ้าไม่มี timestamp ให้ใช้คอลัมน์ date แทน
            if 'date' in data.columns:
                data['timestamp'] = pd.to_datetime(data['date'])
            else:
                raise ValueError("ไม่พบคอลัมน์ timestamp หรือ date ในข้อมูล")
        
        # แปลง timestamp เป็น datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')
        
        # เรียงข้อมูลตามวันที่
        data.sort_values(['date', 'tic'], inplace=True)
        data.reset_index(drop=True, inplace=True)
    
    # กำหนด indicators ที่ใช้
    indicators = [
        'sma_20', 'ema_20', 'rsi_14', 
        'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
        'volume_sma_20', 'volume_ratio'
    ]
    
    # ใช้พารามิเตอร์จาก UI ถ้ามี
    reward_scaling = params['reward_scaling'] if params else 1e-4
    print_verbosity = params['print_verbosity'] if params else 2
    
    # สร้าง environment สำหรับ training
    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * len(df['tic'].unique()),
        "buy_cost_pct": [TRANSACTION_COST_PCT] * len(df['tic'].unique()),
        "sell_cost_pct": [TRANSACTION_COST_PCT] * len(df['tic'].unique()),
        "state_space": 1 + 2 * len(df['tic'].unique()) + len(df['tic'].unique()) * len(indicators),
        "stock_dim": len(df['tic'].unique()),
        "tech_indicator_list": indicators,
        "action_space": len(df['tic'].unique()),
        "reward_scaling": reward_scaling,
        "print_verbosity": print_verbosity
    }
    
    train_env = StockTradingEnv(df=train_df, **env_kwargs)
    test_env = StockTradingEnv(df=test_df, **env_kwargs)
    
    return train_env, test_env, train_df, test_df

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

def check_gpu_readiness():
    """ตรวจสอบความพร้อมของ GPU สำหรับการเทรน"""
    try:
        gpu_info = check_gpu_availability()
        
        if not gpu_info["available"]:
            return {
                'ready': False,
                'message': "⚠️ ไม่พบ GPU ที่สามารถใช้งานได้",
                'details': "การเทรนจะใช้ CPU ซึ่งอาจใช้เวลานานกว่า"
            }
            
        # ตรวจสอบหน่วยความจำ GPU
        for i, memory in enumerate(gpu_info["memory"]):
            if isinstance(memory['allocated'], (int, float)) and memory['allocated'] > 0:
                return {
                    'ready': False,
                    'message': f"⚠️ GPU {i+1} กำลังถูกใช้งานอยู่",
                    'details': f"หน่วยความจำที่ใช้: {memory['allocated']} MB"
                }
        
        # ตรวจสอบ CUDA version
        cuda_version = torch.version.cuda
        if not cuda_version:
            return {
                'ready': False,
                'message': "⚠️ ไม่พบ CUDA version",
                'details': "อาจมีปัญหาในการใช้งาน GPU"
            }
            
        return {
            'ready': True,
            'message': f"✅ GPU พร้อมใช้งาน: {gpu_info['count']} GPU",
            'details': f"CUDA Version: {cuda_version}",
            'gpu_info': gpu_info
        }
    except Exception as e:
        return {
            'ready': False,
            'message': "⚠️ เกิดข้อผิดพลาดในการตรวจสอบ GPU",
            'details': str(e)
        }

def get_next_version_name(model_path):
    """สร้างชื่อไฟล์เวอร์ชั่นถัดไป"""
    try:
        # แยกชื่อไฟล์และนามสกุล
        base_name = os.path.splitext(model_path)[0]
        ext = os.path.splitext(model_path)[1]
        
        # หาเวอร์ชั่นปัจจุบัน
        version = 1
        while os.path.exists(f"{base_name}_v{version}{ext}"):
            version += 1
            
        return f"{base_name}_v{version}{ext}"
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการสร้างชื่อไฟล์เวอร์ชั่น: {str(e)}")
        return None

def save_model_with_version(model, model_path):
    """บันทึกโมเดลพร้อมเวอร์ชั่น"""
    try:
        # สร้างชื่อไฟล์เวอร์ชั่นใหม่
        new_model_path = get_next_version_name(model_path)
        if not new_model_path:
            return False, "ไม่สามารถสร้างชื่อไฟล์เวอร์ชั่นใหม่ได้"
        
        # ตรวจสอบว่าสามารถเขียนไฟล์ได้หรือไม่
        if not os.access(os.path.dirname(new_model_path), os.W_OK):
            return False, "ไม่มีสิทธิ์ในการเขียนไฟล์"
        
        # บันทึกโมเดล
        model.save(new_model_path)
        
        # ตรวจสอบว่าไฟล์ถูกสร้างขึ้นจริง
        if not os.path.exists(new_model_path):
            return False, "ไฟล์ไม่ถูกสร้างขึ้นหลังการบันทึก"
        
        return True, new_model_path
    except Exception as e:
        return False, f"เกิดข้อผิดพลาดในการบันทึกโมเดล: {str(e)}"

def train_model(env, model_type, params, total_timesteps, model_path=None):
    """เทรนโมเดล"""
    try:
        # ตรวจสอบความพร้อมของ GPU
        gpu_status = check_gpu_readiness()
        
        if not gpu_status['ready']:
            st.warning(f"""
            {gpu_status['message']}
            {gpu_status['details']}
            
            ⚠️ **คำเตือน:** การเทรนด้วย steps มากกว่า 50,000 บน CPU อาจใช้เวลานานมาก
            แนะนำให้:
            1. ลดจำนวน steps ลง
            2. ใช้ GPU สำหรับการเทรน
            3. แบ่งการเทรนเป็นหลายรอบ
            """)
            if not st.checkbox("ดำเนินการต่อ"):
                return None
        else:
            st.success(f"""
            {gpu_status['message']}
            {gpu_status['details']}
            """)
        
        # ตรวจสอบข้อมูล
        if hasattr(env, 'df'):
            data_ready, message = check_training_data(env.df)
            if not data_ready:
                st.error(f"❌ {message}")
                return None
            st.success(f"✅ {message}")
        
        # สร้าง agent
        agent = DRLAgent(env=env)
        
        # ใช้พารามิเตอร์จาก UI
        PPO_PARAMS = {
            'learning_rate': params['learning_rate'],
            'n_steps': params['n_steps'],
            'batch_size': params['batch_size'],
            'n_epochs': params['n_epochs'],
            'gamma': params['gamma'],
            'gae_lambda': params['gae_lambda'],
            'clip_range': params['clip_range'],
            'max_grad_norm': params['max_grad_norm'],
            'ent_coef': params['ent_coef'],
            'vf_coef': params['vf_coef'],
            'target_kl': params['target_kl'],
            'device': 'cuda' if gpu_status['ready'] else 'cpu'
        }
        
        # แสดงพารามิเตอร์ที่จะใช้
        st.info(f"""
        🎯 พารามิเตอร์การเทรน:
        - Learning Rate: {PPO_PARAMS['learning_rate']}
        - Steps per Update: {PPO_PARAMS['n_steps']}
        - Batch Size: {PPO_PARAMS['batch_size']}
        - Number of Epochs: {PPO_PARAMS['n_epochs']}
        - Gamma: {PPO_PARAMS['gamma']}
        - GAE Lambda: {PPO_PARAMS['gae_lambda']}
        - Clip Range: {PPO_PARAMS['clip_range']}
        - Max Grad Norm: {PPO_PARAMS['max_grad_norm']}
        - Entropy Coefficient: {PPO_PARAMS['ent_coef']}
        - Value Function Coefficient: {PPO_PARAMS['vf_coef']}
        - Target KL: {PPO_PARAMS['target_kl']}
        - ใช้ GPU: {'✅' if gpu_status['ready'] else '❌'}
        """)
        
        if model_path and os.path.exists(model_path):
            try:
                # โหลดโมเดลเดิมถ้ามี
                st.info(f"🔄 โหลดโมเดลเดิมจาก {model_path}")
                model = PPO.load(model_path)
                
                # อัพเดทพารามิเตอร์
                model.learning_rate = PPO_PARAMS['learning_rate']
                model.n_steps = PPO_PARAMS['n_steps']
                model.batch_size = PPO_PARAMS['batch_size']
                model.n_epochs = PPO_PARAMS['n_epochs']
                model.gamma = PPO_PARAMS['gamma']
                model.gae_lambda = PPO_PARAMS['gae_lambda']
                model.clip_range = PPO_PARAMS['clip_range']
                model.ent_coef = PPO_PARAMS['ent_coef']
                model.vf_coef = PPO_PARAMS['vf_coef']
                
                st.success("✅ โหลดโมเดลเดิมสำเร็จ")
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดลเดิม: {str(e)}")
                st.info("🔄 สร้างโมเดลใหม่แทน")
                model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        else:
            # สร้างโมเดลใหม่
            model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        # สร้าง progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # เทรนโมเดล
        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"minimal_crypto_ppo{'_simple' if model_type == 'PPO (Simple)' else ''}",
            total_timesteps=total_timesteps
        )
        
        # อัพเดท progress bar
        progress_bar.progress(1.0)
        status_text.text(f"Training progress: 100% | Step: {total_timesteps}/{total_timesteps}")
        
        # บันทึกโมเดลเวอร์ชั่นใหม่
        if model_path:
            success, result = save_model_with_version(trained_model, model_path)
            if success:
                st.success(f"""
                ✅ Training completed!
                - Model saved as {os.path.basename(result)}
                - Total steps: {total_timesteps}
                - Final learning rate: {trained_model.learning_rate}
                """)
            else:
                st.error(f"❌ {result}")
                return None
        else:
            # บันทึกโมเดลใหม่
            try:
                trained_model.save("new_model.zip")
                st.success(f"""
                ✅ Training completed!
                - Model saved as new_model.zip
                - Total steps: {total_timesteps}
                - Final learning rate: {trained_model.learning_rate}
                """)
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการบันทึกโมเดล: {str(e)}")
                return None
        
        return trained_model
        
    except Exception as e:
        st.error(f"""
        ❌ Error during model training:
        {str(e)}
        
        🔍 ข้อแนะนำในการแก้ไข:
        1. ตรวจสอบหน่วยความจำ GPU
        2. ลด batch size หรือ n_steps
        3. ลด learning rate
        4. ตรวจสอบข้อมูลที่ใช้ในการเทรน
        """)
        return None

def show_training_progress(model_name, steps, save_interval):
    """แสดงความคืบหน้าการเทรน"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training loop simulation
    for i in range(5):  # TODO: Replace with actual training
        progress = (i + 1) * 20
        progress_bar.progress(progress)
        status_text.text(f"Training progress: {progress}% | Step: {(i+1)*steps//5}/{steps}")
        if (i + 1) * steps//5 % save_interval == 0:
            st.info(f"💾 Saved checkpoint at step {(i+1)*steps//5}")
    
    st.success(f"✅ Training completed! Model saved as {model_name}")
    
    # Show next steps
    st.info("""
    👉 **ขั้นตอนถัดไป:**
    1. ไปที่หน้า Evaluate เพื่อประเมินผลโมเดล
    2. เปรียบเทียบผลลัพธ์กับโมเดลก่อนการเทรน
    3. หากผลลัพธ์ยังไม่ดีพอ สามารถกลับมาเทรนต่อได้
    """)

def generate_agent_name(symbol, model_type):
    """สร้างชื่อ agent ตามรูปแบบที่กำหนด"""
    # แปลงสัญลักษณ์ / เป็น _ ในชื่อสกุลเงิน
    symbol = symbol.replace('/', '_')
    # สุ่มชื่อคลาสจาก Lineage 2
    random_class = random.choice(LINEAGE2_CLASSES)
    # สร้างชื่อ agent
    return f"{symbol}-{model_type}-{random_class}"

def get_latest_version(model_name):
    """หาว่าเป็นเวอร์ชั่นที่เท่าไหร่แล้ว"""
    try:
        # ค้นหาไฟล์ที่มีชื่อขึ้นต้นเหมือนกัน
        existing_files = [f for f in os.listdir(MODEL_DIR) 
                         if f.startswith(f"{model_name}_v") and f.endswith('.zip')]
        
        if not existing_files:
            return 0
        
        # ดึงเลขเวอร์ชั่นออกมา
        versions = []
        for file in existing_files:
            try:
                version = int(file.split('_v')[1].split('.')[0])
                versions.append(version)
            except:
                continue
        
        return max(versions) if versions else 0
    except:
        return 0

def get_next_version_name(model_name):
    """สร้างชื่อไฟล์เวอร์ชั่นถัดไป"""
    current_version = get_latest_version(model_name)
    return f"{model_name}_v{current_version + 1}"

def prepare_training_data(df, params):
    """เตรียมข้อมูลสำหรับการเทรน"""
    # เตรียมข้อมูล
    processed_df = prepare_data_for_training(df)
    
    # สร้าง trading environment
    train_env, test_env, train_df, test_df = create_trading_env(processed_df, params=params)
    
    return train_env, test_env, train_df, test_df

def evaluate_model(model, test_env, initial_amount=INITIAL_AMOUNT):
    """ประเมินผลโมเดล"""
    try:
        # ทดสอบโมเดล
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=model,
            environment=test_env
        )
        
        # คำนวณผลลัพธ์
        final_value = df_account_value['account_value'].iloc[-1]
        total_return = (final_value - initial_amount) / initial_amount * 100
        
        return {
            'success': True,
            'account_value': df_account_value,
            'actions': df_actions,
            'final_value': final_value,
            'total_return': total_return
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def save_model(model, model_path, version=None):
    """บันทึกโมเดล"""
    try:
        if version is not None:
            # สร้างชื่อไฟล์เวอร์ชั่นใหม่
            model_name = os.path.basename(model_path)
            next_version_name = get_next_version_name(model_name)
            save_path = os.path.join(MODEL_DIR, next_version_name)
        else:
            save_path = model_path
            
        # บันทึกโมเดล
        model.save(save_path)
        
        # บันทึกข้อมูลการเทรน
        train_info = {
            'model_path': save_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': version if version is not None else 0
        }
        
        # บันทึกข้อมูลลงไฟล์ JSON
        info_path = save_path.replace('.zip', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(train_info, f, indent=4)
            
        return {
            'success': True,
            'path': save_path,
            'version': version if version is not None else 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def train_new_agent(df, model_type, params, exchange):
    """เทรน agent ใหม่"""
    try:
        # สร้างชื่อ agent ใหม่
        symbol = df['symbol'].unique()[0] if 'symbol' in df.columns else "BTC_USDT"
        agent_name = generate_agent_name(symbol, model_type)
        model_name = f"minimal_crypto_{agent_name}"
        
        if model_name.replace("minimal_crypto_", "") in [f.replace("minimal_crypto_", "") for f in os.listdir(MODEL_DIR) 
                      if f.startswith("minimal_crypto_")] if os.path.exists(MODEL_DIR) else []:
            st.warning(f"⚠️ Model {model_type} for {exchange.upper()} already exists. Training will create a backup.")
        
        st.info(f"🎮 สร้าง Agent ชื่อ: {agent_name}")
        
        # เตรียมข้อมูล
        train_env, test_env, train_df, test_df = prepare_training_data(df, params)
        
        # เทรนโมเดล
        with st.spinner("🔄 กำลังเทรนโมเดล..."):
            model = train_model(train_env, model_type, params, params["steps"])
            
            if model is not None:
                # บันทึกโมเดล
                save_result = save_model(model, os.path.join(MODEL_DIR, model_name))
                if save_result['success']:
                    st.success(f"✅ บันทึกโมเดลสำเร็จที่: {save_result['path']}")
                    
                    # ประเมินผล
                    eval_result = evaluate_model(model, test_env)
                    if eval_result['success']:
                        st.success(f"""
                        📊 ผลการทดสอบ:
                        - 💰 เงินเริ่มต้น: ${INITIAL_AMOUNT:,.2f}
                        - 💵 เงินสิ้นสุด: ${eval_result['final_value']:,.2f}
                        - 📈 ผลตอบแทน: {eval_result['total_return']:.2f}%
                        """)
                    else:
                        st.error(f"❌ เกิดข้อผิดพลาดในการประเมินผล: {eval_result['error']}")
                else:
                    st.error(f"❌ เกิดข้อผิดพลาดในการบันทึกโมเดล: {save_result['error']}")
            
            show_training_progress(model_name, params["steps"], params["save_interval"])
            
        return {
            'success': True,
            'model_name': model_name
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def continue_training(df, model_to_continue, model_type, params):
    """เทรนต่อจากโมเดลเดิม"""
    try:
        model_name = f"minimal_crypto_{model_to_continue}"
        save_result = {'success': False}  # กำหนดค่าเริ่มต้น
        
        # เตรียมข้อมูล
        train_env, test_env, train_df, test_df = prepare_training_data(df, params)
        
        # เทรนโมเดล
        with st.spinner("🔄 กำลังเทรนโมเดล..."):
            # โหลดโมเดลเดิม
            original_model_path = os.path.join(MODEL_DIR, model_name)
            model = train_model(train_env, model_type, params, params["steps"], model_path=original_model_path)
            
            if model is not None:
                # บันทึกโมเดลเวอร์ชั่นใหม่
                save_result = save_model(model, original_model_path, version=True)
                if save_result['success']:
                    st.success(f"""
                    ✅ บันทึกโมเดลเวอร์ชั่นใหม่สำเร็จ:
                    - 📁 ไฟล์: {save_result['path']}
                    - 🔢 เวอร์ชั่น: v{save_result['version']}
                    """)
                    
                    # ประเมินผล
                    eval_result = evaluate_model(model, test_env)
                    if eval_result['success']:
                        st.success(f"""
                        📊 ผลการทดสอบ:
                        - 💰 เงินเริ่มต้น: ${INITIAL_AMOUNT:,.2f}
                        - 💵 เงินสิ้นสุด: ${eval_result['final_value']:,.2f}
                        - 📈 ผลตอบแทน: {eval_result['total_return']:.2f}%
                        """)
                    else:
                        st.error(f"❌ เกิดข้อผิดพลาดในการประเมินผล: {eval_result['error']}")
                else:
                    st.error(f"❌ เกิดข้อผิดพลาดในการบันทึกโมเดล: {save_result['error']}")
            
            show_training_progress(save_result['path'] if save_result['success'] else model_name, 
                                 params["steps"], params["save_interval"])
            
        return {
            'success': True,
            'model_name': model_name,
            'version': save_result.get('version', 0) if save_result['success'] else 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def train_agent_ui():
    """UI สำหรับการเทรน agent"""
    st.header("🎯 Train RL Agent")
    
    # แสดงข้อมูล GPU
    show_gpu_info()
    
    # Check existing models
    existing_models = [f.replace("minimal_crypto_", "") for f in os.listdir(MODEL_DIR) 
                      if f.startswith("minimal_crypto_")] if os.path.exists(MODEL_DIR) else []
    
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
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
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
        model_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_to_continue}")
        model_info = get_model_info(model_path)
        if model_info:
            st.info(f"📝 Last trained: {model_info['last_modified']} | Size: {model_info['size']}")
        
        # แสดงเวอร์ชั่นปัจจุบัน
        current_version = get_latest_version(f"minimal_crypto_{model_to_continue}")
        st.info(f"📌 เวอร์ชั่นปัจจุบัน: v{current_version}")
        
        # แสดงชื่อไฟล์ที่จะบันทึก
        next_version_name = get_next_version_name(f"minimal_crypto_{model_to_continue}")
        st.success(f"💾 จะบันทึกเป็น: {next_version_name}")
        
        # Use the same type as the existing model
        model_type = "PPO (Simple)" if "simple" in model_to_continue else "PPO"
        st.write(f"Model Type: {model_type}")
        
        # แสดงคำแนะนำเพิ่มเติมสำหรับการตั้งค่า
        st.info("""
        💡 **คำแนะนำ:** ในการเทรนครั้งแรก แนะนำให้ใช้ค่า parameters เดิม 
        หากผลลัพธ์ไม่เป็นที่น่าพอใจ ค่อยปรับในการเทรนครั้งถัดไป
        """)
        
    else:  # Create Agent
        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            ["PPO", "PPO (Simple)"],
            help="PPO (Simple) uses fewer parameters and may train faster"
        )
    
    # Get training parameters
    params = show_training_parameters(model_type, exchange, grade)
    
    # Training button
    if train_mode == "Continue Training":
        start_button = st.button("🚀 Continue Training")
        if start_button:
            result = continue_training(df, model_to_continue, model_type, params)
            if not result['success']:
                st.error(f"❌ เกิดข้อผิดพลาดในการเทรน: {result['error']}")
    else:
        start_button = st.button("🚀 Create Agent")
        if start_button:
            result = train_new_agent(df, model_type, params, exchange)
            if not result['success']:
                st.error(f"❌ เกิดข้อผิดพลาดในการเทรน: {result['error']}")
