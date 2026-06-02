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
import time
import traceback
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

REQUIRED_INDICATORS = [
    'sma_20', 'ema_20', 'rsi_14',
    'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
    'volume_sma_20', 'volume_ratio'
]


def prepare_df_for_training(df):
    """
    แปลง DataFrame จาก Data Prepare ให้พร้อมใช้กับ create_environment.
    Returns (df_ready, symbols_list, error_or_None)
    """
    df = df.copy()

    # rename symbol -> tic
    if 'symbol' in df.columns and 'tic' not in df.columns:
        df = df.rename(columns={'symbol': 'tic'})

    if 'tic' not in df.columns:
        return None, [], "ไม่พบคอลัมน์ 'tic' หรือ 'symbol' ในไฟล์ข้อมูล"

    # derive bb_std ถ้าหาย
    if 'bb_std' not in df.columns and 'bb_upper' in df.columns and 'bb_middle' in df.columns:
        df['bb_std'] = (df['bb_upper'] - df['bb_middle']) / 2

    missing = [col for col in REQUIRED_INDICATORS if col not in df.columns]
    if missing:
        return None, [], (
            f"ขาด indicators ที่จำเป็น: {', '.join(missing)}\n"
            "กรุณา Re-run หน้า Data Prepare เพื่อเพิ่ม indicators ให้ครบ"
        )

    keep_cols = ['timestamp', 'tic', 'open', 'high', 'low', 'close', 'volume'] + REQUIRED_INDICATORS
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    symbols = sorted(df['tic'].unique().tolist())
    return df, symbols, None


def resample_and_index(df, indicators):
    """
    Resample intraday -> daily (1 row/symbol/day) แล้วตั้ง day-based index
    ที่ FinRL StockTradingEnv ต้องการ (df.loc[day] ต้องได้ DataFrame)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')

    rows_per_day = df.groupby(['date', 'tic']).size().max()
    if rows_per_day == 1:
        daily = df.sort_values(['date', 'tic']).reset_index(drop=True)
    else:
        agg = {'open': 'first', 'high': 'max', 'low': 'min',
               'close': 'last', 'volume': 'sum'}
        for ind in indicators:
            if ind in df.columns:
                agg[ind] = 'last'
        parts = []
        for tic in df['tic'].unique():
            sub = df[df['tic'] == tic].set_index('timestamp').sort_index()
            d = sub.resample('1D').agg(agg).dropna(subset=['close'])
            d['tic'] = tic
            d['date'] = d.index.strftime('%Y-%m-%d')
            parts.append(d.reset_index(drop=True))
        daily = pd.concat(parts).sort_values(['date', 'tic']).reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume'] + indicators:
        if col in daily.columns:
            daily[col] = pd.Series(daily[col].values).astype('float64').fillna(0.0)
    daily = daily.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # day-based index: df.loc[i] returns DataFrame with all symbols for day i
    dates = sorted(daily['date'].unique())
    idx = []
    for i, d in enumerate(dates):
        idx.extend([i] * (daily['date'] == d).sum())
    daily.index = idx
    return daily, dates


def build_env_from_df(df, reward_scaling=1e-3):
    """
    สร้าง StockTradingEnv จาก DataFrame ที่ผ่าน prepare_df_for_training แล้ว
    ใช้ symbols จากข้อมูลจริง รองรับทั้ง daily และ intraday
    Returns (train_env, test_env, train_df, test_df)
    """
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from config import HMAX, INITIAL_AMOUNT, TRANSACTION_COST_PCT

    indicators = REQUIRED_INDICATORS
    symbols = sorted(df['tic'].unique().tolist())
    n = len(symbols)

    daily_df, dates = resample_and_index(df, indicators)
    split_date = dates[int(len(dates) * 0.8)]

    train_df = daily_df[daily_df['date'] < split_date].copy()
    test_df  = daily_df[daily_df['date'] >= split_date].copy()

    # reset day-based index แยกสำหรับ train และ test
    def _reindex(data):
        ds = sorted(data['date'].unique())
        idx = []
        for i, d in enumerate(ds):
            idx.extend([i] * (data['date'] == d).sum())
        data = data.copy()
        data.index = idx
        return data

    train_df = _reindex(train_df)
    test_df  = _reindex(test_df)

    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * n,
        "buy_cost_pct":  [TRANSACTION_COST_PCT] * n,
        "sell_cost_pct": [TRANSACTION_COST_PCT] * n,
        "state_space": 1 + 2 * n + n * len(indicators),
        "stock_dim": n,
        "tech_indicator_list": indicators,
        "action_space": n,
        "reward_scaling": reward_scaling,
        "print_verbosity": 1,
    }

    train_env = StockTradingEnv(df=train_df, **env_kwargs)
    test_env  = StockTradingEnv(df=test_df,  **env_kwargs)
    return train_env, test_env, train_df, test_df


def run_training(train_env, params, model_type, model_name, existing_model_path=None):
    """
    เทรน (หรือเทรนต่อ) PPO agent และบันทึก .zip
    Returns (trained_model, save_path_without_extension)
    """
    import torch
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DRLAgent(env=train_env)
    is_simple = (model_type == "PPO (Simple)")

    ppo_kwargs = {
        'learning_rate': params['learning_rate'],
        'n_steps':       params['n_steps'],
        'batch_size':    params['batch_size'],
        'gamma':         params['gamma'],
        'gae_lambda':    params['gae_lambda'],
        'device':        device,
    }
    if not is_simple:
        ppo_kwargs.update({
            'n_epochs':   params['n_epochs'],
            'clip_range': params['clip_range'],
            'ent_coef':   params['ent_coef'],
            'vf_coef':    params['vf_coef'],
        })

    if existing_model_path and os.path.exists(existing_model_path):
        vec_env = DummyVecEnv([lambda: train_env])
        model = PPO.load(existing_model_path, env=vec_env, device=device)
        model.learn(
            total_timesteps=params['steps'],
            tb_log_name=f"continue_{model_name}",
            reset_num_timesteps=False,
        )
        trained_model = model
    else:
        model = agent.get_model("ppo", model_kwargs=ppo_kwargs)
        trained_model = agent.train_model(
            model=model,
            tb_log_name=model_name,
            total_timesteps=params['steps'],
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_name}")
    trained_model.save(save_path)
    return trained_model, save_path


def write_sidecar(model_name, agent_name, exchange, grade, model_type,
                  total_steps, symbols, data_file, continued_from=None):
    """บันทึก JSON metadata ควบคู่กับ .zip"""
    meta = {
        "agent_name":    agent_name,
        "exchange":      exchange,
        "grade":         grade,
        "model_type":    model_type,
        "training_date": datetime.now().isoformat(timespec='seconds'),
        "total_steps":   total_steps,
        "symbols":       symbols,
        "data_file":     data_file,
        "continued_from": continued_from,
    }
    sidecar_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_name}.json")
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def simple_train_agent_ui():
    """UI แบบง่ายสำหรับการเทรน agent"""
    st.header("🎯 Train RL Agent")

    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown("""
**ขั้นตอนที่ 3 จาก 6** — เทรน PPO agent ให้เรียนรู้กลยุทธ์การซื้อขาย

1. เลือก **Training Mode** (Create ใหม่ หรือ Continue จากเดิม)
2. เลือก **Exchange** และ **Grade** (N = เร็ว/ง่าย → S = นาน/ดีที่สุด)
3. เลือก **ไฟล์ข้อมูล** จาก `data/data_prepare/` ที่ผ่านขั้นที่ 2 แล้ว
4. กด **Create Agent** — รอจน status เปลี่ยนเป็น ✅

| Grade | Steps | เวลา (CPU) |
|-------|-------|-----------|
| N | 5,000 | ~2 นาที |
| B | 30,000 | ~15 นาที |
| S | 100,000 | ~50 นาที |

> ดูตารางเต็มและ troubleshooting ที่หน้า **📖 Help**
""")

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
    
    # ---- Training buttons ----
    if train_mode == "Continue Training":
        start_button = st.button("🚀 Continue Training")
        if start_button:
            # Pre-flight: โหลดและตรวจข้อมูล
            try:
                df_raw = pd.read_csv(os.path.join(DATA_DIR, selected_data_file))
            except Exception as e:
                st.error(f"❌ อ่านไฟล์ข้อมูลไม่ได้: {e}")
                return

            df_ready, symbols, err = prepare_df_for_training(df_raw)
            if err:
                st.error(f"❌ ข้อมูลไม่พร้อม: {err}")
                return

            existing_zip = os.path.join(MODEL_DIR, f"minimal_crypto_{model_to_continue}.zip")
            if not os.path.exists(existing_zip):
                st.error(f"❌ ไม่พบไฟล์ model: {existing_zip}")
                return

            t_start = time.time()
            try:
                with st.status("กำลังเทรนต่อ agent...", expanded=True) as status:
                    st.write(f"1/3 สร้าง environment ({len(symbols)} symbols, {len(df_ready):,} แถว)...")
                    train_env, _, _, _ = build_env_from_df(df_ready, params['reward_scaling'])

                    st.write(f"2/3 เทรนต่อ {params['steps']:,} steps (อาจใช้เวลาหลายนาที)...")
                    _, save_path = run_training(
                        train_env, params, model_type, model_to_continue,
                        existing_model_path=existing_zip
                    )

                    st.write("3/3 บันทึก metadata...")
                    write_sidecar(
                        model_name=model_to_continue,
                        agent_name=model_to_continue,
                        exchange=exchange,
                        grade=grade,
                        model_type=model_type,
                        total_steps=params['steps'],
                        symbols=symbols,
                        data_file=selected_data_file,
                        continued_from=model_to_continue,
                    )
                    elapsed = time.time() - t_start
                    status.update(label=f"เทรนต่อสำเร็จ! ({elapsed:.0f} วินาที)", state="complete")

                st.success(f"Agent '{model_to_continue}' เทรนต่อเสร็จแล้ว — {save_path}.zip")
                m1, m2, m3 = st.columns(3)
                m1.metric("Symbols", ", ".join(symbols))
                m2.metric("Steps เพิ่ม", f"{params['steps']:,}")
                m3.metric("เวลาที่ใช้", f"{elapsed:.0f}s")
                st.balloons()

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดระหว่างการเทรน: {e}")
                with st.expander("รายละเอียด error"):
                    st.code(traceback.format_exc())

    else:  # Create Agent
        start_button = st.button("🚀 Create Agent")
        if start_button:
            # Pre-flight: โหลดและตรวจข้อมูล
            try:
                df_raw = pd.read_csv(os.path.join(DATA_DIR, selected_data_file))
            except Exception as e:
                st.error(f"❌ อ่านไฟล์ข้อมูลไม่ได้: {e}")
                return

            df_ready, symbols, err = prepare_df_for_training(df_raw)
            if err:
                st.error(f"❌ ข้อมูลไม่พร้อม: {err}")
                return

            symbol_slug = "_".join(s.replace("/", "_") for s in symbols)
            type_slug   = model_type.replace(" ", "").replace("(", "").replace(")", "")
            agent_class = random.choice(LINEAGE2_CLASSES)
            model_name  = f"{symbol_slug}-{type_slug}-{agent_class}"

            t_start = time.time()
            try:
                with st.status(f"กำลังสร้าง agent '{model_name}'...", expanded=True) as status:
                    st.write(f"1/3 สร้าง environment ({len(symbols)} symbols, {len(df_ready):,} แถว)...")
                    try:
                        train_env, _, _, _ = build_env_from_df(df_ready, params['reward_scaling'])
                    except Exception as e:
                        status.update(label="สร้าง environment ล้มเหลว", state="error")
                        raise

                    st.write(f"2/3 เทรน {params['steps']:,} steps (อาจใช้เวลาหลายนาที)...")
                    try:
                        _, save_path = run_training(train_env, params, model_type, model_name)
                    except Exception as e:
                        status.update(label="การเทรนล้มเหลว", state="error")
                        raise

                    st.write("3/3 บันทึก metadata...")
                    write_sidecar(
                        model_name=model_name,
                        agent_name=model_name,
                        exchange=exchange,
                        grade=grade,
                        model_type=model_type,
                        total_steps=params['steps'],
                        symbols=symbols,
                        data_file=selected_data_file,
                        continued_from=None,
                    )
                    elapsed = time.time() - t_start
                    status.update(label=f"เทรนสำเร็จ! ({elapsed:.0f} วินาที)", state="complete")

                st.success(f"Agent '{model_name}' เทรนเสร็จแล้ว — {save_path}.zip")
                m1, m2, m3 = st.columns(3)
                m1.metric("Symbols", ", ".join(symbols))
                m2.metric("Total Steps", f"{params['steps']:,}")
                m3.metric("เวลาที่ใช้", f"{elapsed:.0f}s")
                st.info("ไปหน้า Test Agent หรือ Evaluate Performance เพื่อประเมินผล")
                st.balloons()

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดระหว่างการเทรน: {e}")
                with st.expander("รายละเอียด error"):
                    st.code(traceback.format_exc())

# เรียกใช้ UI
simple_train_agent_ui() 