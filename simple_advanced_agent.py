# simple_advanced_agent.py
# Advanced Crypto Agent แบบง่าย ที่แก้ปัญหา AttributeError: 'numpy.float64' object has no attribute 'values'
# ใช้หลักการจาก main.py ที่ทำงานได้เรียบร้อย

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import yfinance as yf
import torch

# FinRL imports
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# Import config
from config import *

# สร้างโฟลเดอร์สำหรับเก็บข้อมูล
SIMPLE_DATA_DIR = "simple_data"
SIMPLE_MODEL_DIR = "models"  # ใช้ models directory เดียวกัน
for dir_name in [SIMPLE_DATA_DIR, SIMPLE_MODEL_DIR]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def setup_device():
    """ตรวจสอบและตั้งค่าการใช้งาน GPU/CPU"""
    print("\n🔍 ตรวจสอบการใช้งาน GPU/CPU")
    print("-" * 50)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ พบ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 จำนวน GPU: {torch.cuda.device_count()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("ℹ️ ไม่พบ GPU ใช้ CPU แทน")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else "-1"
    return device

def download_crypto_data_simple(symbols=None, lookback_days=365*2, force_download=False):
    """
    ดาวน์โหลดข้อมูล crypto แบบง่าย (ใช้หลักการจาก main.py)
    """
    if symbols is None:
        symbols = CRYPTO_SYMBOLS
    
    # คำนวณวันที่ย้อนหลัง  
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    data_file = os.path.join(SIMPLE_DATA_DIR, "simple_crypto_data.csv")
    
    if os.path.exists(data_file) and not force_download:
        print("📂 Loading existing simple data...")
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"✅ Loaded {len(df)} rows of data")
            print(f"📈 Symbols: {df['tic'].unique()}")
            return df
        except Exception as e:
            print(f"⚠️ Error loading existing data: {str(e)}")
    
    print(f"📊 Downloading simple crypto data for {len(symbols)} symbols...")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    df_list = []
    for symbol in symbols:
        print(f"📥 Downloading {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            
            # ดาวน์โหลดข้อมูลราคา
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D',
                auto_adjust=True
            )
            
            if len(df) == 0:
                print(f"⚠️ No data for {symbol}")
                continue
            
            df['tic'] = symbol
            df['timestamp'] = df.index
            
            # แปลงชื่อคอลัมน์
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df_list.append(df)
            print(f"✅ Downloaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            continue
    
    if not df_list:
        raise ValueError("ไม่พบข้อมูลใดๆ")
    
    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    # บันทึกข้อมูล
    df.to_csv(data_file, index=False)
    print(f"💾 Saved data to {data_file}")
    print(f"✅ Downloaded {len(df)} rows total")
    
    return df

def add_simple_technical_indicators(df):
    """
    เพิ่ม technical indicators แบบง่ายๆ (ใช้หลักการจาก main.py)
    """
    print("📈 Adding simple technical indicators...")
    
    df = df.copy()
    
    # แปลงชื่อคอลัมน์เป็นตัวเล็ก
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    # เรียงข้อมูลก่อนคำนวณ indicators
    df = df.sort_values(['tic', 'timestamp']).reset_index(drop=True)
    
    # 1. Moving Averages (เหมือน main.py)
    df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    
    # 2. RSI (เหมือน main.py)
    df['rsi_14'] = df.groupby('tic')['close'].transform(lambda x: 
        pd.Series(
            100 - (100 / (1 + 
                ((x.diff().where(x.diff() > 0, 0)).rolling(window=14).mean() / 
                 ((-x.diff().where(x.diff() < 0, 0)).rolling(window=14).mean() + 1e-8))
            ))
        ).fillna(50)
    )
    
    # 3. MACD (เหมือน main.py) 
    df['ema_12'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=12).mean())
    df['ema_26'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=26).mean())
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('tic')['macd'].transform(lambda x: x.ewm(span=9).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # แทนที่ NaN
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    df['macd_hist'] = df['macd_hist'].fillna(0)
    
    # ลบ temporary columns
    df = df.drop(['ema_12', 'ema_26'], axis=1)
    
    # 4. Bollinger Bands (เหมือน main.py)
    df['bb_middle'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(20).mean())
    df['bb_std'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # แทนที่ NaN
    df['bb_middle'] = df['bb_middle'].fillna(df['close'])
    df['bb_upper'] = df['bb_upper'].fillna(df['close'])
    df['bb_lower'] = df['bb_lower'].fillna(df['close'])
    
    # ลบ temporary column
    df = df.drop(['bb_std'], axis=1)
    
    # 5. Volume indicators (เหมือน main.py)
    df['volume_sma_20'] = df.groupby('tic')['volume'].transform(lambda x: x.rolling(window=20).mean())
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
    
    # แทนที่ค่า NaN และ inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(0)
    
    print(f"✅ Added simple technical indicators")
    print(f"📊 Final data shape: {df.shape}")
    
    return df

def create_simple_environment(df):
    """
    สร้าง trading environment แบบง่าย (ใช้หลักการจาก main.py ที่ทำงานได้)
    """
    print("🏛️ Creating simple trading environment...")
    
    # ตรวจสอบและแปลงชื่อคอลัมน์เป็นตัวเล็ก
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    # แบ่งข้อมูลเป็น train/test โดยใช้สัดส่วน 80/20 (เหมือน main.py)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    
    # ฟังก์ชันเตรียมข้อมูลสำหรับ FinRL แบบปลอดภัย (เหมือน main.py)
    def prepare_finrl_data(data):
        """แปลงข้อมูลให้เข้ากับ FinRL โดยหลีกเลี่ยง numpy scalar AttributeError"""
        data = data.copy()
        
        # แปลง timestamp และ date
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')  # ใช้ string แทน date object
        
        # เรียงข้อมูลตามวันที่และ symbol
        data = data.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # แปลงข้อมูลตัวเลขให้เป็น float64 และตรวจสอบ NaN
        numeric_columns = ['open', 'high', 'low', 'close', 'volume'] + [
            'sma_20', 'ema_20', 'rsi_14', 
            'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower',
            'volume_sma_20', 'volume_ratio'
        ]
        
        for col in numeric_columns:
            if col in data.columns:
                # แปลงเป็น pandas Series ชัดเจน และแทนที่ NaN ด้วย 0
                data[col] = pd.Series(data[col]).astype('float64').fillna(0.0)
        
        # ตรวจสอบและแก้ไข inf values
        data = data.replace([np.inf, -np.inf], 0.0)
        
        # ตรวจสอบให้แน่ใจว่าไม่มี NaN
        data = data.fillna(0.0)
        
        return data
    
    # เตรียมข้อมูล train และ test
    train_df = prepare_finrl_data(train_df)
    test_df = prepare_finrl_data(test_df)
    
    print(f"📚 Training data: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"📝 Testing data: {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # กำหนด indicators ที่ใช้ (เหมือน main.py)
    indicators = [
        'sma_20', 'ema_20', 'rsi_14', 
        'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_upper', 'bb_lower',
        'volume_sma_20', 'volume_ratio'
    ]
    
    # หาเฉพาะ symbols ที่มีข้อมูลครบ
    symbols = df['tic'].unique().tolist()
    print(f"📊 Using {len(symbols)} symbols: {symbols}")
    
    # สร้าง environment kwargs
    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * len(symbols),
        "buy_cost_pct": [TRANSACTION_COST_PCT] * len(symbols),
        "sell_cost_pct": [TRANSACTION_COST_PCT] * len(symbols),
        "state_space": 1 + 2 * len(symbols) + len(symbols) * len(indicators),
        "stock_dim": len(symbols),
        "tech_indicator_list": indicators,
        "action_space": len(symbols),
        "reward_scaling": 1e-3,  # ปรับ reward scaling
        "print_verbosity": 1     # เพิ่มการแสดงผลข้อมูล
    }
    
    # สร้าง environments
    try:
        train_env = StockTradingEnv(df=train_df, **env_kwargs)
        test_env = StockTradingEnv(df=test_df, **env_kwargs)
        
        print("✅ Simple environments created successfully")
        print(f"📊 Using indicators: {indicators}")
        
        return train_env, test_env, train_df, test_df, symbols
        
    except Exception as e:
        print(f"❌ Error creating environments: {str(e)}")
        raise e

def train_simple_agent(train_env):
    """
    เทรน DRL Agent แบบง่าย (ใช้หลักการจาก main.py)
    """
    print("🤖 Training simple DRL Agent...")
    
    device = setup_device()
    agent = DRLAgent(env=train_env)
    
    # ปรับแต่ง hyperparameters แบบง่าย
    PPO_PARAMS = {
        'learning_rate': 1e-4,
        'n_steps': 1024,
        'batch_size': 128,
        'n_epochs': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'max_grad_norm': 0.5,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'device': device
    }
    
    print(f"🧠 Using PPO model")
    print(f"Model parameters: {PPO_PARAMS}")
    
    try:
        model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        # เทรน model
        print("⏳ Training started...")
        trained_model = agent.train_model(
            model=model,
            tb_log_name="simple_advanced_crypto_ppo",
            total_timesteps=50000  # ลดจำนวน timesteps ลง
        )
        
        # บันทึก model
        model_path = os.path.join(SIMPLE_MODEL_DIR, "simple_advanced_crypto_ppo")
        trained_model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        return trained_model
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        raise e

def test_simple_agent(trained_model, test_env):
    """
    ทดสอบ agent ที่เทรนแล้ว (เหมือน main.py)
    """
    print("📊 Testing simple trained agent...")
    
    # รัน backtest
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=test_env
    )
    
    print("✅ Backtesting completed")
    
    return df_account_value, df_actions

def analyze_simple_results(df_account_value, test_df):
    """
    วิเคราะห์ผลลัพธ์แบบง่าย (เหมือน main.py)
    """
    print("📈 Analyzing simple results...")
    
    # คำนวณ returns
    initial_value = INITIAL_AMOUNT
    final_value = df_account_value['account_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # คำนวณ buy and hold return (BTC)
    btc_data = test_df[test_df['tic'] == 'BTC-USD'].copy()
    if len(btc_data) > 0:
        btc_initial = btc_data['close'].iloc[0]
        btc_final = btc_data['close'].iloc[-1] 
        btc_return = (btc_final - btc_initial) / btc_initial * 100
    else:
        btc_return = 0
    
    print(f"\n📊 SIMPLE ADVANCED RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"💰 Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"💰 Final Portfolio Value: ${final_value:,.2f}")
    print(f"📈 Agent Total Return: {total_return:.2f}%")
    print(f"📈 BTC Buy & Hold Return: {btc_return:.2f}%")
    print(f"🎯 Alpha (Agent - B&H): {total_return - btc_return:.2f}%")
    print(f"{'='*60}")
    
    return {
        'agent_return': total_return,
        'btc_return': btc_return,
        'alpha': total_return - btc_return,
        'final_value': final_value
    }

def main():
    """
    Main function สำหรับ Simple Advanced Crypto Agent
    """
    print("🚀 Starting Simple Advanced Crypto Agent (Fixed AttributeError)")
    print("="*70)
    
    try:
        # Step 1: Download data
        df = download_crypto_data_simple(force_download=False)
        
        # Step 2: Add technical indicators
        df = add_simple_technical_indicators(df)
        
        # Step 3: Create environment
        train_env, test_env, train_df, test_df, symbols = create_simple_environment(df)
        
        # Step 4: Train agent
        trained_model = train_simple_agent(train_env)
        
        # Step 5: Test agent
        df_account_value, df_actions = test_simple_agent(trained_model, test_env)
        
        # Step 6: Analyze results
        results = analyze_simple_results(df_account_value, test_df)
        
        print(f"\n🎉 Simple Advanced Crypto Agent completed successfully!")
        print(f"📈 Agent achieved {results['agent_return']:.2f}% return")
        
        if results['alpha'] > 0:
            print(f"🎯 Excellent! Agent outperformed BTC by {results['alpha']:.2f}%")
        else:
            print(f"⚠️ Agent underperformed BTC by {abs(results['alpha']):.2f}%")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        print(f"🔍 Full error trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 