# main.py
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yfinance as yf
import torch

# FinRL imports
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# Import config
from config import *

# Use standardized directories from config
# DATA_DIR, DATA_PREPARE_DIR, and MODEL_DIR are already defined in config.py

# ตรวจสอบและตั้งค่า GPU
def setup_device():
    """
    ตรวจสอบและตั้งค่าการใช้งาน GPU/CPU
    """
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
    
    # ตั้งค่า environment variable สำหรับ Stable Baselines3
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else "-1"
    
    return device

def download_crypto_data(force_download=False):
    """
    ดาวน์โหลดข้อมูล crypto โดยใช้ yfinance โดยตรง
    มีตัวเลือกให้บังคับดาวน์โหลดใหม่ได้ผ่าน force_download
    """
    data_file = os.path.join(DATA_DIR, "crypto_data.csv")
    
    # ถ้ามีไฟล์ข้อมูลอยู่แล้วและไม่ได้บังคับดาวน์โหลดใหม่
    if os.path.exists(data_file) and not force_download:
        print("📂 Loading existing data...")
        try:
            df = pd.read_csv(data_file)
            # แปลงชื่อคอลัมน์เป็นตัวเล็ก
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"✅ Loaded {len(df)} rows of data")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"📈 Symbols: {df['tic'].unique()}")
            return df
        except Exception as e:
            print(f"⚠️ Error loading existing data: {str(e)}")
            print("🔄 Proceeding with new download...")
    
    print("📊 Downloading crypto data...")
    print(f"📅 Date range: {START_DATE} to {END_DATE}")
    print(f"📈 Symbols: {CRYPTO_SYMBOLS}")
    
    # ดาวน์โหลดข้อมูลโดยตรงจาก yfinance
    df_list = []
    for symbol in CRYPTO_SYMBOLS:
        print(f"📥 Downloading {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=START_DATE,
                end=END_DATE,
                interval='1D',
                auto_adjust=True
            )
            if len(df) == 0:
                print(f"⚠️ Warning: No data found for {symbol}")
                continue
                
            # เพิ่มคอลัมน์ที่จำเป็น
            df['tic'] = symbol
            df['timestamp'] = df.index
            
            # แปลงชื่อคอลัมน์เป็นตัวเล็ก
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นครบหรือไม่
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'tic', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"⚠️ Warning: Missing columns for {symbol}: {', '.join(missing_columns)}")
                continue
            
            df_list.append(df)
            print(f"✅ Downloaded {len(df)} rows for {symbol}")
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            continue
    
    if not df_list:
        raise ValueError(f"ไม่พบข้อมูลในช่วงวันที่ {START_DATE} ถึง {END_DATE}")
    
    # รวมข้อมูล
    df = pd.concat(df_list, axis=0)
    df = df.reset_index(drop=True)
    
    # ตรวจสอบและแปลงชื่อคอลัมน์อีกครั้ง
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # บันทึกข้อมูล
    try:
        # สร้างโฟลเดอร์ถ้ายังไม่มี (directory already created in config.py)
        # บันทึกข้อมูล
        df.to_csv(data_file, index=False)
        print(f"💾 Saved data to {data_file}")
        print(f"📊 File size: {os.path.getsize(data_file) / 1024:.1f} KB")
        
        # ตรวจสอบว่าบันทึกสำเร็จ
        if os.path.exists(data_file):
            print("✅ Data saved successfully")
        else:
            print("❌ Error: File was not saved properly")
            
    except Exception as e:
        print(f"❌ Error saving data: {str(e)}")
        print("⚠️ Continuing without saving...")
    
    print(f"✅ Downloaded {len(df)} rows of data")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📈 Symbols: {df['tic'].unique()}")
    
    return df

def add_technical_indicators(df):
    """
    เพิ่ม technical indicators และ normalize ข้อมูล
    """
    print("📈 Adding technical indicators...")
    
    # สร้างสำเนาข้อมูล
    df = df.copy()
    
    # ตรวจสอบและแปลงชื่อคอลัมน์เป็นตัวเล็ก
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นครบหรือไม่
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'tic', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
    
    # 1. Moving Averages
    df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    
    # 2. RSI (Relative Strength Index)
    def calculate_rsi(group):
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # คำนวณ RSI แยกตาม tic
    rsi_values = []
    for tic in df['tic'].unique():
        tic_data = df[df['tic'] == tic].copy()
        rsi = calculate_rsi(tic_data)
        rsi_values.extend(rsi.values)
    df['rsi_14'] = rsi_values
    
    # 3. MACD
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
    
    macd_df = pd.concat(macd_dfs)
    df['macd'] = macd_df['macd'].values
    df['macd_signal'] = macd_df['macd_signal'].values
    df['macd_hist'] = macd_df['macd_hist'].values
    
    # 4. Bollinger Bands
    df['bb_middle'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['bb_std'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).std())
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # 5. Volume Indicators
    df['volume_sma_20'] = df.groupby('tic')['volume'].transform(lambda x: x.rolling(window=20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Normalize ข้อมูลราคาและ volume
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df.groupby('tic')[col].transform(lambda x: (x - x.mean()) / x.std())
    
    df['volume'] = df.groupby('tic')['volume'].transform(lambda x: (x - x.mean()) / x.std())
    
    # Normalize technical indicators
    indicator_cols = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                     'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'volume_sma_20', 'volume_ratio']
    
    for col in indicator_cols:
        if col in df.columns:
            df[col] = df.groupby('tic')[col].transform(lambda x: (x - x.mean()) / x.std())
    
    # แทนที่ค่า inf และ nan ด้วย 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print(f"✅ Added indicators: {indicator_cols}")
    print(f"✅ Normalized price, volume and indicators")
    print(f"Final columns: {len(df.columns)} columns")
    
    return df

def create_environment(df):
    """
    สร้าง trading environment แบบปลอดภัยจาก numpy AttributeError
    """
    print("🏛️ Creating trading environment...")
    
    # ตรวจสอบและแปลงชื่อคอลัมน์เป็นตัวเล็ก
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    # แบ่งข้อมูลเป็น train/test โดยใช้สัดส่วน 80/20
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    
    # ฟังก์ชันเตรียมข้อมูลสำหรับ FinRL แบบปลอดภัย
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
            'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
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
    
    # กำหนด indicators ที่ใช้
    indicators = [
        'sma_20', 'ema_20', 'rsi_14', 
        'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
        'volume_sma_20', 'volume_ratio'
    ]
    
    # สร้าง environment สำหรับ training
    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * len(CRYPTO_SYMBOLS),
        "buy_cost_pct": [TRANSACTION_COST_PCT] * len(CRYPTO_SYMBOLS),
        "sell_cost_pct": [TRANSACTION_COST_PCT] * len(CRYPTO_SYMBOLS),
        "state_space": 1 + 2 * len(CRYPTO_SYMBOLS) + len(CRYPTO_SYMBOLS) * len(indicators),
        "stock_dim": len(CRYPTO_SYMBOLS),
        "tech_indicator_list": indicators,
        "action_space": len(CRYPTO_SYMBOLS),
        "reward_scaling": 1e-3,  # ปรับ reward scaling
        "print_verbosity": 1     # เพิ่มการแสดงผลข้อมูล
    }
    
    train_env = StockTradingEnv(df=train_df, **env_kwargs)
    test_env = StockTradingEnv(df=test_df, **env_kwargs)
    
    print("✅ Environment created successfully")
    print(f"📊 Using indicators: {indicators}")
    
    return train_env, test_env, train_df, test_df

def train_agent(train_env):
    """
    เทรน DRL Agent ด้วยการปรับแต่ง hyperparameters
    """
    print("🤖 Training DRL Agent...")
    
    # ตรวจสอบและตั้งค่า GPU/CPU
    device = setup_device()
    
    # สร้าง agent
    agent = DRLAgent(env=train_env)
    
    # ปรับแต่ง hyperparameters ให้เหมาะสมกับข้อมูล crypto
    PPO_PARAMS = {
        'learning_rate': 1e-4,      # ลด learning rate ลง
        'n_steps': 1024,           # ลดจำนวน steps ต่อ batch
        'batch_size': 128,         # เพิ่ม batch size
        'n_epochs': 4,             # ลดจำนวน epochs
        'gamma': 0.99,             # discount factor
        'gae_lambda': 0.95,        # GAE parameter
        'clip_range': 0.2,         # PPO clip range
        'max_grad_norm': 0.5,      # gradient clipping
        'ent_coef': 0.01,          # entropy coefficient
        'vf_coef': 0.5,            # value function coefficient
        'target_kl': 0.02,         # target KL divergence
        'device': device           # ใช้ GPU หรือ CPU
    }
    
    # เลือก model (PPO ที่ปรับแต่งแล้ว)
    model_name = "ppo"
    print(f"🧠 Using {model_name.upper()} model")
    print(f"Model parameters: {PPO_PARAMS}")
    
    try:
        model = agent.get_model(model_name, model_kwargs=PPO_PARAMS)
        
        # เทรน model
        print("⏳ Training started... (this may take a few minutes)")
        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"minimal_crypto_{model_name}",
            total_timesteps=100000  # เพิ่มจำนวน timesteps
        )
        
        # บันทึก model (directory already created in config.py)
        model_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_name}")
        trained_model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        return trained_model
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        print("💡 Trying with simplified parameters...")
        
        # ลองใช้ parameters ที่เรียบง่ายขึ้น
        SIMPLE_PARAMS = {
            'learning_rate': 1e-4,
            'batch_size': 128,
            'n_steps': 1024,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'device': device  # ใช้ GPU หรือ CPU
        }
        
        print(f"New parameters: {SIMPLE_PARAMS}")
        model = agent.get_model(model_name, model_kwargs=SIMPLE_PARAMS)
        
        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"minimal_crypto_{model_name}_simple",
            total_timesteps=100000
        )
        
        # Directory already created in config.py
        model_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_name}_simple")
        trained_model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        return trained_model

def test_agent(trained_model, test_env):
    """
    ทดสอบ agent ที่เทรนแล้ว
    """
    print("📊 Testing trained agent...")
    
    # รัน backtest
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=test_env
    )
    
    print("✅ Backtesting completed")
    
    return df_account_value, df_actions

def analyze_results(df_account_value, test_df):
    """
    วิเคราะห์ผลลัพธ์
    """
    print("📈 Analyzing results...")
    
    # คำนวณ returns
    initial_value = INITIAL_AMOUNT
    final_value = df_account_value['account_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # คำนวณ buy and hold return (BTC)
    btc_data = test_df[test_df['tic'] == 'BTC-USD'].copy()
    btc_initial = btc_data['close'].iloc[0]
    btc_final = btc_data['close'].iloc[-1] 
    btc_return = (btc_final - btc_initial) / btc_initial * 100
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'='*50}")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Agent Total Return: {total_return:.2f}%")
    print(f"BTC Buy & Hold Return: {btc_return:.2f}%")
    print(f"Alpha (Agent - B&H): {total_return - btc_return:.2f}%")
    print(f"{'='*50}")
    
    # Plot results
    plot_results(df_account_value, test_df)
    
    return {
        'agent_return': total_return,
        'btc_return': btc_return,
        'alpha': total_return - btc_return,
        'final_value': final_value
    }

def plot_results(df_account_value, test_df):
    """
    สร้างกราฟแสดงผลลัพธ์
    """
    print("📊 Creating performance plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Portfolio value over time
    dates = pd.to_datetime(test_df['timestamp'].unique())
    portfolio_values = df_account_value['account_value'].values
    
    ax1.plot(dates, portfolio_values, label='Agent Portfolio', linewidth=2, color='blue')
    ax1.axhline(y=INITIAL_AMOUNT, color='red', linestyle='--', label='Initial Value')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: BTC Price comparison
    btc_data = test_df[test_df['tic'] == 'BTC-USD'].copy()
    btc_prices = btc_data.groupby('timestamp')['Close'].first().values
    btc_normalized = btc_prices / btc_prices[0] * INITIAL_AMOUNT
    
    portfolio_normalized = portfolio_values
    
    ax2.plot(dates, portfolio_normalized, label='Agent Portfolio', linewidth=2, color='blue')
    ax2.plot(dates, btc_normalized, label='BTC Buy & Hold', linewidth=2, color='orange')
    ax2.set_title('Agent vs Buy & Hold Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Normalized Value ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance plots saved and displayed")

def main():
    """
    Main function - รัน minimal crypto agent
    """
    print("🚀 Starting Minimal Crypto Agent with FinRL")
    print("="*60)
    
    try:
        # ตรวจสอบและตั้งค่า GPU/CPU
        device = setup_device()
        
        # Step 1: Download data (ใส่ force_download=True ถ้าต้องการดาวน์โหลดใหม่)
        df = download_crypto_data(force_download=False)
        
        # Step 2: Add technical indicators  
        df = add_technical_indicators(df)
        
        # Step 3: Create environments
        train_env, test_env, train_df, test_df = create_environment(df)
        
        # Step 4: Train agent
        trained_model = train_agent(train_env)
        
        # Step 5: Test agent
        df_account_value, df_actions = test_agent(trained_model, test_env)
        
        # Step 6: Analyze results
        results = analyze_results(df_account_value, test_df)
        
        print("\n🎉 Minimal Crypto Agent completed successfully!")
        print(f"🏆 Your agent achieved {results['agent_return']:.2f}% return")
        
        if results['alpha'] > 0:
            print(f"🎯 Great! Agent outperformed Buy & Hold by {results['alpha']:.2f}%")
        else:
            print(f"📈 Agent underperformed Buy & Hold by {abs(results['alpha']):.2f}%")
            print("💡 Try adjusting parameters or training longer!")
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Check your internet connection and try again")

if __name__ == "__main__":
    main()
    