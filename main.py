# main.py
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# FinRL imports
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# Import config
from config import *

# สร้างโฟลเดอร์สำหรับเก็บข้อมูล
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def download_crypto_data(force_download=False):
    """
    ดาวน์โหลดข้อมูล crypto โดยใช้ FinRL YahooFinanceProcessor
    มีตัวเลือกให้บังคับดาวน์โหลดใหม่ได้ผ่าน force_download
    """
    data_file = os.path.join(DATA_DIR, "crypto_data.csv")
    
    # ถ้ามีไฟล์ข้อมูลอยู่แล้วและไม่ได้บังคับดาวน์โหลดใหม่
    if os.path.exists(data_file) and not force_download:
        print("📂 Loading existing data...")
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Loaded {len(df)} rows of data")
        return df
    
    print("📊 Downloading crypto data...")
    
    # สร้าง data processor
    processor = YahooFinanceProcessor()
    
    # ดาวน์โหลดข้อมูล
    df = processor.download_data(
        ticker_list=CRYPTO_SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        time_interval='1D'
    )
    
    # บันทึกข้อมูล
    df.to_csv(data_file, index=False)
    print(f"💾 Saved data to {data_file}")
    
    print(f"✅ Downloaded {len(df)} rows of data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Symbols: {df['tic'].unique()}")
    
    return df

def add_technical_indicators(df):
    """
    เพิ่ม technical indicators และ normalize ข้อมูล
    """
    print("📈 Adding technical indicators...")
    
    processor = YahooFinanceProcessor()
    
    # เพิ่ม technical indicators
    df = processor.add_technical_indicator(df, INDICATORS)
    
    # ตรวจสอบและแปลง timestamp ให้ถูกต้อง
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Normalize ข้อมูลราคาและ volume
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].apply(lambda x: (x - x.mean()) / x.std())
    df['volume'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    
    # Normalize technical indicators
    for indicator in INDICATORS:
        if indicator in df.columns:
            df[indicator] = (df[indicator] - df[indicator].mean()) / df[indicator].std()
    
    # แทนที่ค่า inf และ nan ด้วย 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print(f"✅ Added indicators: {INDICATORS}")
    print(f"✅ Normalized price, volume and indicators")
    print(f"Final columns: {len(df.columns)} columns")
    
    return df

def create_environment(df):
    """
    สร้าง trading environment
    """
    print("🏛️ Creating trading environment...")
    
    # แบ่งข้อมูลเป็น train/test โดยใช้สัดส่วน 80/20
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    # ตรวจสอบให้แน่ใจว่า timestamp และ date เป็นรูปแบบที่ถูกต้อง
    for data in [train_df, test_df]:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['date'] = data['timestamp'].dt.date
        # เรียงข้อมูลตามวันที่
        data.sort_values(['date', 'tic'], inplace=True)
        data.reset_index(drop=True, inplace=True)
    
    print(f"📚 Training data: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"📝 Testing data: {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # สร้าง environment สำหรับ training
    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * len(CRYPTO_SYMBOLS),
        "buy_cost_pct": [TRANSACTION_COST_PCT] * len(CRYPTO_SYMBOLS),
        "sell_cost_pct": [TRANSACTION_COST_PCT] * len(CRYPTO_SYMBOLS),
        "state_space": 1 + 2 * len(CRYPTO_SYMBOLS) + len(CRYPTO_SYMBOLS) * len(INDICATORS),
        "stock_dim": len(CRYPTO_SYMBOLS),
        "tech_indicator_list": INDICATORS,
        "action_space": len(CRYPTO_SYMBOLS),
        "reward_scaling": 1e-3,  # ปรับ reward scaling
        "print_verbosity": 1     # เพิ่มการแสดงผลข้อมูล
    }
    
    train_env = StockTradingEnv(df=train_df, **env_kwargs)
    test_env = StockTradingEnv(df=test_df, **env_kwargs)
    
    print("✅ Environment created successfully")
    
    return train_env, test_env, train_df, test_df

def train_agent(train_env):
    """
    เทรน DRL Agent ด้วยการปรับแต่ง hyperparameters
    """
    print("🤖 Training DRL Agent...")
    
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
        'target_kl': 0.02          # target KL divergence
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
        
        # บันทึก model
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
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
            'gae_lambda': 0.95
        }
        
        print(f"New parameters: {SIMPLE_PARAMS}")
        model = agent.get_model(model_name, model_kwargs=SIMPLE_PARAMS)
        
        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"minimal_crypto_{model_name}_simple",
            total_timesteps=100000
        )
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
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
    btc_initial = test_df['close'].iloc[0]
    btc_final = test_df['close'].iloc[-1] 
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
    btc_prices = test_df.groupby('timestamp')['close'].first().values
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
    