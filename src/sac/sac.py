# sac.py - SAC (Soft Actor-Critic) Agent for Cryptocurrency Trading
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
import pickle
import random
import string

# Stable Baselines3 imports โดยตรง
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import config
from config import *

class CryptoTradingEnv(gym.Env):
    """
    Custom Cryptocurrency Trading Environment สำหรับ SAC
    """
    
    def __init__(self, df, initial_amount=100000, transaction_cost_pct=0.001, max_holdings=100):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_holdings = max_holdings
        
        # Current state
        self.current_step = 0
        self.cash = initial_amount
        self.holdings = 0
        self.total_value = initial_amount
        
        # Action space: continuous between -1 and 1
        # -1 = sell all, 0 = hold, 1 = buy all
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: technical indicators + portfolio info
        self.n_features = len(INDICATORS) + 3  # indicators + cash, holdings, total_value
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        
        print(f"Environment initialized: {len(self.df)} timesteps, {self.n_features} features")
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.current_step = 20  # เริ่มจากวันที่ 20 เพื่อให้มี technical indicators
        self.cash = self.initial_amount
        self.holdings = 0
        self.total_value = self.initial_amount
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get current market data
        current_data = self.df.iloc[self.current_step]
        
        # Technical indicators
        indicators = [current_data[indicator] for indicator in INDICATORS]
        
        # Portfolio information (normalized)
        portfolio_info = [
            self.cash / self.initial_amount,  # normalized cash
            self.holdings / self.max_holdings,  # normalized holdings
            self.total_value / self.initial_amount  # normalized total value
        ]
        
        obs = np.array(indicators + portfolio_info, dtype=np.float32)
        
        # Replace NaN with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, True, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        next_step = self.current_step + 1
        next_price = self.df.iloc[next_step]['close']
        
        # Execute action
        action_value = action[0]
        
        if action_value > 0.1:  # Buy
            buy_amount = min(self.cash * action_value, self.cash)
            shares_to_buy = buy_amount / current_price
            transaction_cost = buy_amount * self.transaction_cost_pct
            
            if buy_amount > transaction_cost:
                self.holdings += shares_to_buy
                self.cash -= buy_amount + transaction_cost
                
        elif action_value < -0.1:  # Sell
            sell_ratio = abs(action_value)
            shares_to_sell = self.holdings * sell_ratio
            sell_amount = shares_to_sell * current_price
            transaction_cost = sell_amount * self.transaction_cost_pct
            
            if shares_to_sell > 0:
                self.holdings -= shares_to_sell
                self.cash += sell_amount - transaction_cost
        
        # Move to next step
        self.current_step = next_step
        
        # Calculate reward (portfolio return)
        old_total_value = self.total_value
        self.total_value = self.cash + self.holdings * next_price
        
        # Reward is the percentage change in portfolio value
        reward = (self.total_value - old_total_value) / old_total_value
        
        # Add penalty for holding cash (encourage trading)
        cash_penalty = -0.0001 * (self.cash / self.total_value)
        reward += cash_penalty
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'price': next_price
        }

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
    
    return device

def load_existing_data():
    """
    โหลดข้อมูลที่มีอยู่ในโฟลเดอร์ data
    """
    print("\n📂 โหลดข้อมูลที่มีอยู่...")
    print("-" * 50)
    
    # ตรวจสอบไฟล์ในโฟลเดอร์ data
    data_files = []
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            data_files.append(file)
    
    print(f"📁 พบไฟล์ข้อมูล: {data_files}")
    
    # ใช้ไฟล์ BTC-USD ก่อน
    btc_file = os.path.join(DATA_DIR, "BTC_USD-1d-20230601-20250609.csv")
    
    if os.path.exists(btc_file):
        print(f"📊 โหลดข้อมูล BTC-USD จาก {btc_file}")
        df = pd.read_csv(btc_file)
        
        # ตรวจสอบและแปลงคอลัมน์
        if 'Unnamed: 0' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Unnamed: 0'])
        elif 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        
        # แปลงชื่อคอลัมน์
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # เพิ่มคอลัมน์ tic
        df['tic'] = 'BTC-USD'
        
        print(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} แถว")
        print(f"📅 ช่วงวันที่: {df['timestamp'].min()} ถึง {df['timestamp'].max()}")
        print(f"📊 คอลัมน์: {list(df.columns)}")
        
        return df
    else:
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล BTC-USD ที่ {btc_file}")

def add_technical_indicators(df):
    """
    เพิ่ม technical indicators
    """
    print("\n📈 เพิ่ม Technical Indicators...")
    print("-" * 50)
    
    df = df.copy()
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'tic', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
    
    # 1. Moving Averages
    df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    
    # 2. RSI (Relative Strength Index)
    def calculate_rsi(series):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('tic')['close'].transform(calculate_rsi)
    
    # 3. MACD
    def calculate_macd(group):
        ema_12 = group['close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }, index=group.index)
    
    macd_data = df.groupby('tic').apply(calculate_macd).reset_index(level=0, drop=True)
    df = df.join(macd_data)
    
    # 4. Bollinger Bands
    def calculate_bollinger_bands(group):
        sma_20 = group['close'].rolling(window=20).mean()
        std_20 = group['close'].rolling(window=20).std()
        return pd.DataFrame({
            'bb_middle': sma_20,
            'bb_upper': sma_20 + (2 * std_20),
            'bb_lower': sma_20 - (2 * std_20),
            'bb_std': std_20
        }, index=group.index)
    
    bb_data = df.groupby('tic').apply(calculate_bollinger_bands).reset_index(level=0, drop=True)
    df = df.join(bb_data)
    
    # 5. Volume Indicators
    df['volume_sma_20'] = df.groupby('tic')['volume'].transform(lambda x: x.rolling(window=20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # ลบข้อมูลที่เป็น NaN ออก
    df = df.dropna()
    
    print(f"✅ เพิ่ม Technical Indicators สำเร็จ")
    print(f"📊 จำนวนข้อมูลหลังประมวลผล: {len(df)} แถว")
    print(f"📈 Indicators ที่เพิ่ม: {INDICATORS}")
    
    return df

def create_environment(df):
    """
    สร้าง Trading Environment สำหรับ SAC
    """
    print("\n🏗️ สร้าง Trading Environment...")
    print("-" * 50)
    
    # แบ่งข้อมูลเป็น train และ test (80:20)
    split_index = int(len(df) * 0.8)
    train_df = df[:split_index].copy()
    test_df = df[split_index:].copy()
    
    print(f"📊 ข้อมูล Training: {len(train_df)} แถว")
    print(f"📊 ข้อมูล Testing: {len(test_df)} แถว")
    
    # สร้าง Environment สำหรับ Training
    train_env = CryptoTradingEnv(
        df=train_df,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        max_holdings=HMAX
    )
    
    # สร้าง Environment สำหรับ Testing
    test_env = CryptoTradingEnv(
        df=test_df,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        max_holdings=HMAX
    )
    
    # ตรวจสอบ environment
    check_env(train_env)
    print("✅ สร้าง Environment สำเร็จ")
    
    return train_env, test_env, train_df, test_df

def train_sac_agent(train_env):
    """
    ฝึก SAC Agent
    """
    print("\n🤖 เริ่มฝึก SAC Agent...")
    print("-" * 50)
    
    # Wrap environment
    vec_env = DummyVecEnv([lambda: train_env])
    
    # สร้าง SAC model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=0.0003,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        verbose=1,
        seed=312,
        device=device
    )
    
    print(f"📊 SAC Model สร้างสำเร็จ (device: {device})")
    print(f"🚀 เริ่มการฝึก...")
    
    # ฝึก model
    model.learn(total_timesteps=50000, progress_bar=False)
    
    print("✅ ฝึก SAC Agent สำเร็จ")
    
    return model

def save_sac_agent(trained_model, train_df, test_df):
    """
    บันทึก SAC Agent และข้อมูลเพิ่มเติม
    """
    print("\n💾 บันทึก SAC Agent...")
    print("-" * 50)
    
    # สร้างชื่อไฟล์ที่ไม่ซ้ำ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    model_name = f"sac_agent_{timestamp}_{random_suffix}"
    
    # บันทึก trained model (.zip)
    model_zip_path = os.path.join("agents", "sac", f"{model_name}.zip")
    trained_model.save(model_zip_path)
    print(f"✅ บันทึก model: {model_zip_path}")
    
    # บันทึกข้อมูลเพิ่มเติม (.pkl)
    agent_info = {
        'model_name': model_name,
        'algorithm': 'SAC',
        'created_date': datetime.now().isoformat(),
        'crypto_symbols': CRYPTO_SYMBOLS,
        'indicators': INDICATORS,
        'initial_amount': INITIAL_AMOUNT,
        'transaction_cost_pct': TRANSACTION_COST_PCT,
        'hmax': HMAX,
        'train_data_shape': train_df.shape,
        'test_data_shape': test_df.shape,
        'train_date_range': {
            'start': str(train_df['timestamp'].min()),
            'end': str(train_df['timestamp'].max())
        },
        'test_date_range': {
            'start': str(test_df['timestamp'].min()),
            'end': str(test_df['timestamp'].max())
        }
    }
    
    agent_info_path = os.path.join("agents", "sac", f"{model_name}_info.pkl")
    with open(agent_info_path, 'wb') as f:
        pickle.dump(agent_info, f)
    print(f"✅ บันทึกข้อมูล agent: {agent_info_path}")
    
    # แสดงสรุป
    print(f"\n📋 สรุปการบันทึก SAC Agent:")
    print(f"🔤 ชื่อ Model: {model_name}")
    print(f"📁 โฟลเดอร์: agents/sac/")
    print(f"📦 ไฟล์ Model: {model_name}.zip")
    print(f"📄 ไฟล์ข้อมูล: {model_name}_info.pkl")
    
    return model_name, agent_info

def test_sac_agent(trained_model, test_env):
    """
    ทดสอบ SAC Agent
    """
    print("\n🧪 ทดสอบ SAC Agent...")
    print("-" * 50)
    
    try:
        obs, _ = test_env.reset()
        account_values = []
        actions_taken = []
        
        for step in range(len(test_env.df) - 21):  # -21 เพื่อให้เหลือพอสำหรับ indicators
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            
            account_values.append(info['total_value'])
            actions_taken.append(action[0])
            
            if done or truncated:
                break
        
        print("✅ ทดสอบ SAC Agent สำเร็จ")
        print(f"📊 ผลการทดสอบ: {len(account_values)} วัน")
        
        # แสดงผลการทดสอบสั้นๆ
        if account_values:
            initial_value = INITIAL_AMOUNT
            final_value = account_values[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            print(f"💰 เงินเริ่มต้น: ${initial_value:,.2f}")
            print(f"💰 เงินสุดท้าย: ${final_value:,.2f}")
            print(f"📈 ผลตอบแทนรวม: {total_return:.2f}%")
        
        return account_values, actions_taken
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทดสอบ: {str(e)}")
        return None, None

def main():
    """
    ฟังก์ชันหลักสำหรับการรัน SAC Agent
    """
    print("🚀 SAC (Soft Actor-Critic) Cryptocurrency Trading Agent")
    print("=" * 60)
    
    try:
        # 1. Setup device
        device = setup_device()
        
        # 2. โหลดข้อมูล
        df = load_existing_data()
        
        # 3. เพิ่ม technical indicators
        df = add_technical_indicators(df)
        
        # 4. สร้าง environment
        train_env, test_env, train_df, test_df = create_environment(df)
        
        # 5. ฝึก SAC agent
        trained_model = train_sac_agent(train_env)
        
        # 6. บันทึก agent
        model_name, agent_info = save_sac_agent(trained_model, train_df, test_df)
        
        # 7. ทดสอบ agent
        account_values, actions_taken = test_sac_agent(trained_model, test_env)
        
        print("\n🎉 สำเร็จ! SAC Agent ถูกสร้างและบันทึกแล้ว")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
        print("🔍 กรุณาตรวจสอบข้อมูลและลองใหม่อีกครั้ง")
        sys.exit(1)

if __name__ == "__main__":
    main() 