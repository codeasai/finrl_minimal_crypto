import sys
import os
import pandas as pd
import numpy as np
import pickle
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from config import *

# Test environment creation - ปรับ path ให้ถูกต้อง
PROCESSED_DIR = 'notebooks/processed_data'
pickle_file = os.path.join(PROCESSED_DIR, 'processed_crypto_data.pkl')

print("Loading data...")
with open(pickle_file, 'rb') as f:
    df = pickle.load(f)

print('Original columns:', list(df.columns))
print(f'Original data shape: {df.shape}')

# แบ่งข้อมูล
total_len = len(df)
train_size = int(total_len * 0.7)
val_size = int(total_len * 0.15)
test_df = df.iloc[train_size + val_size:].copy().reset_index(drop=True)

# เตรียมข้อมูลตามมาตรฐาน FinRL
print("\n=== Preparing data for FinRL ===")

# แก้ไข column names ให้เป็น lowercase
price_column_mapping = {
    'Open': 'open',
    'High': 'high', 
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}

for old_col, new_col in price_column_mapping.items():
    if old_col in test_df.columns:
        test_df = test_df.rename(columns={old_col: new_col})
        print(f'Renamed {old_col} -> {new_col}')

# เตรียม timestamp และ date
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

# สำคัญ: FinRL ต้องการข้อมูลเรียงตาม date แล้วตาม tic
# และต้องมี column ชื่อ 'date' ที่เป็น datetime หรือ string
if 'date' in test_df.columns:
    # ใช้ date column ที่มีอยู่แล้ว
    test_df['date'] = pd.to_datetime(test_df['date'])
else:
    # สร้าง date column จาก timestamp
    test_df['date'] = test_df['timestamp'].dt.date
    test_df['date'] = pd.to_datetime(test_df['date'])

# เรียงลำดับข้อมูลอย่างถูกต้อง: date ก่อน แล้วตาม tic
test_df = test_df.sort_values(['date', 'tic']).reset_index(drop=True)

print(f'Test data shape after sorting: {test_df.shape}')
print(f'Test symbols: {test_df["tic"].unique()}')
print(f'Date range: {test_df["date"].min()} to {test_df["date"].max()}')

# ตรวจสอบข้อมูลให้ครบถ้วน
print("\n=== Data Quality Check ===")
required_cols = ['date', 'tic', 'close', 'high', 'low', 'open', 'volume']
missing_cols = [col for col in required_cols if col not in test_df.columns]
if missing_cols:
    print(f'❌ Missing required columns: {missing_cols}')
    exit(1)
else:
    print('✅ All required columns present')

# ลบ NaN values ในคอลัมน์ที่จำเป็น
original_shape = test_df.shape
test_df = test_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
print(f"After removing NaN: {original_shape} -> {test_df.shape}")

# ตรวจสอบว่าทุกวันมีข้อมูลครบทุก symbol หรือไม่
date_symbol_counts = test_df.groupby('date')['tic'].count()
expected_symbols = len(test_df['tic'].unique())
incomplete_dates = date_symbol_counts[date_symbol_counts != expected_symbols]

if len(incomplete_dates) > 0:
    print(f"⚠️ Found {len(incomplete_dates)} dates with incomplete data")
    print("Removing incomplete dates...")
    complete_dates = date_symbol_counts[date_symbol_counts == expected_symbols].index
    test_df = test_df[test_df['date'].isin(complete_dates)].reset_index(drop=True)
    print(f"After removing incomplete dates: {test_df.shape}")

# หา technical indicators ที่สะอาด
tech_cols = [col for col in test_df.columns if col.startswith(('macd', 'rsi', 'cci', 'adx'))]
clean_tech_cols = []
for col in tech_cols:
    if not test_df[col].isna().any():
        clean_tech_cols.append(col)
    else:
        print(f"Warning: {col} has {test_df[col].isna().sum()} NaN values")

print(f"Clean tech indicators: {clean_tech_cols}")

# คำนวณ parameters
stock_dim = len(test_df['tic'].unique())
state_space = 1 + 2 * stock_dim + stock_dim * len(clean_tech_cols)  
action_space = stock_dim
num_stock_shares = [0] * stock_dim

print(f'\nEnvironment parameters:')
print(f'Stock dim: {stock_dim}')
print(f'State space: {state_space}') 
print(f'Action space: {action_space}')
print(f'Tech indicators: {clean_tech_cols}')

# ตรวจสอบ data structure สุดท้าย
print(f'\n=== Final Data Structure ===')
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")

# ตรวจสอบว่า date และ tic เรียงลำดับถูกต้อง
print("First 10 rows (date, tic):")
print(test_df[['date', 'tic']].head(10))

print("Last 10 rows (date, tic):")
print(test_df[['date', 'tic']].tail(10))

# สร้าง subset ที่จำเป็นสำหรับ environment
essential_cols = ['date', 'tic', 'close', 'high', 'low', 'open', 'volume'] + clean_tech_cols
env_df = test_df[essential_cols].copy()

print(f'\nEnvironment dataframe shape: {env_df.shape}')
print(f'Environment columns: {list(env_df.columns)}')

# ลองสร้าง environment
try:
    print("\n=== Creating Environment ===")
    test_env = StockTradingEnv(
        df=env_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        num_stock_shares=num_stock_shares,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=clean_tech_cols,
        print_verbosity=0
    )
    print('✅ Environment created successfully!')
    
    # ทดสอบ reset
    print("Testing environment reset...")
    state = test_env.reset()
    print(f'✅ Initial state shape: {len(state)}')
    print(f'Initial state sample: {state[:min(10, len(state))]}')
    print('✅ Environment working properly!')
    
except Exception as e:
    print(f'❌ Error: {str(e)}')
    import traceback
    traceback.print_exc() 