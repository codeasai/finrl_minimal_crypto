import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# สร้างข้อมูลตัวอย่างที่ง่ายๆ
print("Creating simple test data...")

# สร้างข้อมูล 10 วัน, 2 symbols
dates = pd.date_range('2024-01-01', periods=10)
symbols = ['BTC', 'ETH']

data = []
for date in dates:
    for symbol in symbols:
        data.append({
            'date': date,
            'tic': symbol,
            'open': 100 + np.random.randn(),
            'high': 105 + np.random.randn(),
            'low': 95 + np.random.randn(),
            'close': 102 + np.random.randn(),
            'volume': 1000 + np.random.randn() * 100,
            'rsi': 50 + np.random.randn() * 10
        })

df = pd.DataFrame(data)
df = df.sort_values(['date', 'tic']).reset_index(drop=True)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# ลองสร้าง environment
try:
    print("\nCreating simple environment...")
    
    env = StockTradingEnv(
        df=df,
        stock_dim=2,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0, 0],
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=1 + 2*2 + 2*1,  # 1 + 2*stock_dim + stock_dim*tech_indicators
        action_space=2,
        tech_indicator_list=['rsi'],
        print_verbosity=0
    )
    print("✅ Simple environment created!")
    
    # ทดสอบ reset
    state = env.reset()
    print(f"✅ Environment reset successful, state shape: {len(state)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 