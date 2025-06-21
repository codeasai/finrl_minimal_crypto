# backtest_sac.py - Backtest SAC Agent Performance
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import warnings
warnings.filterwarnings('ignore')

# เพิ่ม path สำหรับ import
import sys
sys.path.append('.')

# Import custom environment จาก sac.py
from sac import CryptoTradingEnv, load_existing_data, add_technical_indicators

print("🧪 Backtest SAC Agent Performance")
print("=" * 50)

# หา SAC model ล่าสุด
sac_dir = "agents/sac"
sac_files = [f for f in os.listdir(sac_dir) if f.endswith('_info.pkl')]
latest_file = sorted(sac_files)[-1]
model_name = latest_file.replace('_info.pkl', '')

print(f"📁 โหลด SAC Model: {model_name}")

# โหลด model
model_path = os.path.join(sac_dir, f"{model_name}.zip")
model = SAC.load(model_path)

# โหลดข้อมูล
print("📊 โหลดข้อมูลสำหรับทดสอบ...")
df = load_existing_data()
df = add_technical_indicators(df)

# แบ่งข้อมูลเป็น train/test เหมือนเดิม
split_index = int(len(df) * 0.8)
test_df = df[split_index:].copy()

print(f"📈 ข้อมูลทดสอบ: {len(test_df)} วัน")
print(f"📅 ช่วงวันที่: {pd.to_datetime(test_df['timestamp']).min().date()} ถึง {pd.to_datetime(test_df['timestamp']).max().date()}")

# สร้าง environment สำหรับทดสอบ
env = CryptoTradingEnv(
    df=test_df,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    max_holdings=100
)

print("\n🚀 เริ่ม Backtesting...")

# รัน backtest
obs, _ = env.reset()
account_values = [100000]  # เริ่มต้นด้วยเงินเริ่มต้น
cash_values = [100000]
holding_values = [0]
actions_taken = []
prices = []

for step in range(len(test_df) - 21):  # -21 เพื่อให้เหลือพอสำหรับ indicators
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    account_values.append(info['total_value'])
    cash_values.append(info['cash'])
    holding_values.append(info['holdings'] * info['price'])
    actions_taken.append(action[0])
    prices.append(info['price'])
    
    if done or truncated:
        break

print("✅ Backtesting เสร็จสิ้น")

# คำนวณผลลัพธ์
initial_value = account_values[0]
final_value = account_values[-1]
total_return = (final_value - initial_value) / initial_value * 100

# Buy and Hold Strategy สำหรับเปรียบเทียบ
btc_initial_price = prices[0]
btc_final_price = prices[-1]
buy_hold_return = (btc_final_price - btc_initial_price) / btc_initial_price * 100

print(f"\n📊 ผลลัพธ์ Backtest ({len(account_values)-1} วัน):")
print("-" * 40)
print(f"💰 เงินเริ่มต้น: ${initial_value:,.2f}")
print(f"💰 เงินสุดท้าย: ${final_value:,.2f}")
print(f"📈 ผลตอบแทนรวม SAC: {total_return:.2f}%")
print(f"📈 ผลตอบแทน Buy & Hold: {buy_hold_return:.2f}%")
print(f"🎯 เอาชนะ Buy & Hold: {total_return - buy_hold_return:.2f}%")

# คำนวณ metrics เพิ่มเติม
returns = np.diff(account_values) / account_values[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

# Max Drawdown
running_max = np.maximum.accumulate(account_values)
drawdowns = (account_values - running_max) / running_max
max_drawdown = np.min(drawdowns) * 100

print(f"\n📈 Risk Metrics:")
print("-" * 40)
print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
print(f"🎯 Win Rate: {len([a for a in actions_taken if abs(a) > 0.1])}/{len(actions_taken)} actions")

# การกระจายของ actions
buy_actions = len([a for a in actions_taken if a > 0.1])
sell_actions = len([a for a in actions_taken if a < -0.1])
hold_actions = len([a for a in actions_taken if abs(a) <= 0.1])

print(f"\n🎮 Action Distribution:")
print("-" * 40)
print(f"🟢 Buy Actions: {buy_actions} ({buy_actions/len(actions_taken)*100:.1f}%)")
print(f"🔴 Sell Actions: {sell_actions} ({sell_actions/len(actions_taken)*100:.1f}%)")
print(f"⚪ Hold Actions: {hold_actions} ({hold_actions/len(actions_taken)*100:.1f}%)")

print(f"\n💾 สร้างกราฟประสิทธิภาพ...")

# สร้างกราฟ
plt.figure(figsize=(15, 10))

# กราฟ 1: Portfolio Value vs BTC Price
plt.subplot(2, 2, 1)
dates = pd.to_datetime(test_df['timestamp']).iloc[20:20+len(account_values)].values
buy_hold_values = [100000 * (p/prices[0]) for p in prices]
plt.plot(dates, account_values, label='SAC Portfolio', linewidth=2, color='blue')
plt.plot(dates[:len(buy_hold_values)], buy_hold_values, label='Buy & Hold', linewidth=2, color='orange')
plt.title('Portfolio Value Comparison')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# กราฟ 2: Cash vs Holdings
plt.subplot(2, 2, 2)
plt.plot(dates, cash_values, label='Cash', linewidth=2, color='green')
plt.plot(dates, holding_values, label='Holdings Value', linewidth=2, color='red')
plt.title('Cash vs Holdings')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# กราฟ 3: Actions Over Time
plt.subplot(2, 2, 3)
plt.plot(dates[:-1], actions_taken, linewidth=1, color='purple')
plt.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Buy threshold')
plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5, label='Sell threshold')
plt.title('Trading Actions')
plt.xlabel('Date')
plt.ylabel('Action Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# กราฟ 4: Drawdown
plt.subplot(2, 2, 4)
plt.plot(dates, drawdowns * 100, linewidth=2, color='red')
plt.fill_between(dates, drawdowns * 100, 0, alpha=0.3, color='red')
plt.title('Portfolio Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('agents/sac/sac_backtest_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ กราฟบันทึกแล้วที่: agents/sac/sac_backtest_results.png")
print("=" * 50) 