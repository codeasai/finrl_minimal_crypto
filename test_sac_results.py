# test_sac_results.py - แสดงผลลัพธ์ SAC Agent
import pickle
import os
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
import pandas as pd

print("🔍 ตรวจสอบผลลัพธ์ SAC Agent")
print("=" * 50)

# หาไฟล์ SAC agent ล่าสุด
sac_dir = "models/sac"
sac_files = [f for f in os.listdir(sac_dir) if f.endswith('_info.pkl')]

if not sac_files:
    print("❌ ไม่พบไฟล์ SAC agent")
    exit()

# ใช้ไฟล์ล่าสุด
latest_file = sorted(sac_files)[-1]
model_name = latest_file.replace('_info.pkl', '')

print(f"📁 โหลดข้อมูลจาก: {latest_file}")

# อ่านข้อมูล agent
info_path = os.path.join(sac_dir, latest_file)
with open(info_path, 'rb') as f:
    agent_info = pickle.load(f)

print("\n📊 ข้อมูล SAC Agent:")
print("-" * 30)
print(f"🔤 ชื่อ Model: {agent_info['model_name']}")
print(f"🤖 Algorithm: {agent_info['algorithm']}")
print(f"📅 วันที่สร้าง: {agent_info['created_date']}")
print(f"💰 เงินเริ่มต้น: ${agent_info['initial_amount']:,}")
print(f"💸 ค่าธรรมเนียม: {agent_info['transaction_cost_pct']*100}%")
print(f"📈 สัญลักษณ์: {agent_info['crypto_symbols']}")

print(f"\n📊 ข้อมูลการฝึก:")
print("-" * 30)
print(f"🏋️ ข้อมูล Training: {agent_info['train_data_shape'][0]} แถว")
print(f"🧪 ข้อมูล Testing: {agent_info['test_data_shape'][0]} แถว")
print(f"📅 ช่วงฝึก: {agent_info['train_date_range']['start'][:10]} ถึง {agent_info['train_date_range']['end'][:10]}")
print(f"📅 ช่วงทดสอบ: {agent_info['test_date_range']['start'][:10]} ถึง {agent_info['test_date_range']['end'][:10]}")

print(f"\n📈 Technical Indicators ({len(agent_info['indicators'])} ตัว):")
print("-" * 30)
for i, indicator in enumerate(agent_info['indicators'], 1):
    print(f"{i:2d}. {indicator}")

# ข้อมูลไฟล์
model_path = os.path.join(sac_dir, f"{model_name}.zip")
model_size = os.path.getsize(model_path) / (1024*1024)
info_size = os.path.getsize(info_path)

print(f"\n💾 ไฟล์ที่บันทึก:")
print("-" * 30)
print(f"📦 Model: {model_name}.zip ({model_size:.1f} MB)")
print(f"📄 Info: {latest_file} ({info_size} bytes)")

print(f"\n✅ SAC Agent พร้อมใช้งาน!")
print("🚀 สามารถโหลดและใช้งานได้ด้วย:")
print(f"   model = SAC.load('models/sac/{model_name}.zip')")
print("=" * 50) 