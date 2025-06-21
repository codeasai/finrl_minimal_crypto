# test_sac_results.py - แสดงผลลัพธ์ SAC Agent แบบ Interactive
import pickle
import os
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
import pandas as pd

def get_available_agents():
    """หาไฟล์ SAC agents ทั้งหมดที่มีในระบบ"""
    sac_dir = "agents/sac"
    
    if not os.path.exists(sac_dir):
        print("❌ ไม่พบ directory agents/sac")
        return []
    
    agents_info = []
    
    # หาไฟล์ .zip ทั้งหมดที่เป็น SAC models
    all_files = os.listdir(sac_dir)
    zip_files = [f for f in all_files if f.endswith('.zip')]
    info_files = [f for f in all_files if f.endswith('_info.pkl')]
    
    print(f"🔍 พบไฟล์ .zip: {len(zip_files)} ไฟล์")
    print(f"🔍 พบไฟล์ _info.pkl: {len(info_files)} ไฟล์")
    
    # ประมวลผลไฟล์ที่มี _info.pkl
    processed_models = set()
    
    for info_file in info_files:
        try:
            model_name = info_file.replace('_info.pkl', '')
            zip_file = f"{model_name}.zip"
            
            info_path = os.path.join(sac_dir, info_file)
            zip_path = os.path.join(sac_dir, zip_file)
            
            # ตรวจสอบขนาดไฟล์ info.pkl
            info_size = os.path.getsize(info_path)
            if info_size == 0:
                print(f"⚠️  ไฟล์ {info_file} เสีย (ขนาด 0 bytes)")
                continue
            
            # ตรวจสอบว่ามีไฟล์ .zip หรือไม่
            if not os.path.exists(zip_path):
                print(f"⚠️  ไม่พบไฟล์ {zip_file} สำหรับ {info_file}")
                continue
            
            # อ่านข้อมูลจาก info.pkl
            with open(info_path, 'rb') as f:
                agent_info = pickle.load(f)
            
            file_size = os.path.getsize(zip_path) / (1024*1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(zip_path))
            
            agents_info.append({
                'file': info_file,
                'model_name': model_name,
                'info': agent_info,
                'size_mb': file_size,
                'modified': mod_time,
                'has_info': True
            })
            
            processed_models.add(model_name)
            print(f"✅ โหลดสำเร็จ: {model_name}")
            
        except Exception as e:
            print(f"❌ ไม่สามารถอ่านไฟล์ {info_file}: {e}")
            continue
    
    # ประมวลผลไฟล์ .zip ที่ไม่มี _info.pkl
    for zip_file in zip_files:
        model_name = zip_file.replace('.zip', '')
        
        if model_name in processed_models:
            continue  # ประมวลผลแล้ว
        
        try:
            zip_path = os.path.join(sac_dir, zip_file)
            file_size = os.path.getsize(zip_path) / (1024*1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(zip_path))
            
            # สร้างข้อมูลพื้นฐาน
            basic_info = {
                'model_name': model_name,
                'algorithm': 'SAC',
                'created_date': mod_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'crypto_symbols': ['Unknown']
            }
            
            agents_info.append({
                'file': f"{model_name}_info.pkl (ไม่มีไฟล์)",
                'model_name': model_name,
                'info': basic_info,
                'size_mb': file_size,
                'modified': mod_time,
                'has_info': False
            })
            
            print(f"⚠️  โหลดแบบไม่มี info: {model_name}")
            
        except Exception as e:
            print(f"❌ ไม่สามารถอ่านไฟล์ {zip_file}: {e}")
            continue
    
    # เรียงตามวันที่ modified (ล่าสุดก่อน)
    agents_info.sort(key=lambda x: x['modified'], reverse=True)
    
    return agents_info

def display_agents_list(agents_info):
    """แสดงรายการ agents ทั้งหมดให้เลือก"""
    print("📋 รายการ SAC Agents ทั้งหมด:")
    print("=" * 70)
    
    for i, agent in enumerate(agents_info, 1):
        info = agent['info']
        status_icon = "✅" if agent['has_info'] else "⚠️"
        
        print(f"{i:2d}. {status_icon} {agent['model_name'][:35]}{'...' if len(agent['model_name']) > 35 else ''}")
        print(f"    📅 วันที่: {agent['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    💾 ขนาด: {agent['size_mb']:.1f} MB")
        print(f"    🤖 Algorithm: {info.get('algorithm', 'N/A')}")
        print(f"    📈 Symbols: {info.get('crypto_symbols', 'N/A')}")
        if not agent['has_info']:
            print(f"    📄 Info: ไม่มีไฟล์ข้อมูลเพิ่มเติม")
        print()

def get_user_choice(max_choice):
    """รับตัวเลือกจากผู้ใช้"""
    while True:
        try:
            print(f"📝 เลือก agent ที่ต้องการดู (1-{max_choice}) หรือ 0 เพื่อออก:")
            choice = input("➤ ตัวเลือกของคุณ: ").strip()
            
            if choice == '0':
                return 0
            
            choice_num = int(choice)
            if 1 <= choice_num <= max_choice:
                return choice_num
            else:
                print(f"❌ กรุณาเลือกตัวเลข 1-{max_choice}")
                
        except ValueError:
            print("❌ กรุณาใส่ตัวเลขเท่านั้น")
        except KeyboardInterrupt:
            print("\n👋 ออกจากโปรแกรม")
            return 0

def display_agent_details(agent):
    """แสดงรายละเอียดของ agent ที่เลือก"""
    info = agent['info']
    model_name = agent['model_name']
    has_info = agent['has_info']
    
    print("\n" + "=" * 70)
    print(f"🔍 รายละเอียด SAC Agent: {model_name}")
    if not has_info:
        print("⚠️  หมายเหตุ: ไม่มีไฟล์ข้อมูลโดยละเอียด แสดงข้อมูลพื้นฐานเท่านั้น")
    print("=" * 70)
    
    print("\n📊 ข้อมูลพื้นฐาน:")
    print("-" * 40)
    print(f"🔤 ชื่อ Model: {info.get('model_name', model_name)}")
    print(f"🤖 Algorithm: {info.get('algorithm', 'N/A')}")
    print(f"📅 วันที่สร้าง: {info.get('created_date', 'N/A')}")
    print(f"📅 Modified: {agent['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    if has_info:
        print(f"💰 เงินเริ่มต้น: ${info.get('initial_amount', 0):,}")
        print(f"💸 ค่าธรรมเนียม: {info.get('transaction_cost_pct', 0)*100:.3f}%")
    print(f"📈 สัญลักษณ์: {info.get('crypto_symbols', 'N/A')}")
    
    if has_info and 'train_data_shape' in info and 'test_data_shape' in info:
        print(f"\n📊 ข้อมูลการฝึก:")
        print("-" * 40)
        print(f"🏋️  ข้อมูล Training: {info['train_data_shape'][0]:,} แถว")
        print(f"🧪 ข้อมูล Testing: {info['test_data_shape'][0]:,} แถว")
        
        if 'train_date_range' in info and 'test_date_range' in info:
            print(f"📅 ช่วงฝึก: {info['train_date_range']['start'][:10]} ถึง {info['train_date_range']['end'][:10]}")
            print(f"📅 ช่วงทดสอบ: {info['test_date_range']['start'][:10]} ถึง {info['test_date_range']['end'][:10]}")
    
    if has_info and 'indicators' in info:
        print(f"\n📈 Technical Indicators ({len(info['indicators'])} ตัว):")
        print("-" * 40)
        for i, indicator in enumerate(info['indicators'], 1):
            print(f"{i:2d}. {indicator}")
    
    print(f"\n💾 ข้อมูลไฟล์:")
    print("-" * 40)
    print(f"📦 Model: {model_name}.zip ({agent['size_mb']:.1f} MB)")
    
    if has_info:
        print(f"📄 Info: {agent['file']} ({os.path.getsize(os.path.join('agents/sac', agent['file']))} bytes)")
    else:
        print(f"📄 Info: ไม่มีไฟล์ข้อมูลเพิ่มเติม")
    
    # ข้อมูลเพิ่มเติม - แสดงเฉพาะตัวที่มี info
    if has_info and 'hyperparameters' in info:
        print(f"\n⚙️  Hyperparameters:")
        print("-" * 40)
        hyperparams = info['hyperparameters']
        important_params = ['learning_rate', 'buffer_size', 'batch_size', 'total_timesteps', 'ent_coef']
        
        for param in important_params:
            if param in hyperparams:
                value = hyperparams[param]
                if isinstance(value, float) and value < 1:
                    print(f"  {param}: {value:.6f}")
                elif isinstance(value, int):
                    print(f"  {param}: {value:,}")
                else:
                    print(f"  {param}: {value}")
    
    print(f"\n✅ SAC Agent พร้อมใช้งาน!")
    print("🚀 สามารถโหลดและใช้งานได้ด้วย:")
    print(f"   model = SAC.load('agents/sac/{model_name}.zip')")
    
    if not has_info:
        print("\n💡 หมายเหตุ:")
        print("   - ไฟล์นี้ไม่มีข้อมูลโดยละเอียด เนื่องจากไม่มีไฟล์ _info.pkl")
        print("   - สามารถใช้งาน model ได้ปกติ แต่จะไม่ทราบรายละเอียดการฝึก")

def main():
    """ฟังก์ชันหลักสำหรับรันโปรแกรม"""
    print("🔍 ตรวจสอบผลลัพธ์ SAC Agent แบบ Interactive")
    print("=" * 70)
    
    # หา agents ทั้งหมด
    agents_info = get_available_agents()
    
    if not agents_info:
        print("❌ ไม่พบ SAC agents ในระบบ")
        return
    
    print(f"🎯 พบ SAC Agents ทั้งหมด {len(agents_info)} ตัว")
    
    while True:
        print("\n")
        display_agents_list(agents_info)
        
        choice = get_user_choice(len(agents_info))
        
        if choice == 0:
            print("👋 ขอบคุณที่ใช้งาน!")
            break
        
        selected_agent = agents_info[choice - 1]
        display_agent_details(selected_agent)
        
        # ถามว่าต้องการดู agent อื่นไหม
        print(f"\n" + "─" * 70)
        while True:
            continue_choice = input("🔄 ต้องการดู agent อื่นไหม? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes', 'ใช่', '']:
                break
            elif continue_choice in ['n', 'no', 'ไม่', 'ไม่ใช่']:
                print("👋 ขอบคุณที่ใช้งาน!")
                return
            else:
                print("❌ กรุณาตอบ y หรือ n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 ออกจากโปรแกรม")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        print("🔧 กรุณาตรวจสอบไฟล์ในโฟลเดอร์ agents/sac") 