# improved_sac.py - Enhanced SAC Agent ด้วย Grade A Configuration
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

# Stable Baselines3 imports
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Import configurations
from config import *
from sac_configs import SAC_GradeConfigs, SAC_GradeSelector

# Import existing CryptoTradingEnv from sac.py
from sac import CryptoTradingEnv, setup_device, load_existing_data, add_technical_indicators, create_environment

def train_improved_sac_agent(train_env, grade='A'):
    """
    ฝึก SAC Agent ด้วย Grade Configuration ที่ปรับปรุงแล้ว
    
    Args:
        train_env: Training environment
        grade: Grade level (N, D, C, B, A, S) - default 'A' สำหรับระบบ 48GB+GPU
    """
    print(f"\n🤖 เริ่มฝึก Enhanced SAC Agent (Grade {grade})...")
    print("-" * 50)
    
    # ดึง configuration สำหรับ grade ที่เลือก
    from sac_configs import RL_GradeSelector
    config = RL_GradeSelector.get_config_by_algorithm_and_grade('SAC', grade)
    
    print(f"📊 การตั้งค่า Grade {grade}:")
    print(f"   - Training timesteps: {config['total_timesteps']:,}")
    print(f"   - Buffer size: {config['buffer_size']:,}")
    print(f"   - Learning starts: {config['learning_starts']:,}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Gradient steps: {config['gradient_steps']}")
    print(f"   - Entropy coefficient: {config['ent_coef']}")
    print(f"   - Use SDE: {config['use_sde']}")
    
    # Wrap environment
    vec_env = DummyVecEnv([lambda: train_env])
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    # สร้าง SAC model ด้วย Grade configuration
    model_params = {
        'policy': config['policy'],
        'env': vec_env,
        'learning_rate': config['learning_rate'],
        'buffer_size': config['buffer_size'],
        'learning_starts': config['learning_starts'],
        'batch_size': config['batch_size'],
        'tau': config['tau'],
        'gamma': config['gamma'],
        'train_freq': config['train_freq'],
        'gradient_steps': config['gradient_steps'],
        'target_update_interval': config['target_update_interval'],
        'verbose': config['verbose'],
        'seed': config.get('seed', 42),
        'device': device,
        'tensorboard_log': config.get('tensorboard_log', './logs/sac_improved/')
    }
    
    # เพิ่ม entropy tuning
    if config['ent_coef'] == 'auto':
        model_params['ent_coef'] = 'auto'
        model_params['target_entropy'] = 'auto'
    else:
        model_params['ent_coef'] = config['ent_coef']
    
    # เพิ่ม SDE ถ้ามี
    if config.get('use_sde', False):
        model_params['use_sde'] = True
        model_params['sde_sample_freq'] = config.get('sde_sample_freq', 64)
    
    # สร้าง model
    model = SAC(**model_params)
    
    print(f"✅ SAC Model สร้างสำเร็จ (Grade {grade})")
    print(f"🚀 เริ่มการฝึก {config['total_timesteps']:,} timesteps...")
    
    # Setup callbacks
    callbacks = []
    
    # Evaluation callback
    eval_freq = config.get('eval_freq', 10000)
    eval_callback = EvalCallback(
        vec_env, 
        best_model_save_path=f'./models/sac/best_model_grade_{grade}/',
        log_path=f'./logs/sac_eval_grade_{grade}/',
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # ฝึก model
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=callbacks,
        progress_bar=True
    )
    
    training_time = datetime.now() - start_time
    print(f"✅ ฝึก SAC Agent สำเร็จ (ใช้เวลา: {training_time})")
    
    return model, config, training_time

def save_improved_sac_agent(trained_model, config, training_time, train_df, test_df):
    """
    บันทึก Improved SAC Agent พร้อมข้อมูลเพิ่มเติม
    """
    print(f"\n💾 บันทึก Enhanced SAC Agent (Grade {config['grade']})...")
    print("-" * 50)
    
    # สร้างชื่อไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    model_name = f"enhanced_sac_grade_{config['grade']}_{timestamp}_{random_suffix}"
    
    # สร้าง directory ถ้ายังไม่มี
    os.makedirs("models/sac", exist_ok=True)
    
    # บันทึก trained model
    model_zip_path = os.path.join("models", "sac", f"{model_name}.zip")
    trained_model.save(model_zip_path)
    print(f"✅ บันทึก model: {model_zip_path}")
    
    # บันทึกข้อมูลเพิ่มเติม
    agent_info = {
        'model_name': model_name,
        'algorithm': 'SAC',
        'grade': config['grade'],
        'grade_description': config['description'],
        'created_date': datetime.now().isoformat(),
        'training_time': str(training_time),
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
        },
        'sac_config': config,
        'improvements': [
            f"Buffer size increased to {config['buffer_size']:,}",
            f"Learning starts reduced to {config['learning_starts']:,}",
            f"Gradient steps increased to {config['gradient_steps']}",
            f"Automatic entropy tuning: {config['ent_coef']}",
            f"SDE exploration: {config.get('use_sde', False)}",
            f"Total timesteps: {config['total_timesteps']:,}"
        ]
    }
    
    agent_info_path = os.path.join("models", "sac", f"{model_name}_info.pkl")
    with open(agent_info_path, 'wb') as f:
        pickle.dump(agent_info, f)
    print(f"✅ บันทึกข้อมูล agent: {agent_info_path}")
    
    # แสดงสรุป
    print(f"\n📋 สรุปการบันทึก Enhanced SAC Agent:")
    print(f"🔤 ชื่อ Model: {model_name}")
    print(f"🏆 Grade: {config['grade']} - {config['description']}")
    print(f"⏱️ ระยะเวลาการฝึก: {training_time}")
    print(f"📁 โฟลเดอร์: models/sac/")
    print(f"📦 ไฟล์ Model: {model_name}.zip")
    print(f"📄 ไฟล์ข้อมูล: {model_name}_info.pkl")
    
    return model_name, agent_info

def compare_with_original(improved_model, original_sac_path, test_env):
    """
    เปรียบเทียบประสิทธิภาพระหว่าง Improved SAC กับ Original SAC
    """
    print(f"\n📊 เปรียบเทียบประสิทธิภาพ...")
    print("-" * 50)
    
    try:
        # ทดสอบ Improved model
        print("🧪 ทดสอบ Enhanced SAC...")
        improved_results = test_agent(improved_model, test_env, "Enhanced SAC")
        
        # โหลดและทดสอบ Original model (ถ้ามี)
        if os.path.exists(original_sac_path):
            print("🧪 ทดสอบ Original SAC...")
            original_model = SAC.load(original_sac_path)
            original_results = test_agent(original_model, test_env, "Original SAC")
            
            # แสดงการเปรียบเทียบ
            print(f"\n📈 ผลการเปรียบเทียบ:")
            print(f"{'Metric':<20} {'Original SAC':<15} {'Enhanced SAC':<15} {'Improvement':<15}")
            print("-" * 65)
            
            original_return = (original_results['final_value'] - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
            improved_return = (improved_results['final_value'] - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
            improvement = improved_return - original_return
            
            print(f"{'Total Return (%)':<20} {original_return:<15.2f} {improved_return:<15.2f} {improvement:<15.2f}")
            print(f"{'Final Value ($)':<20} {original_results['final_value']:<15.2f} {improved_results['final_value']:<15.2f} {improved_results['final_value'] - original_results['final_value']:<15.2f}")
            
        else:
            print("⚠️ ไม่พบ Original SAC model สำหรับเปรียบเทียบ")
        
        return improved_results
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเปรียบเทียบ: {str(e)}")
        return None

def test_agent(model, test_env, model_name):
    """
    ทดสอบ agent และคืนค่าผลลัพธ์
    """
    obs, _ = test_env.reset()
    account_values = []
    actions_taken = []
    
    for step in range(len(test_env.df) - 21):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        
        account_values.append(info['total_value'])
        actions_taken.append(action[0])
        
        if done or truncated:
            break
    
    if account_values:
        final_value = account_values[-1]
        total_return = (final_value - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
        
        print(f"💰 {model_name} - เงินเริ่มต้น: ${INITIAL_AMOUNT:,.2f}")
        print(f"💰 {model_name} - เงินสุดท้าย: ${final_value:,.2f}")
        print(f"📈 {model_name} - ผลตอบแทนรวม: {total_return:.2f}%")
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'account_values': account_values,
            'actions_taken': actions_taken
        }
    
    return None

def main():
    """
    ฟังก์ชันหลักสำหรับการรัน Enhanced SAC Agent
    """
    print("🚀 Enhanced SAC (Soft Actor-Critic) Cryptocurrency Trading Agent")
    print("=" * 70)
    
    try:
        # ตรวจสอบระบบและแนะนำ Grade
        print("🔍 ตรวจสอบระบบและแนะนำ Grade...")
        
        # สำหรับระบบ 48GB RAM + GPU แนะนำ Grade A
        recommended_grade = 'A'
        
        ram_gb = 48  # สมมติใส่จากระบบ
        gpu_available = torch.cuda.is_available()
        
        print(f"💻 ระบบ: {ram_gb}GB RAM, GPU: {'มี' if gpu_available else 'ไม่มี'}")
        print(f"🏆 แนะนำ Grade: {recommended_grade} (Advanced)")
        
        # Setup device
        device = setup_device()
        
        # โหลดข้อมูล
        df = load_existing_data()
        
        # เพิ่ม technical indicators
        df = add_technical_indicators(df)
        
        # สร้าง environment
        train_env, test_env, train_df, test_df = create_environment(df)
        
        # ฝึก Enhanced SAC agent
        trained_model, config, training_time = train_improved_sac_agent(train_env, grade=recommended_grade)
        
        # บันทึก agent
        model_name, agent_info = save_improved_sac_agent(trained_model, config, training_time, train_df, test_df)
        
        # ทดสอบ agent
        results = test_agent(trained_model, test_env, f"Enhanced SAC Grade {recommended_grade}")
        
        # เปรียบเทียบกับ original (ถ้ามี)
        original_sac_path = "models/sac/sac_agent_original.zip"  # แก้ไขเป็นชื่อไฟล์จริง
        compare_with_original(trained_model, original_sac_path, test_env)
        
        print("\n🎉 สำเร็จ! Enhanced SAC Agent ถูกสร้างและบันทึกแล้ว")
        print("🔧 การปรับปรุงหลัก:")
        for improvement in agent_info['improvements']:
            print(f"   • {improvement}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
        print("🔍 กรุณาตรวจสอบข้อมูลและลองใหม่อีกครั้ง")
        sys.exit(1)

if __name__ == "__main__":
    main() 