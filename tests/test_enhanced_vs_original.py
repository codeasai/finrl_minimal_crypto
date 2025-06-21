# test_enhanced_vs_original.py - เปรียบเทียบ Enhanced SAC vs Original SAC
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch

from stable_baselines3 import SAC
from sac import CryptoTradingEnv, load_existing_data, add_technical_indicators, create_environment
from config import *

def load_models():
    """โหลด models ทั้งหมดที่มีอยู่"""
    models = {}
    
    # Enhanced SAC Grade A
    enhanced_model_path = "agents/sac/enhanced_sac_grade_A_20250620_012303_XAJI0Y.zip"
    if os.path.exists(enhanced_model_path):
        models['Enhanced SAC Grade A'] = SAC.load(enhanced_model_path)
        print(f"✅ โหลด Enhanced SAC Grade A สำเร็จ")
    
    # Best Model from evaluation
    best_model_path = "agents/sac/best_model_grade_A/best_model.zip"
    if os.path.exists(best_model_path):
        models['Best Model Grade A'] = SAC.load(best_model_path)
        print(f"✅ โหลด Best Model Grade A สำเร็จ")
    
    # Original SAC (ถ้ามี)
    original_model_path = "agents/sac/sac_agent_20250619_151128_XXBF8G.zip"
    if os.path.exists(original_model_path):
        models['Original SAC'] = SAC.load(original_model_path)
        print(f"✅ โหลด Original SAC สำเร็จ")
    
    return models

def test_single_model(model, model_name, test_env):
    """ทดสอบ model เดี่ยวๆ"""
    print(f"\n🧪 ทดสอบ {model_name}...")
    print("-" * 50)
    
    try:
        obs, _ = test_env.reset()
        account_values = [INITIAL_AMOUNT]
        actions_taken = []
        daily_returns = []
        
        for step in range(len(test_env.df) - 21):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            
            account_values.append(info['total_value'])
            actions_taken.append(action[0])
            
            # คำนวณ daily return
            if len(account_values) > 1:
                daily_return = (account_values[-1] - account_values[-2]) / account_values[-2]
                daily_returns.append(daily_return)
            
            if done or truncated:
                break
        
        # คำนวณ metrics
        final_value = account_values[-1]
        total_return = (final_value - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
        
        # Risk metrics
        daily_returns = np.array(daily_returns)
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        max_drawdown = calculate_max_drawdown(account_values)
        
        results = {
            'model_name': model_name,
            'final_value': final_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'account_values': account_values,
            'actions_taken': actions_taken,
            'trading_days': len(account_values) - 1
        }
        
        print(f"💰 เงินเริ่มต้น: ${INITIAL_AMOUNT:,.2f}")
        print(f"💰 เงินสุดท้าย: ${final_value:,.2f}")
        print(f"📈 ผลตอบแทนรวม: {total_return:.2f}%")
        print(f"📊 Volatility (ปีละ): {volatility:.2f}%")
        print(f"⚡ Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"📅 จำนวนวันเทรด: {results['trading_days']} วัน")
        
        return results
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทดสอบ {model_name}: {str(e)}")
        return None

def calculate_max_drawdown(account_values):
    """คำนวณ Maximum Drawdown"""
    peak = account_values[0]
    max_dd = 0
    
    for value in account_values:
        if value > peak:
            peak = value
        
        drawdown = (peak - value) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

def compare_models(results_list):
    """เปรียบเทียบผลลัพธ์ระหว่าง models"""
    if len(results_list) < 2:
        print("⚠️ ต้องมีอย่างน้อย 2 models เพื่อเปรียบเทียบ")
        return
    
    print(f"\n📊 สรุปการเปรียบเทียบ ({len(results_list)} models)")
    print("=" * 100)
    
    # Table header
    print(f"{'Model Name':<25} {'Final Value':<15} {'Total Return':<15} {'Volatility':<12} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 100)
    
    # Sort by total return (descending)
    results_list.sort(key=lambda x: x['total_return'], reverse=True)
    
    for result in results_list:
        print(f"{result['model_name']:<25} ${result['final_value']:<14,.2f} {result['total_return']:<14.2f}% "
              f"{result['volatility']:<11.2f}% {result['sharpe_ratio']:<9.3f} {result['max_drawdown']:<9.2f}%")
    
    # แสดงการปรับปรুง
    print(f"\n🏆 อันดับที่ 1: {results_list[0]['model_name']}")
    if len(results_list) > 1:
        best = results_list[0]
        baseline = results_list[-1]  # อันดับสุดท้าย
        
        improvement_return = best['total_return'] - baseline['total_return']
        improvement_sharpe = best['sharpe_ratio'] - baseline['sharpe_ratio']
        improvement_dd = baseline['max_drawdown'] - best['max_drawdown']  # ลดลงเป็นเรื่องดี
        
        print(f"📈 ปรับปรุงผลตอบแทน: +{improvement_return:.2f}%")
        print(f"⚡ ปรับปรุง Sharpe Ratio: +{improvement_sharpe:.3f}")
        print(f"📉 ลด Max Drawdown: {improvement_dd:.2f}%")

def plot_comparison(results_list):
    """สร้างกราฟเปรียบเทียบ"""
    if not results_list:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Account Value Over Time
    plt.subplot(2, 2, 1)
    for result in results_list:
        if result and 'account_values' in result:
            plt.plot(result['account_values'], label=result['model_name'], linewidth=2)
    
    plt.title('Portfolio Value Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Returns Comparison (Bar Chart)
    plt.subplot(2, 2, 2)
    model_names = [r['model_name'] for r in results_list if r]
    returns = [r['total_return'] for r in results_list if r]
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    bars = plt.bar(range(len(model_names)), returns, color=colors, alpha=0.7)
    plt.title('Total Returns Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Total Return (%)')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # เพิ่มค่าบน bars
    for bar, return_val in zip(bars, returns):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(returns) * 0.01),
                f'{return_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Risk-Return Scatter
    plt.subplot(2, 2, 3)
    volatilities = [r['volatility'] for r in results_list if r]
    sharpe_ratios = [r['sharpe_ratio'] for r in results_list if r]
    
    scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='RdYlGn', s=100, alpha=0.8)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    for i, result in enumerate(results_list):
        if result:
            plt.annotate(result['model_name'], (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Risk-Return Analysis', fontsize=12, fontweight='bold')
    plt.xlabel('Volatility (% p.a.)')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Metrics Comparison (Radar Chart - simplified)
    plt.subplot(2, 2, 4)
    max_dds = [r['max_drawdown'] for r in results_list if r]
    
    bars = plt.bar(range(len(model_names)), max_dds, color='orange', alpha=0.7)
    plt.title('Maximum Drawdown Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Max Drawdown (%)')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # เพิ่มค่าบน bars
    for bar, dd_val in zip(bars, max_dds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(max_dds) * 0.01),
                f'{dd_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # บันทึกกราฟ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"agents/sac/enhanced_vs_original_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 บันทึกกราฟเปรียบเทียบ: {plot_filename}")
    
    plt.show()

def main():
    """ฟังก์ชันหลัก"""
    print("🔍 Enhanced SAC vs Original SAC Performance Comparison")
    print("=" * 80)
    
    try:
        # โหลด models
        models = load_models()
        
        if not models:
            print("❌ ไม่พบ models ใดๆ สำหรับทดสอบ")
            return
        
        print(f"\n📋 พบ {len(models)} models สำหรับทดสอบ")
        
        # เตรียมข้อมูลทดสอบ
        print("\n📂 เตรียมข้อมูลทดสอบ...")
        df = load_existing_data()
        df = add_technical_indicators(df)
        train_env, test_env, train_df, test_df = create_environment(df)
        
        print(f"📊 ข้อมูลทดสอบ: {len(test_df)} วัน")
        
        # ทดสอบแต่ละ model
        results_list = []
        for model_name, model in models.items():
            result = test_single_model(model, model_name, test_env)
            if result:
                results_list.append(result)
        
        # เปรียบเทียบผลลัพธ์
        if results_list:
            compare_models(results_list)
            plot_comparison(results_list)
        
        print(f"\n🎉 การเปรียบเทียบเสร็จสิ้น!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 