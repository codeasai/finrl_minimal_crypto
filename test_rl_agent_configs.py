# test_rl_agent_grades.py - ตัวอย่างการใช้งาน RL Agent Grade-based Configurations
"""
ตัวอย่างการใช้งาน RL Agent Grade-based Configuration System
รองรับ multiple algorithms: SAC, PPO, DQN, DDPG, TD3, A2C

แสดงวิธีการ:
1. เลือก algorithm และ grade
2. สร้าง agent จาก configuration
3. เทรน agent ด้วย grade-based parameters
4. ประเมินผลและเปรียบเทียบ
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# เพิ่ม path สำหรับ import modules ในโปรเจค
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration system
from rl_agent_configs import RL_GradeSelector, SAC_GradeSelector
from config import *

def demonstrate_multi_algorithm_configs():
    """แสดงการใช้งาน configurations สำหรับ algorithms ต่างๆ"""
    print("🤖 Multi-Algorithm RL Agent Configuration Demo")
    print("=" * 60)
    
    # แสดง algorithms ที่รองรับ
    algorithms = RL_GradeSelector.get_available_algorithms()
    print(f"📋 Supported Algorithms: {algorithms}")
    print()
    
    # ทดสอบแต่ละ algorithm กับ grade ต่างๆ
    test_grades = ['N', 'C', 'A']
    
    for algorithm in algorithms:
        print(f"\n🔧 {algorithm} Configuration Examples:")
        print("-" * 40)
        
        for grade in test_grades:
            try:
                config = RL_GradeSelector.get_config_by_algorithm_and_grade(algorithm, grade)
                print(f"  Grade {grade}: {config['description']}")
                
                # แสดงพารามิเตอร์สำคัญ
                if 'total_timesteps' in config:
                    print(f"    Timesteps: {config['total_timesteps']:,}")
                if 'buffer_size' in config:
                    print(f"    Buffer: {config['buffer_size']:,}")
                if 'batch_size' in config:
                    print(f"    Batch: {config['batch_size']}")
                    
            except Exception as e:
                print(f"  Grade {grade}: Not available ({str(e)[:50]}...)")

def demonstrate_algorithm_selection():
    """แสดงการเลือก algorithm ตาม action space"""
    print("\n🎯 Algorithm Selection for Different Action Spaces")
    print("=" * 60)
    
    action_spaces = ['continuous', 'discrete', 'both']
    
    for space in action_spaces:
        recommended = RL_GradeSelector.get_recommended_algorithm_for_crypto(space)
        print(f"\n{space.upper()} Action Space:")
        print(f"  Recommended: {recommended}")
        
        # แสดงข้อมูลเพิ่มเติมของ algorithm แรก
        if recommended:
            first_algo = recommended[0]
            try:
                config = RL_GradeSelector.get_config_by_algorithm_and_grade(first_algo, 'C')
                print(f"  {first_algo} Grade C: {config['description']}")
            except:
                print(f"  {first_algo}: Configuration not available")

def demonstrate_performance_based_selection():
    """แสดงการเลือก configuration ตาม target performance"""
    print("\n📈 Performance-based Configuration Selection")
    print("=" * 60)
    
    test_scenarios = [
        {'algorithm': 'SAC', 'target_return': 5.0, 'time_hours': 4, 'scenario': 'Quick Test'},
        {'algorithm': 'PPO', 'target_return': 15.0, 'time_hours': 12, 'scenario': 'Medium Training'},
        {'algorithm': 'TD3', 'target_return': 30.0, 'time_hours': 48, 'scenario': 'Long Training'}
    ]
    
    for scenario in test_scenarios:
        config = RL_GradeSelector.get_config_by_performance(
            scenario['algorithm'], 
            scenario['target_return'], 
            scenario['time_hours']
        )
        
        print(f"\n{scenario['scenario']}:")
        print(f"  Algorithm: {scenario['algorithm']}")
        print(f"  Target Return: {scenario['target_return']}%")
        print(f"  Available Time: {scenario['time_hours']} hours")
        print(f"  → Recommended Grade: {config['grade']}")
        print(f"  → Description: {config['description']}")

def demonstrate_resource_based_selection():
    """แสดงการเลือก configuration ตามทรัพยากร"""
    print("\n💻 Resource-based Configuration Selection")
    print("=" * 60)
    
    resource_scenarios = [
        {'ram': 8, 'gpu': False, 'scenario': 'Basic Setup'},
        {'ram': 32, 'gpu': False, 'scenario': 'High RAM'},
        {'ram': 64, 'gpu': True, 'scenario': 'High-end Setup'}
    ]
    
    algorithms_to_test = ['SAC', 'PPO', 'DQN']
    
    for scenario in resource_scenarios:
        print(f"\n{scenario['scenario']} ({scenario['ram']}GB RAM, GPU: {scenario['gpu']}):")
        
        for algorithm in algorithms_to_test:
            try:
                config = RL_GradeSelector.get_config_by_resources(
                    algorithm, 
                    scenario['ram'], 
                    scenario['gpu']
                )
                print(f"  {algorithm}: Grade {config['grade']} - {config['description']}")
            except Exception as e:
                print(f"  {algorithm}: Error - {str(e)[:50]}...")

def create_sample_environment():
    """สร้าง environment ตัวอย่างสำหรับการทดสอบ"""
    print("\n🏗️ Creating Sample Environment for Testing")
    print("=" * 50)
    
    try:
        # Import finrl modules
        from finrl.env.env_stocktrading import StockTradingEnv
        
        # สร้างข้อมูลตัวอย่าง
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'tic': ['BTC-USD'] * 100,
            'close': 30000 + np.random.randn(100) * 1000,
            'volume': np.random.randint(1000, 10000, 100),
            'sma_20': 30000 + np.random.randn(100) * 500,
            'ema_20': 30000 + np.random.randn(100) * 500,
            'rsi_14': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 100,
            'macd_signal': np.random.randn(100) * 100,
            'macd_histogram': np.random.randn(100) * 50
        })
        
        # สร้าง environment
        env = StockTradingEnv(
            df=data,
            stock_dim=1,
            hmax=100,
            initial_amount=100000,
            transaction_cost_pct=0.001,
            reward_scaling=1e-4,
            tech_indicator_list=['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram']
        )
        
        print("✅ Environment created successfully")
        return env
        
    except ImportError as e:
        print(f"❌ Cannot create environment: {e}")
        print("💡 Install FinRL to test with real environment")
        return None
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None

def test_agent_creation():
    """ทดสอบการสร้าง agents ด้วย configurations ต่างๆ"""
    print("\n🤖 Testing Agent Creation")
    print("=" * 40)
    
    # สร้าง environment (ถ้าเป็นไปได้)
    env = create_sample_environment()
    
    if env is None:
        print("⚠️ Skipping agent creation test (no environment)")
        return
    
    # ทดสอบสร้าง agents
    test_configs = [
        {'algorithm': 'SAC', 'grade': 'N'},
        {'algorithm': 'PPO', 'grade': 'D'},
        {'algorithm': 'DQN', 'grade': 'N'}  # DQN อาจต้อง discrete action space
    ]
    
    for test_config in test_configs:
        try:
            print(f"\n🔧 Testing {test_config['algorithm']} Grade {test_config['grade']}:")
            
            # ใช้ function ใหม่ที่รองรับ multiple algorithms
            from rl_agent_configs import create_rl_agent_by_algorithm_and_grade
            
            agent, config = create_rl_agent_by_algorithm_and_grade(
                test_config['algorithm'],
                test_config['grade'],
                env
            )
            
            print(f"✅ {test_config['algorithm']} agent created successfully")
            print(f"   Model type: {type(agent).__name__}")
            
        except Exception as e:
            print(f"❌ Failed to create {test_config['algorithm']} agent: {str(e)[:100]}...")

def demonstrate_legacy_compatibility():
    """แสดงการรองรับ backward compatibility"""
    print("\n🔄 Legacy Compatibility Test")
    print("=" * 40)
    
    print("Testing legacy SAC_GradeSelector:")
    
    try:
        # ทดสอบ legacy methods
        legacy_config = SAC_GradeSelector.get_config_by_grade('B')
        print(f"✅ Legacy get_config_by_grade: {legacy_config['grade']} - {legacy_config['description']}")
        
        legacy_perf = SAC_GradeSelector.get_config_by_performance(15.0, 12)
        print(f"✅ Legacy get_config_by_performance: Grade {legacy_perf['grade']}")
        
        legacy_resource = SAC_GradeSelector.get_config_by_resources(32, True)
        print(f"✅ Legacy get_config_by_resources: Grade {legacy_resource['grade']}")
        
        print("✅ All legacy methods working correctly")
        
    except Exception as e:
        print(f"❌ Legacy compatibility error: {e}")

def generate_configuration_comparison_report():
    """สร้างรายงานเปรียบเทียบ configurations"""
    print("\n📊 Configuration Comparison Report")
    print("=" * 50)
    
    # สร้างตารางเปรียบเทียบ
    algorithms = ['SAC', 'PPO', 'DQN']
    grades = ['N', 'D', 'C']
    
    print(f"{'Algorithm':<10} {'Grade':<6} {'Timesteps':<12} {'Description':<40}")
    print("-" * 80)
    
    for algorithm in algorithms:
        for grade in grades:
            try:
                config = RL_GradeSelector.get_config_by_algorithm_and_grade(algorithm, grade)
                timesteps = config.get('total_timesteps', 'N/A')
                description = config.get('description', 'No description')[:35] + "..."
                
                print(f"{algorithm:<10} {grade:<6} {timesteps:<12} {description:<40}")
                
            except:
                print(f"{algorithm:<10} {grade:<6} {'N/A':<12} {'Configuration not available':<40}")

def main():
    """Main function สำหรับการทดสอบ"""
    print("🚀 RL Agent Grade-based Configuration System Test")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_multi_algorithm_configs()
    demonstrate_algorithm_selection()
    demonstrate_performance_based_selection()
    demonstrate_resource_based_selection()
    test_agent_creation()
    demonstrate_legacy_compatibility()
    generate_configuration_comparison_report()
    
    print("\n🎉 All tests completed!")
    print("💡 Use RL_GradeSelector for new code, SAC_GradeSelector for legacy compatibility")

if __name__ == "__main__":
    main() 