# integration_example.py - Enhanced SAC System Integration Example
"""
ตัวอย่างการใช้งาน Enhanced SAC System กับ finrl_minimal_crypto project

คุณสมบัติที่แสดง:
1. การสร้าง trading environment
2. การเลือก grade และ configuration
3. การเทรน agent พร้อม metadata tracking
4. การ evaluate และ compare performance
5. การ export results สำหรับ analysis
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append('../../')

# Import FinRL components
try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    FINRL_AVAILABLE = True
except ImportError:
    print("FinRL not available, using mock environment")
    FINRL_AVAILABLE = False

# Import our enhanced system
from enhanced_sac_trainer import create_enhanced_sac_trainer
from sac_metadata_manager import print_agent_summary

# Import config
try:
    from config import *
except ImportError:
    print("Main config not found, using default values")
    CRYPTO_SYMBOLS = ["BTC-USD"]
    INITIAL_AMOUNT = 100000

class MockTradingEnvironment:
    """Mock environment for testing when FinRL is not available"""
    
    def __init__(self):
        self.action_space = type('ActionSpace', (), {'shape': (3,)})()
        self.observation_space = type('ObservationSpace', (), {'shape': (20,)})()
    
    def reset(self):
        return np.random.random(20)
    
    def step(self, action):
        obs = np.random.random(20)
        reward = np.random.normal(0.001, 0.01)
        done = np.random.random() < 0.01  # 1% chance of episode end
        info = {}
        return obs, reward, done, info

def create_sample_environment():
    """Create sample trading environment for testing"""
    
    if FINRL_AVAILABLE:
        # Use real FinRL environment (simplified version)
        try:
            # Sample data for testing
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            sample_data = []
            
            for symbol in CRYPTO_SYMBOLS:
                for date in dates:
                    sample_data.append({
                        'date': date,
                        'tic': symbol,
                        'close': 50000 + np.random.normal(0, 1000),  # Sample BTC price
                        'volume': np.random.normal(1000, 100),
                        'high': 50000 + np.random.normal(500, 500),
                        'low': 50000 + np.random.normal(-500, 500),
                        'open': 50000 + np.random.normal(0, 800)
                    })
            
            df = pd.DataFrame(sample_data)
            
            # Create environment
            env = StockTradingEnv(
                df=df,
                stock_dim=len(CRYPTO_SYMBOLS),
                hmax=100,
                initial_amount=INITIAL_AMOUNT,
                buy_cost_pct=0.001,
                sell_cost_pct=0.001,
                reward_scaling=1e-4,
                state_space=len(CRYPTO_SYMBOLS) * 6,
                action_space=len(CRYPTO_SYMBOLS),
                tech_indicator_list=[]
            )
            
            print("✅ Created FinRL StockTradingEnv")
            return env
            
        except Exception as e:
            print(f"Failed to create FinRL environment: {e}")
            print("Using mock environment instead")
            return MockTradingEnvironment()
    else:
        print("Using mock trading environment")
        return MockTradingEnvironment()

def demo_enhanced_sac_training():
    """Demonstrate enhanced SAC training with different grades"""
    print("🚀 Enhanced SAC Training Demo")
    print("="*50)
    
    # Create environment
    env = create_sample_environment()
    
    # Create enhanced trainer
    trainer = create_enhanced_sac_trainer()
    
    # Demo 1: Train agents with different grades
    print("\n📈 Demo 1: Training agents with different grades")
    grades_to_test = ['N', 'C', 'B']  # Test subset for demo
    trained_agents = {}
    
    for grade in grades_to_test:
        print(f"\n🎯 Training Grade {grade} agent...")
        
        # Create agent
        model, metadata = trainer.create_agent(env, grade=grade)
        
        # Quick training (reduce timesteps for demo)
        quick_timesteps = {
            'N': 5000,
            'C': 10000,
            'B': 15000
        }
        
        # Simulate training (without actual RL training for demo)
        metadata.start_training()
        
        # Add mock training progress
        timesteps = quick_timesteps[grade]
        for i in range(0, timesteps, 1000):
            reward = np.random.normal(0.1 * (ord(grade) - ord('A') + 5), 0.05)
            metadata.add_training_step(i, reward)
        
        metadata.end_training()
        metadata.calculate_performance_summary()
        
        # Add evaluation result
        eval_reward = np.random.normal(0.15 * (ord(grade) - ord('A') + 5), 0.02)
        metadata.add_evaluation_result({
            'mean_reward': eval_reward,
            'std_reward': 0.02,
            'n_episodes': 10
        })
        
        trained_agents[grade] = (model, metadata)
        
        print(f"✅ Grade {grade} training completed")
        print(f"   Duration: {metadata.get_training_duration_formatted()}")
        print(f"   Mean Reward: {metadata.performance_metrics['mean_reward']:.4f}")
    
    # Demo 2: Performance comparison
    print(f"\n📊 Demo 2: Performance Comparison")
    comparison_df = trainer.compare_agents()
    print(comparison_df[['Agent ID', 'Grade', 'Mean Reward', 'Best Reward', 'Training Duration']])
    
    # Demo 3: Grade statistics
    print(f"\n📈 Demo 3: Grade Statistics")
    stats = trainer.metadata_manager.get_grade_statistics()
    for grade, data in stats.items():
        if data['count'] > 0:
            print(f"Grade {grade}: {data['count']} agents, "
                  f"avg reward: {data['avg_reward']:.4f}, "
                  f"avg duration: {data['avg_duration_hours']:.2f}h")
    
    # Demo 4: Best agents
    print(f"\n🏆 Demo 4: Best Agents")
    best_agents = trainer.get_best_agents(top_n=3)
    for i, agent in enumerate(best_agents, 1):
        print(f"#{i} Agent {agent.agent_id} (Grade {agent.grade}): "
              f"reward {agent.performance_metrics.get('mean_reward', 0):.4f}")
    
    # Demo 5: Export results
    print(f"\n📄 Demo 5: Export Results")
    export_file = trainer.metadata_manager.export_to_csv('enhanced_sac_demo_results.csv')
    print(f"Results exported to: {export_file}")
    
    return trained_agents

def demo_grade_selection_guide():
    """Demo grade selection guidelines"""
    print("\n🎯 Grade Selection Guide")
    print("="*50)
    
    scenarios = [
        {
            'scenario': 'Quick Prototype Testing',
            'recommended_grade': 'N',
            'reason': 'Fast training, minimal resources, good for concept validation'
        },
        {
            'scenario': 'Development & Debugging',
            'recommended_grade': 'D',
            'reason': 'Balanced speed/performance, moderate resource usage'
        },
        {
            'scenario': 'Standard Production Use',
            'recommended_grade': 'C',
            'reason': 'Good baseline performance, reasonable training time'
        },
        {
            'scenario': 'High-Performance Trading',
            'recommended_grade': 'B',
            'reason': 'Optimized for performance, suitable for live trading'
        },
        {
            'scenario': 'Maximum Performance',
            'recommended_grade': 'A',
            'reason': 'Near-optimal performance, ensemble-ready'
        },
        {
            'scenario': 'Research & Experimentation',
            'recommended_grade': 'S',
            'reason': 'Maximum resources, research-grade performance'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['scenario']}")
        print(f"   Recommended Grade: {scenario['recommended_grade']}")
        print(f"   Reason: {scenario['reason']}")

def demo_integration_with_streamlit():
    """Show how to integrate with Streamlit UI"""
    print("\n🌐 Streamlit Integration Example")
    print("="*50)
    
    streamlit_code = '''
# In your Streamlit app (ui/pages/3_Train_Agent.py)
import streamlit as st
from .enhanced_sac_trainer import create_enhanced_sac_trainer

# Grade selection
grade = st.selectbox(
    "Select Agent Grade",
    ['N', 'D', 'C', 'B', 'A', 'S'],
    index=2,  # Default to 'C'
    help="Choose grade based on your requirements"
)

# Show grade info
grade_info = {
    'N': "Novice: Fast training, basic performance",
    'D': "Developing: Balanced speed and performance", 
    'C': "Competent: Standard production ready",
    'B': "Proficient: High performance focus",
    'A': "Advanced: Near-optimal performance",
    'S': "Supreme: Maximum research-grade performance"
}
st.info(grade_info[grade])

# Training controls
col1, col2 = st.columns(2)
with col1:
    custom_timesteps = st.number_input("Training Timesteps", 
                                     min_value=1000, 
                                     max_value=1000000, 
                                     value=50000)
with col2:
    custom_buffer = st.number_input("Buffer Size", 
                                  min_value=10000, 
                                  max_value=2000000, 
                                  value=100000)

if st.button("Train Enhanced SAC Agent"):
    # Create trainer
    trainer = create_enhanced_sac_trainer()
    
    # Custom config if needed
    custom_config = {
        'total_timesteps': custom_timesteps,
        'buffer_size': custom_buffer
    }
    
    # Create and train agent
    with st.spinner(f"Training Grade {grade} agent..."):
        model, metadata = trainer.create_agent(env, grade=grade, custom_config=custom_config)
        results = trainer.train_agent(model, metadata)
        
        if results['success']:
            st.success(f"Training completed! Agent ID: {results['agent_id']}")
            
            # Show performance metrics
            st.subheader("Performance Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Mean Reward", 
                         f"{metadata.performance_metrics.get('mean_reward', 0):.4f}")
                st.metric("Training Duration", 
                         metadata.get_training_duration_formatted())
            
            with metrics_col2:
                st.metric("Best Reward", 
                         f"{metadata.performance_metrics.get('best_reward', 0):.4f}")
                st.metric("Total Timesteps", 
                         f"{metadata.total_timesteps_trained:,}")
            
            # Show comparison table
            st.subheader("Agent Comparison")
            comparison = trainer.compare_agents()
            st.dataframe(comparison)
            
        else:
            st.error(f"Training failed: {results.get('error', 'Unknown error')}")
'''
    
    print("📝 Add this code to your Streamlit UI:")
    print(streamlit_code)

def main():
    """Main demo function"""
    print("🎉 Enhanced SAC System Integration Demo")
    print("="*80)
    
    try:
        # Run demos
        trained_agents = demo_enhanced_sac_training()
        demo_grade_selection_guide()
        demo_integration_with_streamlit()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📂 Check agents/sac/metadata/ for saved agent metadata")
        print(f"📊 Check enhanced_sac_demo_results.csv for exported data")
        
        # Cleanup demo export file
        if os.path.exists('enhanced_sac_demo_results.csv'):
            print(f"🧹 Cleaning up demo export file...")
            os.remove('enhanced_sac_demo_results.csv')
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 