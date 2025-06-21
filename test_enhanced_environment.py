# test_enhanced_environment.py - Test Enhanced Environment vs Original
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Import environments
from sac import CryptoTradingEnv, load_existing_data, add_technical_indicators
from enhanced_crypto_env import EnhancedCryptoTradingEnv, create_enhanced_environment
from sac_configs import RL_GradeSelector
from config import *

def create_training_environments(df):
    """Create both original and enhanced environments"""
    
    # Split data
    split_index = int(len(df) * 0.8)
    train_df = df[:split_index].copy()
    test_df = df[split_index:].copy()
    
    print(f"📊 Data split: {len(train_df)} train, {len(test_df)} test")
    
    # Original environment
    original_train_env = CryptoTradingEnv(
        df=train_df,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        max_holdings=HMAX
    )
    
    original_test_env = CryptoTradingEnv(
        df=test_df,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        max_holdings=HMAX
    )
    
    # Enhanced environment
    enhanced_train_env = EnhancedCryptoTradingEnv(
        df=train_df,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        max_holdings=HMAX,
        lookback_window=20,
        enable_risk_management=True
    )
    
    enhanced_test_env = EnhancedCryptoTradingEnv(
        df=test_df,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        max_holdings=HMAX,
        lookback_window=20,
        enable_risk_management=True
    )
    
    return {
        'original': {'train': original_train_env, 'test': original_test_env},
        'enhanced': {'train': enhanced_train_env, 'test': enhanced_test_env}
    }

def train_sac_on_environment(env, env_name, grade='C', timesteps=200000):
    """Train SAC agent on specific environment"""
    
    print(f"\n🤖 Training SAC on {env_name} Environment (Grade {grade})...")
    print("-" * 60)
    
    # Get SAC configuration for specified grade
    config = RL_GradeSelector.get_config_by_algorithm_and_grade('SAC', grade)
    
    # Use specified timesteps
    config['total_timesteps'] = timesteps
    
    print(f"📊 Configuration:")
    print(f"   - Total timesteps: {config['total_timesteps']:,}")
    print(f"   - Buffer size: {config['buffer_size']:,}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Learning rate: {config['learning_rate']}")
    
    # Wrap environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create SAC model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        'verbose': 1,
        'seed': 42,
        'device': device
    }
    
    # Add entropy tuning if available
    if config.get('ent_coef') == 'auto':
        model_params['ent_coef'] = 'auto'
        model_params['target_entropy'] = 'auto'
    
    # Add SDE if available
    if config.get('use_sde', False):
        model_params['use_sde'] = True
        model_params['sde_sample_freq'] = config.get('sde_sample_freq', 64)
    
    # Create model
    model = SAC(**model_params)
    
    # Setup evaluation callback
    eval_freq = max(timesteps // 10, 5000)  # Evaluate 10 times during training
    
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=f'./models/sac/best_{env_name}_env/',
        log_path=f'./logs/sac_{env_name}_env/',
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Train model
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        progress_bar=True
    )
    
    training_time = datetime.now() - start_time
    
    # Save trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"sac_{env_name}_env_grade_{grade}_{timestamp}"
    model_path = f"models/sac/{model_name}.zip"
    
    os.makedirs("models/sac", exist_ok=True)
    model.save(model_path)
    
    print(f"✅ Training completed in {training_time}")
    print(f"💾 Model saved: {model_path}")
    
    return model, model_path, training_time

def test_trained_model(model, test_env, env_name):
    """Test trained model on environment"""
    
    print(f"\n🧪 Testing model on {env_name} environment...")
    print("-" * 50)
    
    try:
        obs, _ = test_env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < 200:  # Limit steps for safety
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            step_count += 1
            
            if done or truncated:
                break
        
        # Get trading statistics if available
        if hasattr(test_env, 'get_trading_statistics'):
            stats = test_env.get_trading_statistics()
        else:
            # Fallback for original environment
            final_value = info.get('total_value', 100000)
            total_return = (final_value - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
            stats = {
                'total_return': total_return,
                'final_value': final_value,
                'total_reward': total_reward,
                'steps': step_count
            }
        
        print(f"📊 Results for {env_name}:")
        print(f"   - Steps completed: {step_count}")
        print(f"   - Total reward: {total_reward:.4f}")
        
        if 'total_return' in stats:
            print(f"   - Total return: {stats['total_return']:.2f}%")
        if 'volatility' in stats:
            print(f"   - Volatility: {stats['volatility']:.2f}%")
        if 'sharpe_ratio' in stats:
            print(f"   - Sharpe ratio: {stats['sharpe_ratio']:.3f}")
        if 'max_drawdown' in stats:
            print(f"   - Max drawdown: {stats['max_drawdown']:.2f}%")
        if 'excess_return' in stats:
            print(f"   - Excess return: {stats['excess_return']:.2f}%")
        
        return stats
        
    except Exception as e:
        print(f"❌ Testing failed for {env_name}: {str(e)}")
        return None

def compare_environment_results(original_stats, enhanced_stats):
    """Compare results between environments"""
    
    print(f"\n📊 Environment Comparison Results")
    print("=" * 80)
    
    if not original_stats or not enhanced_stats:
        print("❌ Missing results for comparison")
        return
    
    metrics = ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'excess_return']
    
    print(f"{'Metric':<20} {'Original Env':<15} {'Enhanced Env':<15} {'Improvement':<15}")
    print("-" * 80)
    
    improvements = {}
    
    for metric in metrics:
        if metric in original_stats and metric in enhanced_stats:
            original_val = original_stats[metric]
            enhanced_val = enhanced_stats[metric]
            
            if metric == 'max_drawdown':
                # Lower is better for drawdown
                improvement = original_val - enhanced_val
                improvement_sign = "+" if improvement > 0 else ""
            else:
                # Higher is better for other metrics
                improvement = enhanced_val - original_val
                improvement_sign = "+" if improvement > 0 else ""
            
            improvements[metric] = improvement
            
            print(f"{metric:<20} {original_val:<15.3f} {enhanced_val:<15.3f} {improvement_sign}{improvement:<15.3f}")
    
    # Summary
    print(f"\n🏆 Summary:")
    better_count = sum(1 for imp in improvements.values() if imp > 0)
    total_count = len(improvements)
    
    print(f"   Enhanced environment performs better in {better_count}/{total_count} metrics")
    
    if 'excess_return' in improvements:
        if improvements['excess_return'] > 0:
            print(f"   🎯 Enhanced environment beats benchmark by additional {improvements['excess_return']:.2f}%")
        else:
            print(f"   ⚠️ Enhanced environment underperforms benchmark by {abs(improvements['excess_return']):.2f}%")
    
    return improvements

def plot_environment_comparison(original_stats, enhanced_stats):
    """Create comparison plots"""
    
    if not original_stats or not enhanced_stats:
        return
    
    # Metrics to compare
    metrics = []
    original_values = []
    enhanced_values = []
    
    metric_names = ['total_return', 'sharpe_ratio', 'excess_return']
    display_names = ['Total Return (%)', 'Sharpe Ratio', 'Excess Return (%)']
    
    for i, metric in enumerate(metric_names):
        if metric in original_stats and metric in enhanced_stats:
            metrics.append(display_names[i])
            original_values.append(original_stats[metric])
            enhanced_values.append(enhanced_stats[metric])
    
    if not metrics:
        print("No common metrics found for plotting")
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original_values, width, label='Original Environment', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced Environment', alpha=0.8, color='green')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Environment Comparison: Original vs Enhanced')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"models/sac/environment_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved: {plot_filename}")
    
    plt.show()

def main():
    """Main function to run environment comparison"""
    
    print("🚀 Enhanced vs Original Environment Comparison")
    print("=" * 80)
    
    try:
        # Load and prepare data
        print("📂 Loading data...")
        df = load_existing_data()
        df = add_technical_indicators(df)
        
        # Create environments
        print("🏗️ Creating environments...")
        environments = create_training_environments(df)
        
        # Compare environment specifications
        print(f"\n🔍 Environment Specifications:")
        original_env = environments['original']['train']
        enhanced_env = environments['enhanced']['train']
        
        print(f"Original Environment:")
        print(f"   - Observation space: {original_env.observation_space.shape}")
        print(f"   - Action space: {original_env.action_space.shape}")
        
        print(f"Enhanced Environment:")
        print(f"   - Observation space: {enhanced_env.observation_space.shape}")
        print(f"   - Action space: {enhanced_env.action_space.shape}")
        print(f"   - Features: Technical indicators + Portfolio info + Risk metrics + Momentum")
        print(f"   - Reward components: 7 (portfolio, sharpe, benchmark, transaction, drawdown, volatility, action quality)")
        
        # Train on both environments (shorter training for comparison)
        timesteps = 100000  # Reduced for faster comparison
        grade = 'C'         # Use Grade C for balance of speed and performance
        
        # Train on original environment
        original_model, original_path, original_time = train_sac_on_environment(
            environments['original']['train'], 'original', grade, timesteps
        )
        
        # Train on enhanced environment
        enhanced_model, enhanced_path, enhanced_time = train_sac_on_environment(
            environments['enhanced']['train'], 'enhanced', grade, timesteps
        )
        
        print(f"\n⏱️ Training Time Comparison:")
        print(f"   Original: {original_time}")
        print(f"   Enhanced: {enhanced_time}")
        
        # Test both models
        print(f"\n🧪 Testing trained models...")
        
        original_stats = test_trained_model(
            original_model, environments['original']['test'], 'Original'
        )
        
        enhanced_stats = test_trained_model(
            enhanced_model, environments['enhanced']['test'], 'Enhanced'
        )
        
        # Compare results
        if original_stats and enhanced_stats:
            improvements = compare_environment_results(original_stats, enhanced_stats)
            plot_environment_comparison(original_stats, enhanced_stats)
        
        print(f"\n🎉 Environment comparison completed!")
        print(f"📁 Models saved in models/sac/")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error in environment comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 