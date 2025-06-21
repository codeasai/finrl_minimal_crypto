# improved_sac_strategy.py - Quick Win Improvements for SAC Agent
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, Tuple
from stable_baselines3 import SAC

from sac import CryptoTradingEnv
from config import *

class ImprovedRewardFunction:
    """Enhanced reward function to fix current issues"""
    
    def __init__(self):
        self.weights = {
            'excess_return': 0.40,        # Beat benchmark 
            'risk_adjusted_return': 0.25, # Sharpe-like metric
            'drawdown_penalty': 0.15,     # Penalize high drawdown
            'volatility_penalty': 0.10,   # Penalize high volatility
            'action_diversity': 0.05,     # Reward action variety
            'transaction_cost': 0.05      # Penalize excessive trading
        }
        
        self.portfolio_history = []
        self.action_history = []
        
    def calculate_enhanced_reward(self, old_value: float, new_value: float, 
                                action: float, transaction_cost: float,
                                benchmark_return: float) -> Tuple[float, Dict]:
        """Calculate improved reward with multiple components"""
        rewards = {}
        
        # 1. Portfolio return
        portfolio_return = (new_value - old_value) / old_value if old_value > 0 else 0
        
        # 2. Excess return vs benchmark
        excess_return = portfolio_return - benchmark_return
        rewards['excess_return'] = excess_return * self.weights['excess_return']
        
        # 3. Risk-adjusted return
        self.portfolio_history.append(portfolio_return)
        if len(self.portfolio_history) > 20:
            recent_returns = np.array(self.portfolio_history[-20:])
            if np.std(recent_returns) > 0:
                sharpe_approx = np.mean(recent_returns) / np.std(recent_returns)
                rewards['risk_adjusted_return'] = np.tanh(sharpe_approx) * 0.01 * self.weights['risk_adjusted_return']
            else:
                rewards['risk_adjusted_return'] = 0
        else:
            rewards['risk_adjusted_return'] = 0
        
        # 4. Drawdown penalty
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > 0.05:
            rewards['drawdown_penalty'] = -((current_drawdown - 0.05) ** 2) * 20 * self.weights['drawdown_penalty']
        else:
            rewards['drawdown_penalty'] = 0
        
        # 5. Volatility penalty
        if len(self.portfolio_history) > 10:
            volatility = np.std(self.portfolio_history[-10:]) * np.sqrt(252)
            if volatility > 0.30:
                rewards['volatility_penalty'] = -(volatility - 0.30) * 2 * self.weights['volatility_penalty']
            else:
                rewards['volatility_penalty'] = 0
        else:
            rewards['volatility_penalty'] = 0
        
        # 6. Action diversity reward
        self.action_history.append(self._classify_action(action))
        if len(self.action_history) >= 10:
            diversity = len(set(self.action_history[-10:])) / 3.0
            rewards['action_diversity'] = diversity * 0.005 * self.weights['action_diversity']
        else:
            rewards['action_diversity'] = 0
        
        # 7. Transaction cost penalty
        cost_ratio = transaction_cost / old_value if old_value > 0 else 0
        rewards['transaction_cost'] = -cost_ratio * 1000 * self.weights['transaction_cost']
        
        return sum(rewards.values()), rewards
    
    def _classify_action(self, action: float) -> str:
        if action > 0.3:
            return 'BUY'
        elif action < -0.3:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_drawdown(self) -> float:
        if len(self.portfolio_history) < 2:
            return 0.0
        cum_returns = np.cumprod([1 + r for r in self.portfolio_history])
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        return float(drawdown[-1])

class ImprovedCryptoTradingEnv(CryptoTradingEnv):
    """Enhanced environment with improved reward and risk management"""
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.reward_function = ImprovedRewardFunction()
        self.benchmark_holdings = self.initial_amount / df.iloc[20]['close']
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, True, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        next_price = self.df.iloc[self.current_step + 1]['close']
        old_total_value = self.total_value
        
        # Risk management: Force sell if drawdown too high
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.max_drawdown_limit:
            action = [-1.0]  # Force sell
        
        # Execute original step logic
        obs, _, done, truncated, info = super().step(action)
        
        # Calculate benchmark return
        benchmark_value = self.benchmark_holdings * next_price
        benchmark_return = (next_price - current_price) / current_price
        
        # Calculate enhanced reward
        reward, reward_breakdown = self.reward_function.calculate_enhanced_reward(
            old_total_value, self.total_value, action[0],
            info.get('transaction_cost', 0), benchmark_return
        )
        
        # Enhanced info
        info.update({
            'benchmark_return': benchmark_return,
            'current_drawdown': current_drawdown,
            'reward_breakdown': reward_breakdown
        })
        
        return obs, reward, done, truncated, info
    
    def _calculate_current_drawdown(self):
        if len(self.reward_function.portfolio_history) < 2:
            return 0.0
        return self.reward_function._calculate_drawdown()

def create_improved_sac_config():
    """Quick win SAC configuration improvements"""
    return {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-4,        # More stable
        'buffer_size': 500000,        # Larger buffer
        'learning_starts': 5000,      # Start learning earlier
        'batch_size': 512,            # Larger batch
        'tau': 0.01,                  # Faster target updates
        'gamma': 0.995,               # Long-term focus
        'train_freq': 4,              # Less frequent training
        'gradient_steps': 4,          # More gradient steps
        'ent_coef': 'auto',          # Auto entropy tuning
        'verbose': 1,
        'seed': 42
    }

def quick_test():
    """Quick test of improvements"""
    print("🧪 Testing SAC Improvements")
    print("=" * 40)
    
    # Create sample data
    data = {
        'close': np.random.randn(100).cumsum() + 100,
        'sma_20': np.random.randn(100).cumsum() + 100,
        'ema_20': np.random.randn(100).cumsum() + 100,
        'rsi_14': np.random.uniform(20, 80, 100),
        'macd': np.random.randn(100),
        'macd_signal': np.random.randn(100),
        'macd_hist': np.random.randn(100),
        'bb_middle': np.random.randn(100).cumsum() + 100,
        'bb_std': np.random.uniform(1, 5, 100),
        'bb_upper': np.random.randn(100).cumsum() + 105,
        'bb_lower': np.random.randn(100).cumsum() + 95,
        'volume_sma_20': np.random.uniform(1000, 10000, 100),
        'volume_ratio': np.random.uniform(0.5, 2.0, 100)
    }
    df = pd.DataFrame(data)
    
    # Test environments
    original_env = CryptoTradingEnv(df)
    improved_env = ImprovedCryptoTradingEnv(df)
    
    # Test rewards
    original_env.reset()
    improved_env.reset()
    
    rewards_original = []
    rewards_improved = []
    
    for i in range(10):
        action = [0.5]  # Buy action
        _, r1, _, _, _ = original_env.step(action)
        _, r2, _, _, info = improved_env.step(action)
        
        rewards_original.append(r1)
        rewards_improved.append(r2)
        
        print(f"Step {i+1}: Original={r1:.4f}, Improved={r2:.4f}")
        if 'reward_breakdown' in info:
            print(f"  Breakdown: {info['reward_breakdown']}")
    
    print(f"\nAverage rewards:")
    print(f"Original: {np.mean(rewards_original):.4f}")
    print(f"Improved: {np.mean(rewards_improved):.4f}")
    
    return improved_env

if __name__ == "__main__":
    quick_test()
    print("\n✅ Improvements ready to use!")
    print("Replace CryptoTradingEnv with ImprovedCryptoTradingEnv")
    print("Use create_improved_sac_config() for better SAC parameters") 