# enhanced_crypto_env.py - Enhanced Cryptocurrency Trading Environment
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple
import math

from config import *

class EnhancedCryptoTradingEnv(gym.Env):
    """
    Enhanced Cryptocurrency Trading Environment with:
    1. Multi-component reward function
    2. Risk-adjusted metrics
    3. Benchmark comparison
    4. Advanced position sizing
    5. Risk management features
    """
    
    def __init__(self, 
                 df, 
                 initial_amount=100000, 
                 transaction_cost_pct=0.001, 
                 max_holdings=100,
                 reward_weights: Dict = None,
                 lookback_window: int = 20,
                 enable_risk_management: bool = True):
        super(EnhancedCryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_holdings = max_holdings
        self.lookback_window = lookback_window
        self.enable_risk_management = enable_risk_management
        
        # Reward function weights
        self.reward_weights = reward_weights or {
            'portfolio_return': 1.0,      # Primary objective
            'sharpe_reward': 0.3,         # Risk-adjusted return
            'benchmark_reward': 0.2,      # Beat benchmark
            'transaction_penalty': 1.0,   # Cost awareness
            'drawdown_penalty': 0.5,      # Risk aversion
            'volatility_penalty': 0.2,    # Volatility control
            'concentration_penalty': 0.1   # Diversification (future use)
        }
        
        # Environment state
        self.reset_environment_state()
        
        # Action space: Enhanced continuous action
        # [position_change, confidence_level]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observation space: Enhanced features
        self.n_features = (
            len(INDICATORS) +           # Technical indicators
            3 +                         # Portfolio info (cash, holdings, total_value)
            5 +                         # Risk metrics (volatility, sharpe, drawdown, etc.)
            self.lookback_window        # Price momentum features
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        
        print(f"Enhanced Environment initialized:")
        print(f"  - {len(self.df)} timesteps")
        print(f"  - {self.n_features} features")
        print(f"  - Lookback window: {self.lookback_window}")
        print(f"  - Risk management: {self.enable_risk_management}")
        
    def reset_environment_state(self):
        """Reset all environment state variables"""
        self.current_step = self.lookback_window  # Start after lookback window
        self.cash = self.initial_amount
        self.holdings = 0
        self.total_value = self.initial_amount
        
        # Historical tracking for enhanced metrics
        self.portfolio_history = [self.initial_amount]
        self.action_history = []
        self.transaction_history = []
        self.daily_returns = []
        
        # Risk metrics
        self.peak_value = self.initial_amount
        self.current_drawdown = 0
        self.max_drawdown = 0
        
        # Benchmark tracking (Buy & Hold)
        self.benchmark_initial_price = None
        self.benchmark_value = self.initial_amount
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reset_environment_state()
        return self._get_enhanced_observation(), {}
    
    def _get_enhanced_observation(self):
        """Get enhanced observation with additional features"""
        if self.current_step >= len(self.df):
            # Return zeros if out of bounds
            return np.zeros(self.n_features, dtype=np.float32)
        
        current_data = self.df.iloc[self.current_step]
        
        # 1. Technical indicators
        indicators = [current_data[indicator] for indicator in INDICATORS]
        
        # 2. Portfolio information (normalized)
        portfolio_info = [
            self.cash / self.initial_amount,                    # Normalized cash
            self.holdings / self.max_holdings,                  # Normalized holdings
            self.total_value / self.initial_amount              # Normalized total value
        ]
        
        # 3. Risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # 4. Price momentum features (lookback window)
        momentum_features = self._get_momentum_features()
        
        # Combine all features
        obs = np.array(
            indicators + portfolio_info + risk_metrics + momentum_features, 
            dtype=np.float32
        )
        
        # Replace NaN with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _calculate_risk_metrics(self):
        """Calculate current risk metrics"""
        if len(self.portfolio_history) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]  # volatility, sharpe, drawdown, calmar, sortino
        
        # Calculate returns
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Sharpe ratio approximation
        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        sharpe = mean_return / (volatility + 1e-8) * np.sqrt(252)
        
        # Current drawdown
        self.current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        
        # Calmar ratio (return / max drawdown)
        annual_return = (self.total_value / self.initial_amount) ** (252 / max(len(self.portfolio_history), 1)) - 1
        calmar = annual_return / (self.max_drawdown + 1e-8)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0] if len(returns) > 0 else np.array([])
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino = mean_return / (downside_std + 1e-8) * np.sqrt(252)
        
        return [volatility, sharpe, self.current_drawdown, calmar, sortino]
    
    def _get_momentum_features(self):
        """Get price momentum features from lookback window"""
        if self.current_step < self.lookback_window:
            return [0.0] * self.lookback_window
        
        # Get price changes over lookback window
        prices = []
        for i in range(self.lookback_window):
            step = self.current_step - self.lookback_window + i + 1
            if step >= 0 and step < len(self.df):
                prices.append(self.df.iloc[step]['close'])
            else:
                prices.append(self.df.iloc[0]['close'])  # Fallback
        
        # Calculate momentum features (normalized price changes)
        if len(prices) > 1:
            current_price = prices[-1]
            momentum = [(price / current_price - 1) for price in prices]
        else:
            momentum = [0.0] * self.lookback_window
        
        return momentum
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_enhanced_observation(), 0, True, True, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        next_step = self.current_step + 1
        next_price = self.df.iloc[next_step]['close']
        
        # Enhanced action parsing
        position_change = action[0]    # -1 to 1 (sell to buy)
        confidence = action[1]         # 0 to 1 (confidence level)
        
        # Store action for analysis
        self.action_history.append({
            'step': self.current_step,
            'position_change': position_change,
            'confidence': confidence,
            'price': current_price
        })
        
        # Execute enhanced trade
        transaction_cost = self._execute_enhanced_trade(position_change, confidence, current_price)
        
        # Move to next step
        self.current_step = next_step
        
        # Update portfolio value
        old_total_value = self.total_value
        self.total_value = self.cash + self.holdings * next_price
        self.portfolio_history.append(self.total_value)
        
        # Update peak and drawdown
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        
        current_dd = (self.peak_value - self.total_value) / self.peak_value
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Update benchmark (Buy & Hold)
        if self.benchmark_initial_price is None:
            self.benchmark_initial_price = current_price
        self.benchmark_value = self.initial_amount * (next_price / self.benchmark_initial_price)
        
        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(
            old_total_value, transaction_cost, position_change, confidence
        )
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Enhanced info dictionary
        info = {
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'price': next_price,
            'benchmark_value': self.benchmark_value,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'transaction_cost': transaction_cost,
            'action_position': position_change,
            'action_confidence': confidence
        }
        
        return self._get_enhanced_observation(), reward, done, False, info
    
    def _execute_enhanced_trade(self, position_change, confidence, current_price):
        """Execute trade with enhanced position sizing and risk management"""
        
        # Base position sizing with confidence adjustment
        max_trade_value = self.total_value * 0.5  # Maximum 50% of portfolio per trade
        confidence_adjusted_size = max_trade_value * confidence
        position_size = position_change * confidence_adjusted_size
        
        # Risk management: Volatility adjustment
        if self.enable_risk_management and len(self.portfolio_history) > 10:
            volatility = self._calculate_risk_metrics()[0]  # Get current volatility
            volatility_adjustment = 1 / (1 + volatility * 2)  # Reduce size when volatile
            position_size *= volatility_adjustment
        
        transaction_cost = 0
        
        if position_size > 0:  # Buy
            buy_amount = min(position_size, self.cash)
            if buy_amount > 0:
                shares_to_buy = buy_amount / current_price
                transaction_cost = buy_amount * self.transaction_cost_pct
                
                if buy_amount > transaction_cost:
                    self.holdings += shares_to_buy
                    self.cash -= buy_amount + transaction_cost
                    
                    self.transaction_history.append({
                        'type': 'buy',
                        'amount': buy_amount,
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': transaction_cost,
                        'step': self.current_step
                    })
                    
        elif position_size < 0:  # Sell
            sell_ratio = min(abs(position_size) / (self.holdings * current_price + 1e-8), 1.0)
            shares_to_sell = self.holdings * sell_ratio
            
            if shares_to_sell > 0:
                sell_amount = shares_to_sell * current_price
                transaction_cost = sell_amount * self.transaction_cost_pct
                
                self.holdings -= shares_to_sell
                self.cash += sell_amount - transaction_cost
                
                self.transaction_history.append({
                    'type': 'sell',
                    'amount': sell_amount,
                    'shares': shares_to_sell,
                    'price': current_price,
                    'cost': transaction_cost,
                    'step': self.current_step
                })
        
        return transaction_cost
    
    def _calculate_enhanced_reward(self, old_total_value, transaction_cost, position_change, confidence):
        """Calculate enhanced reward with multiple components"""
        
        # 1. Portfolio return (primary)
        portfolio_return = (self.total_value - old_total_value) / old_total_value
        portfolio_reward = portfolio_return * self.reward_weights['portfolio_return']
        
        # 2. Risk-adjusted return (Sharpe-like)
        risk_metrics = self._calculate_risk_metrics()
        sharpe_approx = risk_metrics[1]  # Sharpe ratio approximation
        sharpe_reward = np.tanh(sharpe_approx) * 0.01 * self.reward_weights['sharpe_reward']
        
        # 3. Benchmark comparison (vs Buy & Hold)
        if len(self.portfolio_history) > 1:
            agent_return = (self.total_value - self.portfolio_history[0]) / self.portfolio_history[0]
            benchmark_return = (self.benchmark_value - self.initial_amount) / self.initial_amount
            excess_return = agent_return - benchmark_return
            benchmark_reward = excess_return * self.reward_weights['benchmark_reward']
        else:
            benchmark_reward = 0
        
        # 4. Transaction cost penalty
        transaction_penalty = -(transaction_cost / self.total_value) * self.reward_weights['transaction_penalty']
        
        # 5. Drawdown penalty
        drawdown_penalty = -self.current_drawdown * self.reward_weights['drawdown_penalty']
        
        # 6. Volatility penalty (encourage stable growth)
        volatility = risk_metrics[0]  # Current volatility
        volatility_penalty = -volatility * self.reward_weights['volatility_penalty']
        
        # 7. Action quality reward (reward confident, profitable actions)
        action_quality = 0
        if len(self.portfolio_history) > 1:
            recent_return = portfolio_return
            if recent_return > 0:
                action_quality = confidence * recent_return * 0.1  # Reward confident profitable actions
            else:
                action_quality = -(1 - confidence) * abs(recent_return) * 0.05  # Penalize confident bad actions
        
        # Combine all reward components
        total_reward = (
            portfolio_reward +
            sharpe_reward +
            benchmark_reward +
            transaction_penalty +
            drawdown_penalty +
            volatility_penalty +
            action_quality
        )
        
        # Store daily return for analysis
        if len(self.portfolio_history) > 1:
            daily_return = (self.portfolio_history[-1] - self.portfolio_history[-2]) / self.portfolio_history[-2]
            self.daily_returns.append(daily_return)
        
        return total_reward
    
    def get_trading_statistics(self):
        """Get comprehensive trading statistics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        # Portfolio metrics
        total_return = (self.total_value - self.initial_amount) / self.initial_amount * 100
        
        # Risk metrics
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        volatility = np.std(returns) * np.sqrt(252) * 100
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Trading metrics
        total_trades = len(self.transaction_history)
        buy_trades = sum(1 for t in self.transaction_history if t['type'] == 'buy')
        sell_trades = sum(1 for t in self.transaction_history if t['type'] == 'sell')
        total_costs = sum(t['cost'] for t in self.transaction_history)
        
        # Benchmark comparison
        benchmark_return = (self.benchmark_value - self.initial_amount) / self.initial_amount * 100
        excess_return = total_return - benchmark_return
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown * 100,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_transaction_costs': total_costs,
            'cost_ratio': total_costs / self.initial_amount * 100,
            'final_cash': self.cash,
            'final_holdings': self.holdings,
            'peak_value': self.peak_value
        }

# Utility functions
def create_enhanced_environment(df, **kwargs):
    """Create enhanced environment with default settings"""
    return EnhancedCryptoTradingEnv(df, **kwargs)

def compare_environments(df, steps=100):
    """Compare original vs enhanced environment"""
    from sac import CryptoTradingEnv
    
    # Original environment
    original_env = CryptoTradingEnv(df)
    
    # Enhanced environment  
    enhanced_env = EnhancedCryptoTradingEnv(df)
    
    print("Environment Comparison:")
    print(f"Original observation space: {original_env.observation_space.shape}")
    print(f"Enhanced observation space: {enhanced_env.observation_space.shape}")
    print(f"Original action space: {original_env.action_space.shape}")
    print(f"Enhanced action space: {enhanced_env.action_space.shape}")
    
    return original_env, enhanced_env 