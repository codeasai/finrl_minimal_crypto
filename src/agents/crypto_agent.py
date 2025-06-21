# crypto_agent.py - Unified SAC Agent with Grade System and Metadata Tracking
"""
Unified Cryptocurrency SAC Agent Implementation

คุณสมบัติหลัก:
1. Grade System Integration (N, D, C, B, A, S)
2. Metadata Tracking และ Performance Monitoring
3. Enhanced Trading Environment Support
4. Interactive CLI Integration
5. Modular Design สำหรับ Native Python First Strategy

Usage:
    agent = CryptoSACAgent(grade='C')
    agent.create_environment(data)
    agent.train()
    agent.evaluate()
    agent.save()
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import pickle
import json
import random
import string
from typing import Dict, List, Optional, Any, Tuple

# Core ML imports
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import configuration
from config.config import *
try:
    from config.sac_configs import RL_GradeSelector
    from ..sac.sac_metadata_manager import SAC_AgentMetadata, SAC_MetadataManager
except ImportError:
    print("Warning: sac_configs not available. Using basic configuration.")
    RL_GradeSelector = None
    SAC_AgentMetadata = None
    SAC_MetadataManager = None

class MetadataCallback(BaseCallback):
    """Callback สำหรับ tracking metadata ระหว่างการเทรน"""
    
    def __init__(self, metadata, callback_freq=1000, verbose=0):
        super(MetadataCallback, self).__init__(verbose)
        self.metadata = metadata
        self.callback_freq = callback_freq
        self.step_count = 0
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        if self.step_count % self.callback_freq == 0:
            # Get current reward from training
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                info = self.locals['infos'][0] if isinstance(self.locals['infos'], list) else self.locals['infos']
                reward = info.get('reward', 0)
            else:
                reward = 0
            
            # Add training step to metadata
            if self.metadata:
                if hasattr(self.metadata, 'add_training_step'):
                    self.metadata.add_training_step(
                        timestep=self.step_count,
                        reward=reward,
                        additional_metrics={
                            'learning_rate': self.model.learning_rate,
                            'buffer_size': self.model.replay_buffer.size() if hasattr(self.model, 'replay_buffer') else 0
                        }
                    )
                else:
                    self.metadata['training_history'].append({
                        'timestep': self.step_count,
                        'reward': reward,
                        'timestamp': datetime.now()
                    })
        
        return True

class EnhancedCryptoTradingEnv(gym.Env):
    """Enhanced Cryptocurrency Trading Environment with Grade-based Features"""
    
    def __init__(self, df, grade='C', initial_amount=100000, transaction_cost_pct=0.001, max_holdings=100):
        super(EnhancedCryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.grade = grade
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_holdings = max_holdings
        
        # Grade-specific features
        self.enable_advanced_features = grade in ['A', 'S']
        self.enable_multi_asset = grade in ['B', 'A', 'S'] and len(df['tic'].unique()) > 1
        self.enable_portfolio_management = grade in ['C', 'B', 'A', 'S']
        self.enable_risk_management = grade in ['B', 'A', 'S']
        
        # Current state
        self.current_step = 0
        self.cash = initial_amount
        self.holdings = 0
        self.total_value = initial_amount
        self.trade_history = []
        
        # Setup spaces
        self._setup_action_space()
        self._setup_observation_space()
        
        print(f"🏗️ Created Enhanced Crypto Environment (Grade {grade})")
        print(f"   📊 Features: Advanced={self.enable_advanced_features}, "
              f"Multi-asset={self.enable_multi_asset}, "
              f"Portfolio Mgmt={self.enable_portfolio_management}")
    
    def _setup_action_space(self):
        """Setup action space based on grade"""
        if self.enable_multi_asset:
            # Multi-asset portfolio allocation (-1 to 1 for each asset)
            n_assets = len(self.df['tic'].unique())
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(n_assets,), dtype=np.float32
            )
        else:
            # Single asset position sizing (-1 = sell all, 1 = buy all)
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
    
    def _setup_observation_space(self):
        """Setup observation space based on available indicators"""
        # Count available indicators
        indicator_columns = [col for col in self.df.columns if col in INDICATORS]
        n_indicators = len(indicator_columns)
        
        # Portfolio features
        if self.enable_portfolio_management:
            n_portfolio_features = 5  # cash, holdings, total_value, prev_action, volatility
        else:
            n_portfolio_features = 3  # cash, holdings, total_value
        
        # Market features for advanced grades
        if self.enable_advanced_features:
            n_market_features = 3  # volume_ratio, price_momentum, volatility
        else:
            n_market_features = 0
        
        total_features = n_indicators + n_portfolio_features + n_market_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        print(f"   🔍 Observation space: {total_features} features "
              f"({n_indicators} indicators + {n_portfolio_features} portfolio + {n_market_features} market)")
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_amount
        self.holdings = 0
        self.total_value = self.initial_amount
        self.trade_history = []
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step"""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Get current price
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']
        
        # Calculate portfolio value before action
        prev_total_value = self.cash + self.holdings * current_price
        
        # Execute action
        if self.enable_multi_asset:
            reward = self._execute_multi_asset_action(action, current_data)
        else:
            reward = self._execute_single_asset_action(action[0], current_data)
        
        # Update step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]['close']
            self.total_value = self.cash + self.holdings * next_price
        else:
            self.total_value = self.cash + self.holdings * current_price
        
        # Enhanced reward calculation based on grade
        if self.enable_advanced_features:
            reward = self._calculate_advanced_reward(action, prev_total_value, self.total_value)
        elif self.enable_risk_management:
            reward = self._calculate_risk_adjusted_reward(action, prev_total_value, self.total_value)
        else:
            reward = self._calculate_basic_reward(prev_total_value, self.total_value)
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        obs = self._get_observation()
        info = {
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'current_price': current_price
        }
        
        return obs, reward, done, truncated, info
    
    def _execute_single_asset_action(self, action, current_data):
        """Execute single asset trading action"""
        current_price = current_data['close']
        action = np.clip(action, -1, 1)
        
        if action > 0:  # Buy
            # Calculate how much to buy
            buy_amount = action * self.cash
            shares_to_buy = buy_amount / current_price
            transaction_cost = buy_amount * self.transaction_cost_pct
            
            if buy_amount + transaction_cost <= self.cash:
                self.cash -= (buy_amount + transaction_cost)
                self.holdings += shares_to_buy
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'amount': buy_amount,
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': transaction_cost
                })
        
        elif action < 0:  # Sell
            # Calculate how much to sell
            sell_ratio = abs(action)
            shares_to_sell = sell_ratio * self.holdings
            sell_amount = shares_to_sell * current_price
            transaction_cost = sell_amount * self.transaction_cost_pct
            
            if shares_to_sell <= self.holdings:
                self.holdings -= shares_to_sell
                self.cash += (sell_amount - transaction_cost)
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'amount': sell_amount,
                    'shares': shares_to_sell,
                    'price': current_price,
                    'cost': transaction_cost
                })
        
        return 0  # Basic reward calculation will be done in step()
    
    def _execute_multi_asset_action(self, actions, current_data):
        """Execute multi-asset portfolio action (for advanced grades)"""
        # This is a placeholder for multi-asset trading
        # In practice, would need multi-asset data and more complex logic
        return self._execute_single_asset_action(actions[0], current_data)
    
    def _calculate_basic_reward(self, prev_value, current_value):
        """Basic reward calculation"""
        return (current_value - prev_value) / prev_value
    
    def _calculate_risk_adjusted_reward(self, action, prev_value, current_value):
        """Risk-adjusted reward for grades B+"""
        base_reward = self._calculate_basic_reward(prev_value, current_value)
        
        # Penalize large position changes
        position_change_penalty = abs(np.mean(action)) * 0.01
        
        # Reward stability
        stability_bonus = 0.001 if abs(base_reward) < 0.02 else 0
        
        return base_reward - position_change_penalty + stability_bonus
    
    def _calculate_advanced_reward(self, action, prev_value, current_value):
        """Advanced reward calculation for grades A/S"""
        base_reward = self._calculate_risk_adjusted_reward(action, prev_value, current_value)
        
        # Add Sharpe ratio component
        if len(self.trade_history) > 10:
            recent_returns = [
                (trade.get('amount', 0) - trade.get('cost', 0)) / prev_value 
                for trade in self.trade_history[-10:]
            ]
            if len(recent_returns) > 1:
                sharpe_bonus = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * 0.01
                base_reward += sharpe_bonus
        
        return base_reward
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        current_data = self.df.iloc[self.current_step]
        
        # Technical indicators
        indicator_values = []
        for indicator in INDICATORS:
            if indicator in current_data:
                value = current_data[indicator]
                # Handle NaN values
                if pd.isna(value):
                    value = 0.0
                indicator_values.append(float(value))
        
        # Portfolio features
        current_price = current_data['close']
        portfolio_features = [
            self.cash / self.initial_amount,  # Normalized cash
            self.holdings * current_price / self.initial_amount,  # Normalized holdings value
            self.total_value / self.initial_amount  # Normalized total value
        ]
        
        if self.enable_portfolio_management:
            # Add previous action and volatility
            prev_action = 0.0 if not self.trade_history else self.trade_history[-1].get('amount', 0) / self.initial_amount
            
            # Calculate volatility from recent prices
            if self.current_step >= 10:
                recent_prices = self.df.iloc[self.current_step-10:self.current_step]['close'].values
                volatility = np.std(recent_prices) / np.mean(recent_prices)
            else:
                volatility = 0.0
            
            portfolio_features.extend([prev_action, volatility])
        
        # Market features for advanced grades
        market_features = []
        if self.enable_advanced_features:
            # Volume ratio
            if 'volume' in current_data:
                avg_volume = self.df['volume'].rolling(20).mean().iloc[self.current_step]
                volume_ratio = current_data['volume'] / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # Price momentum
            if self.current_step >= 5:
                price_momentum = (current_price - self.df.iloc[self.current_step-5]['close']) / self.df.iloc[self.current_step-5]['close']
            else:
                price_momentum = 0.0
            
            # Volatility (already calculated above)
            market_features = [volume_ratio, price_momentum, volatility if 'volatility' in locals() else 0.0]
        
        # Combine all features
        observation = np.array(indicator_values + portfolio_features + market_features, dtype=np.float32)
        
        # Handle any remaining NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation

class CryptoSACAgent:
    """Unified SAC Agent with Grade System and Metadata Tracking"""
    
    def __init__(self, grade='C', config=None, agent_id=None):
        """
        Initialize SAC Agent
        
        Args:
            grade: Performance grade (N, D, C, B, A, S)
            config: Custom configuration (optional)
            agent_id: Custom agent ID (optional)
        """
        self.grade = grade
        self.agent_id = agent_id or self._generate_agent_id()
        
        # Load configuration
        self.config = self._load_grade_config(grade, config)
        
        # Initialize metadata tracking
        self.metadata = self._initialize_metadata()
        
        # Core components
        self.model = None
        self.env = None
        self.train_env = None
        self.test_env = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        print(f"🤖 Created SAC Agent: {self.agent_id}")
        print(f"   🎯 Grade: {grade} ({self.config.get('description', 'No description')})")
        print(f"   ⚙️ Timesteps: {self.config.get('total_timesteps', 'N/A'):,}")
        print(f"   💾 Buffer Size: {self.config.get('buffer_size', 'N/A'):,}")
    
    def _generate_agent_id(self):
        """Generate unique agent ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"sac_agent_{timestamp}_{random_suffix}"
    
    def _load_grade_config(self, grade, custom_config=None):
        """Load configuration based on grade"""
        # Try to use RL_GradeSelector if available
        if RL_GradeSelector:
            try:
                base_config = RL_GradeSelector.get_config_by_algorithm_and_grade('SAC', grade)
            except:
                base_config = self._get_default_config(grade)
        else:
            base_config = self._get_default_config(grade)
        
        # Apply custom config if provided
        if custom_config:
            base_config.update(custom_config)
        
        return base_config
    
    def _get_default_config(self, grade):
        """Get default configuration for grade"""
        configs = {
            'N': {
                'total_timesteps': 50000,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'description': 'Novice - Basic learning'
            },
            'D': {
                'total_timesteps': 100000,
                'buffer_size': 100000,
                'learning_starts': 2000,
                'batch_size': 128,
                'learning_rate': 3e-4,
                'description': 'Developing - Improved parameters'
            },
            'C': {
                'total_timesteps': 200000,
                'buffer_size': 250000,
                'learning_starts': 5000,
                'batch_size': 256,
                'learning_rate': 3e-4,
                'description': 'Competent - Professional setup'
            },
            'B': {
                'total_timesteps': 500000,
                'buffer_size': 500000,
                'learning_starts': 10000,
                'batch_size': 512,
                'learning_rate': 3e-4,
                'description': 'Proficient - High performance'
            },
            'A': {
                'total_timesteps': 1000000,
                'buffer_size': 1000000,
                'learning_starts': 25000,
                'batch_size': 1024,
                'learning_rate': 1e-4,
                'description': 'Advanced - Research grade'
            },
            'S': {
                'total_timesteps': 2000000,
                'buffer_size': 2000000,
                'learning_starts': 50000,
                'batch_size': 2048,
                'learning_rate': 1e-4,
                'description': 'Supreme - State-of-the-art'
            }
        }
        
        return configs.get(grade, configs['C'])
    
    def _initialize_metadata(self):
        """Initialize metadata tracking"""
        if SAC_AgentMetadata:
            metadata = SAC_AgentMetadata()
            metadata.agent_id = self.agent_id
            metadata.grade = self.grade
            metadata.set_config(self.config)
            return metadata
        else:
            # Simple metadata tracking if SAC_AgentMetadata not available
            return {
                'agent_id': self.agent_id,
                'grade': self.grade,
                'created_at': datetime.now(),
                'config': self.config,
                'training_history': [],
                'performance_metrics': {}
            }
    
    def create_environment(self, data, test_split=0.2):
        """
        Create trading environment with train/test split
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            test_split: Fraction of data to use for testing
        """
        print(f"🏗️ Creating trading environment...")
        
        # Validate data
        required_columns = ['timestamp', 'tic', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Split data
        split_index = int(len(data) * (1 - test_split))
        train_data = data.iloc[:split_index].reset_index(drop=True)
        test_data = data.iloc[split_index:].reset_index(drop=True)
        
        print(f"   📊 Train data: {len(train_data)} rows")
        print(f"   📊 Test data: {len(test_data)} rows")
        
        # Create environments
        self.train_env = EnhancedCryptoTradingEnv(
            df=train_data,
            grade=self.grade,
            initial_amount=INITIAL_AMOUNT,
            transaction_cost_pct=TRANSACTION_COST_PCT
        )
        
        self.test_env = EnhancedCryptoTradingEnv(
            df=test_data,
            grade=self.grade,
            initial_amount=INITIAL_AMOUNT,
            transaction_cost_pct=TRANSACTION_COST_PCT
        )
        
        # Set primary environment
        self.env = self.train_env
        
        # Validate environment
        try:
            check_env(self.env)
            print("   ✅ Environment validation passed")
        except Exception as e:
            print(f"   ⚠️ Environment validation warning: {e}")
        
        return self.train_env, self.test_env
    
    def train(self, timesteps=None, callback_freq=1000, save_freq=10000, verbose=1):
        """
        Train SAC agent with metadata tracking
        
        Args:
            timesteps: Number of training timesteps (uses config if None)
            callback_freq: Frequency for metadata logging
            save_freq: Frequency for model saving
            verbose: Verbosity level
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment() first.")
        
        timesteps = timesteps or self.config.get('total_timesteps', 100000)
        
        print(f"🚀 Starting SAC training...")
        print(f"   ⏱️ Timesteps: {timesteps:,}")
        print(f"   🎯 Grade: {self.grade}")
        
        # Update metadata
        if hasattr(self.metadata, 'start_training'):
            self.metadata.start_training()
        else:
            self.metadata['training_start_time'] = datetime.now()
        
        # Create SAC model
        sac_params = {
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'buffer_size': self.config.get('buffer_size', 100000),
            'learning_starts': self.config.get('learning_starts', 1000),
            'batch_size': self.config.get('batch_size', 256),
            'tau': self.config.get('tau', 0.005),
            'gamma': self.config.get('gamma', 0.99),
            'train_freq': self.config.get('train_freq', 1),
            'gradient_steps': self.config.get('gradient_steps', 1),
            'verbose': verbose,
            'device': 'auto'
        }
        
        print(f"   ⚙️ SAC Parameters: {sac_params}")
        
        self.model = SAC(
            'MlpPolicy',
            self.env,
            **sac_params
        )
        
        # Create callback for metadata tracking
        callback = MetadataCallback(
            metadata=self.metadata,
            callback_freq=callback_freq,
            verbose=verbose
        )
        
        try:
            # Train model
            start_time = datetime.now()
            self.model.learn(
                total_timesteps=timesteps,
                callback=callback,
                log_interval=callback_freq // 10 if callback_freq >= 10 else 1
            )
            end_time = datetime.now()
            
            # Update training status
            self.is_trained = True
            training_duration = (end_time - start_time).total_seconds()
            
            # Update metadata
            if hasattr(self.metadata, 'end_training'):
                self.metadata.end_training()
                self.metadata.calculate_performance_summary()
            else:
                self.metadata['training_end_time'] = end_time
                self.metadata['training_duration'] = training_duration
            
            print(f"✅ Training completed!")
            print(f"   ⏱️ Duration: {training_duration:.1f} seconds")
            print(f"   🎯 Final timesteps: {timesteps:,}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            
            # Update metadata with error
            if hasattr(self.metadata, 'end_training'):
                self.metadata.end_training()
            
            raise e
    
    def evaluate(self, test_env=None, n_episodes=10, verbose=1):
        """
        Evaluate agent performance
        
        Args:
            test_env: Test environment (uses self.test_env if None)
            n_episodes: Number of evaluation episodes
            verbose: Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        eval_env = test_env or self.test_env
        if eval_env is None:
            raise ValueError("Test environment not available. Call create_environment() first.")
        
        print(f"🔍 Evaluating agent performance...")
        print(f"   📊 Episodes: {n_episodes}")
        
        try:
            # Evaluate using stable-baselines3
            mean_reward, std_reward = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=n_episodes,
                deterministic=True,
                return_episode_rewards=False
            )
            
            # Additional metrics
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(min(n_episodes, 5)):  # Detailed evaluation for first 5 episodes
                obs, _ = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Calculate additional metrics
            results = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'max_reward': float(max(episode_rewards)) if episode_rewards else mean_reward,
                'min_reward': float(min(episode_rewards)) if episode_rewards else mean_reward,
                'mean_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0,
                'total_episodes': n_episodes,
                'evaluation_date': datetime.now()
            }
            
            # Update metadata
            if hasattr(self.metadata, 'add_evaluation_result'):
                self.metadata.add_evaluation_result(results)
                self.metadata.update_performance_metrics(results)
            else:
                if 'evaluation_results' not in self.metadata:
                    self.metadata['evaluation_results'] = []
                self.metadata['evaluation_results'].append(results)
                self.metadata['performance_metrics'].update(results)
            
            print(f"📊 Evaluation Results:")
            print(f"   🎯 Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
            print(f"   🏆 Best Reward: {results['max_reward']:.4f}")
            print(f"   📉 Worst Reward: {results['min_reward']:.4f}")
            print(f"   📏 Avg Episode Length: {results['mean_episode_length']:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            raise e
    
    def save(self, path=None, save_metadata=True):
        """
        Save model and metadata
        
        Args:
            path: Save path (auto-generated if None)
            save_metadata: Whether to save metadata
        """
        if self.model is None:
            raise ValueError("No model to save. Train the agent first.")
        
        # Generate path if not provided
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(MODEL_DIR, "sac", f"sac_agent_{self.grade}_{timestamp}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            # Save model
            model_path = f"{path}.zip"
            self.model.save(model_path)
            
            # Save metadata
            if save_metadata:
                if hasattr(self.metadata, 'to_dict'):
                    metadata_dict = self.metadata.to_dict()
                else:
                    metadata_dict = self.metadata
                
                metadata_path = f"{path}_info.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata_dict, f)
                
                # Also save as JSON for readability
                json_path = f"{path}_metadata.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    # Convert datetime objects to strings for JSON serialization
                    json_dict = self._prepare_for_json(metadata_dict)
                    json.dump(json_dict, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Agent saved successfully!")
            print(f"   📄 Model: {model_path}")
            if save_metadata:
                print(f"   📊 Metadata: {metadata_path}")
                print(f"   📋 JSON: {json_path}")
            
            return path
            
        except Exception as e:
            print(f"❌ Save failed: {str(e)}")
            raise e
    
    def load(self, path, load_metadata=True):
        """
        Load model and metadata
        
        Args:
            path: Load path (without extension)
            load_metadata: Whether to load metadata
        """
        try:
            # Load model
            model_path = f"{path}.zip"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = SAC.load(model_path)
            self.is_trained = True
            
            # Load metadata
            if load_metadata:
                metadata_path = f"{path}_info.pkl"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata_dict = pickle.load(f)
                    
                    if SAC_AgentMetadata and hasattr(SAC_AgentMetadata, 'from_dict'):
                        self.metadata = SAC_AgentMetadata.from_dict(metadata_dict)
                    else:
                        self.metadata = metadata_dict
                    
                    # Update agent properties from metadata
                    if hasattr(self.metadata, 'agent_id'):
                        self.agent_id = self.metadata.agent_id
                        self.grade = self.metadata.grade
                    elif isinstance(self.metadata, dict):
                        self.agent_id = self.metadata.get('agent_id', self.agent_id)
                        self.grade = self.metadata.get('grade', self.grade)
                
                else:
                    print(f"⚠️ Metadata file not found: {metadata_path}")
            
            print(f"✅ Agent loaded successfully!")
            print(f"   🤖 Agent ID: {self.agent_id}")
            print(f"   🎯 Grade: {self.grade}")
            
            return self
            
        except Exception as e:
            print(f"❌ Load failed: {str(e)}")
            raise e
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_info(self):
        """Get agent information"""
        info = {
            'agent_id': self.agent_id,
            'grade': self.grade,
            'is_trained': self.is_trained,
            'config': self.config
        }
        
        if hasattr(self.metadata, 'to_dict'):
            info['metadata'] = self.metadata.to_dict()
        else:
            info['metadata'] = self.metadata
        
        return info
    
    def __str__(self):
        """String representation"""
        status = "Trained" if self.is_trained else "Not Trained"
        return f"CryptoSACAgent(id={self.agent_id}, grade={self.grade}, status={status})"
    
    def __repr__(self):
        return self.__str__()

# Utility functions
def create_crypto_sac_agent(grade='C', config=None):
    """Factory function to create SAC agent"""
    return CryptoSACAgent(grade=grade, config=config)

def load_crypto_sac_agent(path):
    """Factory function to load SAC agent"""
    agent = CryptoSACAgent()
    return agent.load(path)

# Example usage
if __name__ == "__main__":
    print("🤖 Crypto SAC Agent - Unified Implementation")
    print("=" * 50)
    
    # Example: Create and configure agent
    agent = create_crypto_sac_agent(grade='C')
    print(f"Created agent: {agent}")
    
    # Example configuration
    print(f"Configuration: {agent.config}") 