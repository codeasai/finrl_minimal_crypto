# enhanced_sac_trainer.py - Enhanced SAC Trainer with Metadata Management
"""
Enhanced SAC Trainer ที่รวม metadata management และ performance tracking

คุณสมบัติหลัก:
1. รองรับ grade system configuration (N, D, C, B, A, S)
2. บันทึก training progress แบบ real-time
3. เก็บ performance metrics และ timestamps
4. สร้าง reports และ visualizations
5. Auto-save และ checkpoint management
6. Integration กับ Streamlit UI
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# FinRL imports
try:
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
except ImportError:
    print("Warning: FinRL not installed. Some features may not work.")

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Warning: Stable-baselines3 not installed. Please install it.")

# Import metadata manager
from sac_metadata_manager import SAC_AgentMetadata, SAC_MetadataManager, create_sac_metadata_manager

# Import configs
try:
    sys.path.append('../../')
    from rl_agent_configs import SAC_GradeConfigs, SAC_GradeSelector
except ImportError:
    try:
        from sac_configs import SAC_GradeConfigs, SAC_GradeSelector
    except ImportError:
        print("Warning: Cannot import SAC configs. Will use default configurations.")
        SAC_GradeConfigs = None
        SAC_GradeSelector = None

class SAC_MetadataCallback(BaseCallback):
    """Callback สำหรับ track training progress และอัพเดท metadata"""
    
    def __init__(self, metadata: SAC_AgentMetadata, log_freq: int = 1000, verbose=0):
        super(SAC_MetadataCallback, self).__init__(verbose)
        self.metadata = metadata
        self.log_freq = log_freq
        self.last_timestep = 0
        
    def _on_training_start(self) -> None:
        """Called before training starts"""
        self.metadata.start_training()
        if self.verbose > 0:
            print(f"Training started for agent {self.metadata.agent_id}")
    
    def _on_step(self) -> bool:
        """Called at each step"""
        # Log progress at specified frequency
        if self.num_timesteps % self.log_freq == 0:
            # Get current reward (approximation)
            if hasattr(self.training_env, 'get_attr'):
                try:
                    infos = self.training_env.get_attr('get_info')
                    if infos and len(infos) > 0:
                        reward = infos[0].get('reward', 0)
                    else:
                        reward = 0
                except:
                    reward = 0
            else:
                reward = 0
            
            # Add training step to metadata
            self.metadata.add_training_step(
                timestep=self.num_timesteps,
                reward=reward,
                additional_metrics={
                    'learning_rate': self.model.learning_rate,
                    'episode': getattr(self, '_episode_num', 0)
                }
            )
        
        return True
    
    def _on_training_end(self) -> None:
        """Called after training ends"""
        self.metadata.end_training()
        self.metadata.calculate_performance_summary()
        if self.verbose > 0:
            print(f"Training completed for agent {self.metadata.agent_id}")
            print(f"Duration: {self.metadata.get_training_duration_formatted()}")


class Enhanced_SAC_Trainer:
    """Enhanced SAC Trainer with metadata management"""
    
    def __init__(self, metadata_manager: SAC_MetadataManager = None):
        self.metadata_manager = metadata_manager or create_sac_metadata_manager()
        self.current_agent_metadata = None
        self.current_model = None
        self.current_env = None
        
    def create_agent(self, 
                    env, 
                    grade: str = 'C',
                    custom_config: Dict = None,
                    agent_id: str = None) -> Tuple[SAC, SAC_AgentMetadata]:
        """
        สร้าง SAC agent พร้อม metadata
        
        Args:
            env: Trading environment
            grade: Grade level (N, D, C, B, A, S)
            custom_config: Custom configuration (optional)
            agent_id: Custom agent ID (optional)
            
        Returns:
            Tuple of (SAC model, metadata)
        """
        
        # Get configuration
        if custom_config:
            config = custom_config
        else:
            if SAC_GradeSelector:
                config = SAC_GradeSelector.get_config_by_grade(grade)
            else:
                # Default config
                config = {
                    'policy': 'MlpPolicy',
                    'learning_rate': 3e-4,
                    'buffer_size': 100000,
                    'learning_starts': 1000,
                    'batch_size': 256,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1,
                    'verbose': 1,
                    'device': 'auto',
                    'total_timesteps': 50000,
                    'grade': grade
                }
        
        # Create metadata
        metadata = self.metadata_manager.create_agent(grade=grade, config=config)
        if agent_id:
            metadata.agent_id = agent_id
        
        # Set environment config
        metadata.environment_config = {
            'action_space': str(env.action_space),
            'observation_space': str(env.observation_space),
            'env_type': type(env).__name__
        }
        
        # Create SAC model
        model_config = {k: v for k, v in config.items() 
                       if k not in ['total_timesteps', 'eval_freq', 'grade', 'description']}
        
        model = SAC(env=env, **model_config)
        
        # Set paths
        models_dir = "models/sac"
        os.makedirs(models_dir, exist_ok=True)
        
        metadata.model_path = os.path.join(models_dir, f"{metadata.agent_id}.zip")
        metadata.tensorboard_log_path = config.get('tensorboard_log', './logs/sac_graded/')
        
        self.current_model = model
        self.current_agent_metadata = metadata
        self.current_env = env
        
        print(f"✅ Created SAC agent (Grade {grade}): {metadata.agent_id}")
        print(f"📊 Configuration: Buffer={config.get('buffer_size', 'N/A')}, "
              f"LR={config.get('learning_rate', 'N/A')}, "
              f"Timesteps={config.get('total_timesteps', 'N/A')}")
        
        return model, metadata
    
    def train_agent(self, 
                   model: SAC = None, 
                   metadata: SAC_AgentMetadata = None,
                   total_timesteps: int = None,
                   callback_freq: int = 1000,
                   save_freq: int = 10000,
                   eval_freq: int = 5000) -> Dict:
        """
        Train SAC agent with metadata tracking
        
        Args:
            model: SAC model (uses current if None)
            metadata: Agent metadata (uses current if None)
            total_timesteps: Training timesteps (uses config if None)
            callback_freq: Frequency for metadata logging
            save_freq: Frequency for model saving
            eval_freq: Frequency for evaluation
            
        Returns:
            Training results dictionary
        """
        
        if model is None:
            model = self.current_model
        if metadata is None:
            metadata = self.current_agent_metadata
        if total_timesteps is None:
            total_timesteps = metadata.config.get('total_timesteps', 50000)
        
        if model is None or metadata is None:
            raise ValueError("Model and metadata must be provided or set as current")
        
        print(f"🚀 Starting training for agent {metadata.agent_id}")
        print(f"⏱️  Training timesteps: {total_timesteps:,}")
        print(f"📈 Grade: {metadata.grade}")
        
        # Create callback
        metadata_callback = SAC_MetadataCallback(
            metadata=metadata, 
            log_freq=callback_freq, 
            verbose=1
        )
        
        # Start training
        start_time = time.time()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=metadata_callback,
                log_interval=eval_freq,
                progress_bar=True
            )
            
            # Save model
            model.save(metadata.model_path)
            print(f"💾 Model saved: {metadata.model_path}")
            
            # Update metadata
            metadata.total_timesteps_trained = total_timesteps
            
            # Save metadata
            self.metadata_manager.save_agent_metadata(metadata.agent_id)
            
            training_time = time.time() - start_time
            
            print(f"✅ Training completed successfully!")
            print(f"⏱️  Total time: {metadata.get_training_duration_formatted()}")
            
            return {
                'success': True,
                'agent_id': metadata.agent_id,
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'model_path': metadata.model_path
            }
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': metadata.agent_id
            }
    
    def evaluate_agent(self, 
                      model: SAC = None, 
                      metadata: SAC_AgentMetadata = None,
                      eval_env = None,
                      n_eval_episodes: int = 10) -> Dict:
        """
        Evaluate SAC agent performance
        
        Args:
            model: SAC model
            metadata: Agent metadata
            eval_env: Evaluation environment
            n_eval_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        
        if model is None:
            model = self.current_model
        if metadata is None:
            metadata = self.current_agent_metadata
        if eval_env is None:
            eval_env = self.current_env
        
        if model is None or metadata is None or eval_env is None:
            raise ValueError("Model, metadata, and environment must be provided")
        
        print(f"🔍 Evaluating agent {metadata.agent_id}")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate metrics
        eval_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'n_episodes': n_eval_episodes,
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Update metadata
        metadata.add_evaluation_result(eval_results)
        metadata.update_performance_metrics({
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'best_reward': eval_results['max_reward'],
            'worst_reward': eval_results['min_reward']
        })
        
        # Save metadata
        self.metadata_manager.save_agent_metadata(metadata.agent_id)
        
        print(f"📊 Evaluation completed!")
        print(f"🎯 Mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"🏆 Best reward: {eval_results['max_reward']:.4f}")
        
        return eval_results
    
    def load_agent(self, agent_id: str) -> Tuple[Optional[SAC], Optional[SAC_AgentMetadata]]:
        """Load existing agent"""
        metadata = self.metadata_manager.get_agent(agent_id)
        if metadata is None:
            print(f"❌ Agent {agent_id} not found")
            return None, None
        
        if not os.path.exists(metadata.model_path):
            print(f"❌ Model file not found: {metadata.model_path}")
            return None, metadata
        
        try:
            model = SAC.load(metadata.model_path)
            self.current_model = model
            self.current_agent_metadata = metadata
            
            print(f"✅ Loaded agent {agent_id}")
            return model, metadata
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            return None, metadata
    
    def compare_agents(self, agent_ids: List[str] = None) -> pd.DataFrame:
        """Compare multiple agents"""
        return self.metadata_manager.get_performance_comparison(agent_ids)
    
    def get_best_agents(self, grade: str = None, top_n: int = 5) -> List[SAC_AgentMetadata]:
        """Get best performing agents"""
        agents = self.metadata_manager.list_agents(grade=grade)
        
        # Sort by mean reward
        agents.sort(
            key=lambda x: x.performance_metrics.get('mean_reward', -float('inf')), 
            reverse=True
        )
        
        return agents[:top_n]
    
    def create_performance_report(self, agent_id: str = None) -> str:
        """Create detailed performance report"""
        if agent_id is None and self.current_agent_metadata:
            agent_id = self.current_agent_metadata.agent_id
        
        metadata = self.metadata_manager.get_agent(agent_id)
        if metadata is None:
            return f"Agent {agent_id} not found"
        
        report = f"""
=== SAC Agent Performance Report ===
Agent ID: {metadata.agent_id}
Grade: {metadata.grade}
Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}
Last Updated: {metadata.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

=== Training Information ===
Total Timesteps: {metadata.total_timesteps_trained:,}
Training Duration: {metadata.get_training_duration_formatted()}
Training Start: {metadata.training_start_time.strftime('%Y-%m-%d %H:%M:%S') if metadata.training_start_time else 'N/A'}
Training End: {metadata.training_end_time.strftime('%Y-%m-%d %H:%M:%S') if metadata.training_end_time else 'N/A'}

=== Configuration ===
Buffer Size: {metadata.hyperparameters.get('buffer_size', 'N/A')}
Learning Rate: {metadata.hyperparameters.get('learning_rate', 'N/A')}
Batch Size: {metadata.hyperparameters.get('batch_size', 'N/A')}
Gamma: {metadata.hyperparameters.get('gamma', 'N/A')}
Tau: {metadata.hyperparameters.get('tau', 'N/A')}

=== Performance Metrics ===
Mean Reward: {metadata.performance_metrics.get('mean_reward', 'N/A')}
Best Reward: {metadata.performance_metrics.get('best_reward', 'N/A')}
Worst Reward: {metadata.performance_metrics.get('worst_reward', 'N/A')}
Std Reward: {metadata.performance_metrics.get('std_reward', 'N/A')}
Stability Score: {metadata.performance_metrics.get('stability_score', 'N/A')}

=== System Information ===
Python Version: {metadata.system_info.get('python_version', 'N/A')}
CPU Count: {metadata.system_info.get('cpu_count', 'N/A')}
Memory (GB): {metadata.system_info.get('memory_total_gb', 'N/A')}
GPU Available: {metadata.system_info.get('gpu_available', 'N/A')}
GPU Count: {metadata.system_info.get('gpu_count', 'N/A')}

=== Model Files ===
Model Path: {metadata.model_path}
Tensorboard Log: {metadata.tensorboard_log_path}

=== Notes ===
{metadata.notes if metadata.notes else 'No notes'}
"""
        return report


# Utility functions for easy usage
def create_enhanced_sac_trainer() -> Enhanced_SAC_Trainer:
    """Create Enhanced SAC Trainer instance"""
    return Enhanced_SAC_Trainer()

def quick_train_sac(env, grade: str = 'C', timesteps: int = None) -> Dict:
    """Quick SAC training function"""
    trainer = create_enhanced_sac_trainer()
    model, metadata = trainer.create_agent(env, grade=grade)
    
    if timesteps:
        metadata.config['total_timesteps'] = timesteps
    
    results = trainer.train_agent(model, metadata)
    return results

# Example usage
if __name__ == "__main__":
    print("Enhanced SAC Trainer with Metadata Management")
    print("Use this module to train SAC agents with full performance tracking")
    
    # Example: Create trainer
    trainer = create_enhanced_sac_trainer()
    
    # Show available agents
    comparison = trainer.compare_agents()
    print("\n=== Current Agents ===")
    print(comparison) 