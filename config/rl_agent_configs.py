# rl_agent_configs.py - Optimized RL Agent Configurations for Different Grades
"""
Reinforcement Learning Agent Configurations ที่ optimize สำหรับระดับ Agent Grades ต่างๆ
โดยใช้ Grade System: N, D, C, B, A, S (จากต่ำไปสูง)

รองรับ RL Algorithms:
- SAC (Soft Actor-Critic) - Off-policy, continuous actions
- PPO (Proximal Policy Optimization) - On-policy, stable
- DQN (Deep Q-Network) - Off-policy, discrete actions  
- DDPG (Deep Deterministic Policy Gradient) - Off-policy, continuous actions
- A2C (Advantage Actor-Critic) - On-policy, fast
- TD3 (Twin Delayed DDPG) - Off-policy, improved DDPG

Grade หมายถึง:
- N (Novice): เริ่มต้น, ทรัพยากรน้อย, simple strategies
- D (Developing): กำลังพัฒนา, moderate resources
- C (Competent): ความสามารถดี, standard resources  
- B (Proficient): เชี่ยวชาญ, high resources
- A (Advanced): ระดับสูง, very high resources
- S (Supreme): ระดับสูงสุด, maximum resources, complex strategies
"""

import os
import numpy as np

# Import config หลัก
from config import *

def linear_schedule(initial_value, final_value=None):
    """
    Linear learning rate schedule
    
    Args:
        initial_value (float): Initial learning rate
        final_value (float, optional): Final learning rate. If None, will use initial_value
        
    Returns:
        function: Scheduler function
    """
    if final_value is None:
        final_value = initial_value
        
    def scheduler(progress_remaining):
        """
        Progress will go from 1 (beginning) to 0 (end)
        """
        return final_value + progress_remaining * (initial_value - final_value)
    
    return scheduler

class BaseGradeConfigs:
    """Base class สำหรับ RL Agent Configurations"""
    
    @staticmethod
    def get_common_config(algorithm: str):
        """Common configuration สำหรับทุก algorithms"""
        return {
            'policy': 'MlpPolicy',
            'verbose': 1,
            'device': 'auto',
            'seed': 42,
            'tensorboard_log': f'./logs/{algorithm.lower()}_graded/',
        }

class SAC_GradeConfigs(BaseGradeConfigs):
    """SAC Configurations สำหรับแต่ละ Grade Level"""
    
    @staticmethod
    def get_base_config():
        """Base configuration สำหรับ SAC"""
        config = BaseGradeConfigs.get_common_config('SAC')
        config.update({
            'algorithm': 'SAC',
            'action_space_type': 'continuous'
        })
        return config
    
    @staticmethod
    def get_grade_N_config():
        """
        Grade N (Novice) - เริ่มต้น
        - ทรัพยากรน้อย, training เร็ว
        - เหมาะสำหรับการทดลองและเรียนรู้
        - Stable แต่ไม่ซับซ้อน
        """
        config = SAC_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 3e-4,              # Standard learning rate
            'buffer_size': 50000,               # Small buffer สำหรับ memory
            'learning_starts': 1000,            # เริ่มเรียนรู้เร็ว
            'batch_size': 128,                  # Small batch
            'tau': 0.005,                       # Standard target update
            'gamma': 0.99,                      # Standard discount
            'train_freq': 1,                    # เทรนบ่อย
            'gradient_steps': 1,                # Simple gradient steps
            'target_update_interval': 1,
            'ent_coef': 0.2,                   # Fixed entropy (stable)
            'target_entropy': 'auto',
            'use_sde': False,                   # ไม่ใช้ SDE (simple)
            'total_timesteps': 50000,           # Training สั้น
            'eval_freq': 5000,                  # Evaluate บ่อย
            'grade': 'N',
            'description': 'Novice: Fast training, stable but simple'
        })
        return config
    
    @staticmethod
    def get_grade_D_config():
        """
        Grade D (Developing) - กำลังพัฒนา
        - ทรัพยากรปานกลาง
        - เริ่มใช้ advanced features
        - Balance ระหว่าง speed และ performance
        """
        config = SAC_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 1e-4,              # ลดลงเพื่อ stability
            'buffer_size': 100000,              # เพิ่มขนาด buffer
            'learning_starts': 2000,            # เรียนรู้หลังจากมีข้อมูลมากขึ้น
            'batch_size': 256,                  # Larger batch
            'tau': 0.01,                        # Faster target update
            'gamma': 0.995,                     # Long-term thinking
            'train_freq': 2,                    # ลด frequency เล็กน้อย
            'gradient_steps': 2,                # Multiple gradient steps
            'target_update_interval': 1,
            'ent_coef': 'auto',                # Automatic entropy tuning
            'target_entropy': 'auto',
            'use_sde': False,                   # ยังไม่ใช้ SDE
            'total_timesteps': 100000,          # Training นานขึ้น
            'eval_freq': 10000,
            'grade': 'D',
            'description': 'Developing: Balanced speed and performance'
        })
        return config
    
    @staticmethod
    def get_grade_C_config():
        """
        Grade C (Competent) - ความสามารถดี
        - Standard resources
        - ใช้ advanced features มากขึ้น
        - Good performance baseline
        """
        config = SAC_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(1e-4, 3e-5),  # Learning rate decay
            'buffer_size': 250000,              # Larger buffer
            'learning_starts': 5000,            # More warm-up
            'batch_size': 512,                  # Large batch
            'tau': 0.01,                        # Fast target update
            'gamma': 0.995,                     # Long-term focus
            'train_freq': 4,                    # Less frequent training
            'gradient_steps': 4,                # Multiple gradient steps
            'target_update_interval': 1,
            'ent_coef': 'auto',                # Auto entropy
            'target_entropy': 'auto',
            'use_sde': True,                    # เริ่มใช้ SDE
            'sde_sample_freq': 64,              # SDE sampling
            'total_timesteps': 200000,          # Longer training
            'eval_freq': 15000,
            'grade': 'C',
            'description': 'Competent: Good baseline performance'
        })
        return config
    
    @staticmethod
    def get_grade_B_config():
        """
        Grade B (Proficient) - เชี่ยวชาญ
        - High resources
        - Advanced optimization techniques
        - Focus on performance
        """
        config = SAC_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(3e-4, 1e-5),  # More aggressive decay
            'buffer_size': 500000,              # Large buffer
            'learning_starts': 10000,           # Extensive warm-up
            'batch_size': 1024,                 # Very large batch
            'tau': 0.02,                        # Faster target update
            'gamma': 0.999,                     # Very long-term
            'train_freq': 8,                    # Less frequent but intensive
            'gradient_steps': 8,                # Many gradient steps
            'target_update_interval': 1,
            'ent_coef': 'auto',                # Auto entropy
            'target_entropy': 'auto',
            'use_sde': True,                    # Advanced exploration
            'sde_sample_freq': 32,              # More frequent SDE
            'total_timesteps': 500000,          # Long training
            'eval_freq': 25000,
            'grade': 'B',
            'description': 'Proficient: High performance focus'
        })
        return config
    
    @staticmethod
    def get_grade_A_config():
        """
        Grade A (Advanced) - ระดับสูง
        - Very high resources
        - Advanced techniques + ensemble ready
        - High performance requirements
        """
        config = SAC_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(3e-4, 5e-6),  # Extended decay
            'buffer_size': 1000000,             # Very large buffer
            'learning_starts': 20000,           # Extensive warm-up
            'batch_size': 2048,                 # Massive batch
            'tau': 0.02,                        # Fast target update
            'gamma': 0.999,                     # Maximum long-term
            'train_freq': 16,                   # Intensive training
            'gradient_steps': 16,               # Many gradient steps
            'target_update_interval': 1,
            'ent_coef': 'auto',                # Auto entropy
            'target_entropy': 'auto',
            'use_sde': True,                    # Advanced exploration
            'sde_sample_freq': 16,              # High frequency SDE
            'total_timesteps': 1000000,         # Very long training
            'eval_freq': 50000,
            'policy_kwargs': {                  # Advanced policy network
                'net_arch': [512, 512, 256],
                'activation_fn': 'relu'
            },
            'grade': 'A',
            'description': 'Advanced: Very high performance, ensemble ready'
        })
        return config
    
    @staticmethod
    def get_grade_S_config():
        """
        Grade S (Supreme) - ระดับสูงสุด
        - Maximum resources
        - All advanced techniques
        - Supreme performance
        - Research-grade configuration
        """
        config = SAC_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(5e-4, 1e-6),  # Complex decay
            'buffer_size': 2000000,             # Maximum buffer
            'learning_starts': 50000,           # Massive warm-up
            'batch_size': 4096,                 # Maximum batch
            'tau': 0.025,                       # Aggressive target update
            'gamma': 0.9995,                    # Supreme long-term
            'train_freq': 32,                   # Maximum intensive training
            'gradient_steps': 32,               # Maximum gradient steps
            'target_update_interval': 1,
            'ent_coef': 'auto',                # Auto entropy
            'target_entropy': 'auto',
            'use_sde': True,                    # Maximum exploration
            'sde_sample_freq': 8,               # Maximum frequency SDE
            'total_timesteps': 2000000,         # Supreme training
            'eval_freq': 100000,
            'policy_kwargs': {                  # Supreme policy network
                'net_arch': [1024, 1024, 512, 256],
                'activation_fn': 'relu',
                'dropout': 0.1
            },
            'grade': 'S',
            'description': 'Supreme: Maximum performance, research-grade'
        })
        return config

class PPO_GradeConfigs(BaseGradeConfigs):
    """PPO Configurations สำหรับแต่ละ Grade Level"""
    
    @staticmethod
    def get_base_config():
        """Base configuration สำหรับ PPO"""
        config = BaseGradeConfigs.get_common_config('PPO')
        config.update({
            'algorithm': 'PPO',
            'action_space_type': 'both'
        })
        return config
    
    @staticmethod
    def get_grade_N_config():
        """Grade N (Novice) - PPO"""
        config = PPO_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 128,
            'n_epochs': 4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'total_timesteps': 50000,
            'eval_freq': 5000,
            'grade': 'N',
            'description': 'PPO Novice: Simple and stable'
        })
        return config
    
    @staticmethod
    def get_grade_D_config():
        """Grade D (Developing) - PPO"""
        config = PPO_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(3e-4, 1e-4),
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 8,
            'gamma': 0.995,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'total_timesteps': 100000,
            'eval_freq': 10000,
            'grade': 'D',
            'description': 'PPO Developing: Balanced performance'
        })
        return config
    
    @staticmethod
    def get_grade_C_config():
        """Grade C (Competent) - PPO"""
        config = PPO_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(3e-4, 5e-5),
            'n_steps': 4096,
            'batch_size': 512,
            'n_epochs': 10,
            'gamma': 0.995,
            'gae_lambda': 0.98,
            'clip_range': linear_schedule(0.2, 0.1),
            'ent_coef': 0.005,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'total_timesteps': 200000,
            'eval_freq': 15000,
            'grade': 'C',
            'description': 'PPO Competent: Good baseline'
        })
        return config
    
    @staticmethod
    def get_grade_B_config():
        """Grade B (Proficient) - PPO"""
        config = PPO_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(3e-4, 1e-5),
            'n_steps': 8192,
            'batch_size': 1024,
            'n_epochs': 15,
            'gamma': 0.999,
            'gae_lambda': 0.98,
            'clip_range': linear_schedule(0.2, 0.05),
            'ent_coef': 0.001,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'total_timesteps': 500000,
            'eval_freq': 25000,
            'grade': 'B',
            'description': 'PPO Proficient: High performance'
        })
        return config
    
    @staticmethod
    def get_grade_A_config():
        """Grade A (Advanced) - PPO"""
        config = PPO_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(3e-4, 5e-6),
            'n_steps': 16384,
            'batch_size': 2048,
            'n_epochs': 20,
            'gamma': 0.999,
            'gae_lambda': 0.99,
            'clip_range': linear_schedule(0.2, 0.02),
            'ent_coef': 0.0005,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {
                'net_arch': [512, 512, 256],
                'activation_fn': 'relu'
            },
            'total_timesteps': 1000000,
            'eval_freq': 50000,
            'grade': 'A',
            'description': 'PPO Advanced: Very high performance'
        })
        return config
    
    @staticmethod
    def get_grade_S_config():
        """Grade S (Supreme) - PPO"""
        config = PPO_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(5e-4, 1e-6),
            'n_steps': 32768,
            'batch_size': 4096,
            'n_epochs': 30,
            'gamma': 0.9995,
            'gae_lambda': 0.99,
            'clip_range': linear_schedule(0.3, 0.01),
            'ent_coef': 0.0001,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {
                'net_arch': [1024, 1024, 512, 256],
                'activation_fn': 'relu',
                'dropout': 0.1
            },
            'total_timesteps': 2000000,
            'eval_freq': 100000,
            'grade': 'S',
            'description': 'PPO Supreme: Maximum performance'
        })
        return config

class DQN_GradeConfigs(BaseGradeConfigs):
    """DQN Configurations สำหรับแต่ละ Grade Level"""
    
    @staticmethod
    def get_base_config():
        """Base configuration สำหรับ DQN"""
        config = BaseGradeConfigs.get_common_config('DQN')
        config.update({
            'algorithm': 'DQN',
            'action_space_type': 'discrete'
        })
        return config
    
    @staticmethod
    def get_grade_N_config():
        """Grade N (Novice) - DQN"""
        config = DQN_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 1e-4,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000,
            'eval_freq': 5000,
            'grade': 'N',
            'description': 'DQN Novice: Simple discrete actions'
        })
        return config
    
    @staticmethod
    def get_grade_D_config():
        """Grade D (Developing) - DQN"""
        config = DQN_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 5e-5,
            'buffer_size': 100000,
            'learning_starts': 2000,
            'batch_size': 64,
            'tau': 1.0,
            'gamma': 0.995,
            'train_freq': 8,
            'gradient_steps': 2,
            'target_update_interval': 2000,
            'exploration_fraction': 0.4,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.02,
            'total_timesteps': 100000,
            'eval_freq': 10000,
            'grade': 'D',
            'description': 'DQN Developing: Improved exploration'
        })
        return config
    
    @staticmethod
    def get_grade_C_config():
        """Grade C (Competent) - DQN"""
        config = DQN_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': linear_schedule(1e-4, 1e-5),
            'buffer_size': 250000,
            'learning_starts': 5000,
            'batch_size': 128,
            'tau': 1.0,
            'gamma': 0.995,
            'train_freq': 16,
            'gradient_steps': 4,
            'target_update_interval': 5000,
            'exploration_fraction': 0.5,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
            'total_timesteps': 200000,
            'eval_freq': 15000,
            'grade': 'C',
            'description': 'DQN Competent: Advanced exploration'
        })
        return config

class DDPG_GradeConfigs(BaseGradeConfigs):
    """DDPG Configurations สำหรับแต่ละ Grade Level"""
    
    @staticmethod
    def get_base_config():
        """Base configuration สำหรับ DDPG"""
        config = BaseGradeConfigs.get_common_config('DDPG')
        config.update({
            'algorithm': 'DDPG',
            'action_space_type': 'continuous'
        })
        return config
    
    @staticmethod
    def get_grade_N_config():
        """Grade N (Novice) - DDPG"""
        config = DDPG_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 1e-4,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'batch_size': 128,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'action_noise': 'normal',
            'total_timesteps': 50000,
            'eval_freq': 5000,
            'grade': 'N',
            'description': 'DDPG Novice: Basic continuous control'
        })
        return config

class TD3_GradeConfigs(BaseGradeConfigs):
    """TD3 Configurations สำหรับแต่ละ Grade Level"""
    
    @staticmethod
    def get_base_config():
        """Base configuration สำหรับ TD3"""
        config = BaseGradeConfigs.get_common_config('TD3')
        config.update({
            'algorithm': 'TD3',
            'action_space_type': 'continuous'
        })
        return config
    
    @staticmethod
    def get_grade_N_config():
        """Grade N (Novice) - TD3"""
        config = TD3_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 1e-4,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'batch_size': 128,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'total_timesteps': 50000,
            'eval_freq': 5000,
            'grade': 'N',
            'description': 'TD3 Novice: Improved DDPG'
        })
        return config

class A2C_GradeConfigs(BaseGradeConfigs):
    """A2C Configurations สำหรับแต่ละ Grade Level"""
    
    @staticmethod
    def get_base_config():
        """Base configuration สำหรับ A2C"""
        config = BaseGradeConfigs.get_common_config('A2C')
        config.update({
            'algorithm': 'A2C',
            'action_space_type': 'both'
        })
        return config
    
    @staticmethod
    def get_grade_N_config():
        """Grade N (Novice) - A2C"""
        config = A2C_GradeConfigs.get_base_config()
        config.update({
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'rms_prop_eps': 1e-5,
            'use_rms_prop': True,
            'total_timesteps': 50000,
            'eval_freq': 5000,
            'grade': 'N',
            'description': 'A2C Novice: Fast and simple'
        })
        return config

class RL_GradeSelector:
    """Universal RL Agent Selector สำหรับทุก algorithms"""
    
    # Algorithm mappings
    ALGORITHM_CONFIGS = {
        'SAC': SAC_GradeConfigs,
        'PPO': PPO_GradeConfigs,
        'DQN': DQN_GradeConfigs,
        'DDPG': DDPG_GradeConfigs,
        'TD3': TD3_GradeConfigs,
        'A2C': A2C_GradeConfigs
    }
    
    @staticmethod
    def get_available_algorithms():
        """รายการ algorithms ที่รองรับ"""
        return list(RL_GradeSelector.ALGORITHM_CONFIGS.keys())
    
    @staticmethod
    def get_config_by_algorithm_and_grade(algorithm: str, grade: str):
        """
        รับ config ตาม algorithm และ grade
        
        Args:
            algorithm (str): Algorithm name ('SAC', 'PPO', 'DQN', etc.)
            grade (str): Grade level ('N', 'D', 'C', 'B', 'A', 'S')
            
        Returns:
            dict: RL Agent configuration
        """
        algorithm = algorithm.upper()
        grade = grade.upper()
        
        if algorithm not in RL_GradeSelector.ALGORITHM_CONFIGS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Available: {RL_GradeSelector.get_available_algorithms()}")
        
        config_class = RL_GradeSelector.ALGORITHM_CONFIGS[algorithm]
        
        # ตรวจสอบว่ามี grade method หรือไม่
        grade_method = f'get_grade_{grade}_config'
        if not hasattr(config_class, grade_method):
            # ถ้าไม่มี grade นั้น ให้ใช้ grade N แทน
            if hasattr(config_class, 'get_grade_N_config'):
                print(f"⚠️ Grade {grade} not available for {algorithm}, using Grade N instead")
                grade_method = 'get_grade_N_config'
            else:
                raise ValueError(f"No configurations available for {algorithm}")
        
        return getattr(config_class, grade_method)()
    
    @staticmethod
    def get_recommended_algorithm_for_crypto(action_space_type: str = 'continuous'):
        """
        แนะนำ algorithm สำหรับ cryptocurrency trading
        
        Args:
            action_space_type (str): 'continuous', 'discrete', or 'both'
            
        Returns:
            list: Recommended algorithms in order of preference
        """
        if action_space_type == 'continuous':
            return ['SAC', 'TD3', 'DDPG', 'PPO']
        elif action_space_type == 'discrete':
            return ['DQN', 'PPO', 'A2C']
        else:  # both
            return ['PPO', 'SAC', 'A2C', 'TD3']
    
    @staticmethod
    def get_config_by_performance(algorithm: str, target_return: float, available_time_hours: int = 24):
        """
        เลือก config ตาม algorithm, target performance และเวลาที่มี
        
        Args:
            algorithm (str): Algorithm name
            target_return (float): Target return (%)
            available_time_hours (int): Available training time (hours)
            
        Returns:
            dict: Recommended RL Agent configuration
        """
        if target_return <= 5 or available_time_hours <= 2:
            grade = 'N'
        elif target_return <= 10 or available_time_hours <= 6:
            grade = 'D'
        elif target_return <= 15 or available_time_hours <= 12:
            grade = 'C'
        elif target_return <= 25 or available_time_hours <= 24:
            grade = 'B'
        elif target_return <= 40 or available_time_hours <= 48:
            grade = 'A'
        else:
            grade = 'S'
        
        return RL_GradeSelector.get_config_by_algorithm_and_grade(algorithm, grade)
    
    @staticmethod
    def get_config_by_resources(algorithm: str, ram_gb: int, gpu_available: bool = False):
        """
        เลือก config ตาม algorithm และทรัพยากรที่มี
        
        Args:
            algorithm (str): Algorithm name
            ram_gb (int): Available RAM in GB
            gpu_available (bool): GPU availability
            
        Returns:
            dict: Recommended RL Agent configuration
        """
        if ram_gb <= 8:
            grade = 'N'
        elif ram_gb <= 16:
            grade = 'D'
        elif ram_gb <= 32:
            grade = 'C'
        elif ram_gb <= 64 and not gpu_available:
            grade = 'B'
        elif ram_gb <= 64 and gpu_available:
            grade = 'A'
        else:  # ram_gb > 64 and gpu_available
            grade = 'S'
        
        return RL_GradeSelector.get_config_by_algorithm_and_grade(algorithm, grade)

# Legacy support - เก็บไว้เพื่อ backward compatibility
class SAC_GradeSelector:
    """Legacy SAC Selector - รองรับ backward compatibility"""
    
    @staticmethod
    def get_config_by_grade(grade: str):
        """Legacy method - ใช้ SAC เป็นค่าเริ่มต้น"""
        return RL_GradeSelector.get_config_by_algorithm_and_grade('SAC', grade)
    
    @staticmethod
    def get_config_by_performance(target_return: float, available_time_hours: int = 24):
        """Legacy method - ใช้ SAC เป็นค่าเริ่มต้น"""
        return RL_GradeSelector.get_config_by_performance('SAC', target_return, available_time_hours)
    
    @staticmethod
    def get_config_by_resources(ram_gb: int, gpu_available: bool = False):
        """Legacy method - ใช้ SAC เป็นค่าเริ่มต้น"""
        return RL_GradeSelector.get_config_by_resources('SAC', ram_gb, gpu_available)

def print_algorithm_comparison():
    """แสดงการเปรียบเทียบ algorithms ทั้งหมด"""
    algorithms = RL_GradeSelector.get_available_algorithms()
    
    print("🤖 RL Algorithms Comparison")
    print("=" * 80)
    
    for algorithm in algorithms:
        try:
            config = RL_GradeSelector.get_config_by_algorithm_and_grade(algorithm, 'N')
            action_space = config.get('action_space_type', 'unknown')
            description = config.get('description', 'No description')
            
            print(f"\n{algorithm}:")
            print(f"  Action Space: {action_space}")
            print(f"  Description: {description}")
        except Exception as e:
            print(f"\n{algorithm}: Configuration not available")

def print_grade_comparison():
    """แสดงการเปรียบเทียบ configs ทุก grades สำหรับ SAC"""
    grades = ['N', 'D', 'C', 'B', 'A', 'S']
    
    print("🎯 SAC Configuration Comparison by Grade")
    print("=" * 80)
    
    configs = []
    for grade in grades:
        try:
            config = RL_GradeSelector.get_config_by_algorithm_and_grade('SAC', grade)
            configs.append(config)
        except:
            configs.append(None)
    
    # Key parameters to compare
    key_params = [
        'buffer_size', 'total_timesteps', 'batch_size', 
        'gradient_steps', 'learning_starts'
    ]
    
    print(f"{'Grade':<6} {'Buffer':<10} {'Timesteps':<12} {'Batch':<8} {'Grad Steps':<12} {'Learning Starts':<15}")
    print("-" * 80)
    
    for i, grade in enumerate(grades):
        config = configs[i]
        if config:
            print(f"{grade:<6} {config.get('buffer_size', 'N/A'):<10} {config.get('total_timesteps', 'N/A'):<12} "
                  f"{config.get('batch_size', 'N/A'):<8} {config.get('gradient_steps', 'N/A'):<12} {config.get('learning_starts', 'N/A'):<15}")
        else:
            print(f"{grade:<6} {'N/A':<10} {'N/A':<12} {'N/A':<8} {'N/A':<12} {'N/A':<15}")
    
    print("\n📝 Grade Descriptions:")
    print("-" * 40)
    for i, grade in enumerate(grades):
        config = configs[i]
        if config:
            print(f"{grade}: {config.get('description', 'No description')}")

def get_recommended_grade_for_crypto():
    """
    แนะนำ grade สำหรับ cryptocurrency trading
    ตาม market characteristics
    """
    print("🪙 Recommended Grades for Cryptocurrency Trading")
    print("=" * 60)
    
    recommendations = {
        'Beginner Trader': {
            'algorithm': 'PPO',
            'grade': 'D',
            'reason': 'Stable and easy to understand',
            'suitable_for': 'Learning, small portfolios'
        },
        'Active Trader': {
            'algorithm': 'SAC',
            'grade': 'B',
            'reason': 'Excellent for continuous action spaces',
            'suitable_for': 'Daily trading, medium portfolios'
        },
        'Professional Trader': {
            'algorithm': 'SAC',
            'grade': 'A',
            'reason': 'Advanced performance for professional use',
            'suitable_for': 'Large portfolios, algorithmic trading'
        },
        'Research & Development': {
            'algorithm': 'TD3',
            'grade': 'S',
            'reason': 'State-of-the-art for continuous control',
            'suitable_for': 'Strategy development, academic research'
        }
    }
    
    for trader_type, info in recommendations.items():
        print(f"\n{trader_type}:")
        print(f"  Recommended Algorithm: {info['algorithm']}")
        print(f"  Recommended Grade: {info['grade']}")
        print(f"  Reason: {info['reason']}")
        print(f"  Suitable for: {info['suitable_for']}")

# Example usage functions
def create_rl_agent_by_algorithm_and_grade(algorithm: str, grade: str, env):
    """
    สร้าง RL agent ตาม algorithm และ grade ที่กำหนด
    
    Args:
        algorithm (str): Algorithm name
        grade (str): Grade level
        env: Trading environment
        
    Returns:
        tuple: (agent, config)
    """
    config = RL_GradeSelector.get_config_by_algorithm_and_grade(algorithm, grade)
    
    # Import algorithm class
    algorithm_imports = {
        'SAC': 'from stable_baselines3 import SAC',
        'PPO': 'from stable_baselines3 import PPO',
        'DQN': 'from stable_baselines3 import DQN',
        'DDPG': 'from stable_baselines3 import DDPG',
        'TD3': 'from stable_baselines3 import TD3',
        'A2C': 'from stable_baselines3 import A2C'
    }
    
    if algorithm not in algorithm_imports:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # แยก parameters สำหรับ agent
    agent_params = config.copy()
    
    # ลบ parameters ที่ไม่ใช่ของ agent
    non_agent_params = ['total_timesteps', 'eval_freq', 'grade', 'description', 'algorithm', 'action_space_type']
    for param in non_agent_params:
        if param in agent_params:
            del agent_params[param]
    
    print(f"🤖 Creating {algorithm} Agent - Grade {grade}")
    print(f"📝 Description: {config['description']}")
    print(f"⏱️ Training timesteps: {config['total_timesteps']:,}")
    if 'buffer_size' in config:
        print(f"💾 Buffer size: {config['buffer_size']:,}")
    if 'batch_size' in config:
        print(f"🎯 Batch size: {config['batch_size']}")
    
    # สร้าง agent ตาม algorithm
    if algorithm == 'SAC':
        from stable_baselines3 import SAC
        agent = SAC(env=env, **agent_params)
    elif algorithm == 'PPO':
        from stable_baselines3 import PPO
        agent = PPO(env=env, **agent_params)
    elif algorithm == 'DQN':
        from stable_baselines3 import DQN
        agent = DQN(env=env, **agent_params)
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG
        agent = DDPG(env=env, **agent_params)
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3
        agent = TD3(env=env, **agent_params)
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C
        agent = A2C(env=env, **agent_params)
    else:
        raise ValueError(f"Algorithm {algorithm} not implemented")
    
    return agent, config

# Legacy function - รองรับ backward compatibility
def create_sac_agent_by_grade(grade: str, env):
    """Legacy function - สร้าง SAC agent ตาม grade"""
    return create_rl_agent_by_algorithm_and_grade('SAC', grade, env)

if __name__ == "__main__":
    # แสดงการเปรียบเทียบ algorithms
    print_algorithm_comparison()
    print("\n")
    
    # แสดงการเปรียบเทียบ SAC grades
    print_grade_comparison()
    print("\n")
    
    # แสดงคำแนะนำสำหรับ crypto trading
    get_recommended_grade_for_crypto()
    
    # ตัวอย่างการใช้งาน
    print(f"\n🔧 Example Usage:")
    print("=" * 40)
    
    # เลือกตาม algorithm และ grade
    config_sac_b = RL_GradeSelector.get_config_by_algorithm_and_grade('SAC', 'B')
    print(f"SAC Grade B: {config_sac_b['description']}")
    
    config_ppo_a = RL_GradeSelector.get_config_by_algorithm_and_grade('PPO', 'A')
    print(f"PPO Grade A: {config_ppo_a['description']}")
    
    # เลือกตาม performance target
    config_perf = RL_GradeSelector.get_config_by_performance('TD3', target_return=15.0, available_time_hours=12)
    print(f"TD3 Performance-based: Grade {config_perf['grade']}")
    
    # เลือกตาม resources
    config_resource = RL_GradeSelector.get_config_by_resources('DDPG', ram_gb=32, gpu_available=True)
    print(f"DDPG Resource-based: Grade {config_resource['grade']}")
    
    # แสดง recommended algorithms สำหรับ crypto
    print(f"\n🪙 Crypto Trading Recommendations:")
    print(f"Continuous Actions: {RL_GradeSelector.get_recommended_algorithm_for_crypto('continuous')}")
    print(f"Discrete Actions: {RL_GradeSelector.get_recommended_algorithm_for_crypto('discrete')}")
    print(f"Both Action Types: {RL_GradeSelector.get_recommended_algorithm_for_crypto('both')}")
    
    # Legacy compatibility test
    print(f"\n🔄 Legacy Compatibility Test:")
    legacy_config = SAC_GradeSelector.get_config_by_grade('B')
    print(f"Legacy SAC Grade B: {legacy_config['description']}") 