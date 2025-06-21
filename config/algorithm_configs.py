# algorithm_configs.py - Configuration for Different RL Algorithms
"""
Algorithm Configuration System for Crypto Trading Agents

รองรับ algorithms:
- SAC (Soft Actor-Critic) - Default, recommended for continuous actions
- PPO (Proximal Policy Optimization) - Stable, good baseline
- DDPG (Deep Deterministic Policy Gradient) - Deterministic actions
- TD3 (Twin Delayed Deep Deterministic) - Improved DDPG
- A2C (Advantage Actor-Critic) - Fast training

แต่ละ algorithm มี configuration ที่ปรับให้เหมาะสมกับ cryptocurrency trading
"""

from typing import Dict, Any, List
import numpy as np

class AlgorithmConfigs:
    """Algorithm configuration manager"""
    
    # Base configurations for each algorithm
    ALGORITHM_CONFIGS = {
        'SAC': {
            'name': 'Soft Actor-Critic',
            'description': 'Off-policy, entropy-regularized, continuous actions',
            'action_space_type': 'continuous',
            'recommended_for': 'cryptocurrency trading (continuous position sizing)',
            'pros': ['Sample efficient', 'Stable training', 'Good exploration'],
            'cons': ['Complex hyperparameters', 'Memory intensive'],
            'default_params': {
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'target_update_interval': 1,
                'use_sde': False,
                'sde_sample_freq': -1
            }
        },
        
        'PPO': {
            'name': 'Proximal Policy Optimization',
            'description': 'On-policy, stable, good baseline performance',
            'action_space_type': 'continuous',
            'recommended_for': 'stable training, baseline comparisons',
            'pros': ['Very stable', 'Simple hyperparameters', 'Robust'],
            'cons': ['Sample inefficient', 'Slower convergence'],
            'default_params': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'clip_range_vf': None,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'use_sde': False,
                'sde_sample_freq': -1
            }
        },
        
        'DDPG': {
            'name': 'Deep Deterministic Policy Gradient',
            'description': 'Off-policy, deterministic, continuous actions',
            'action_space_type': 'continuous',
            'recommended_for': 'deterministic trading strategies',
            'pros': ['Deterministic actions', 'Sample efficient', 'Simple concept'],
            'cons': ['Sensitive to hyperparameters', 'Can be unstable'],
            'default_params': {
                'learning_rate': 1e-3,
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'action_noise': None,  # Will be set based on action space
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5
            }
        },
        
        'TD3': {
            'name': 'Twin Delayed Deep Deterministic',
            'description': 'Improved DDPG with delayed updates and target noise',
            'action_space_type': 'continuous',
            'recommended_for': 'stable deterministic trading',
            'pros': ['More stable than DDPG', 'Good performance', 'Handles overestimation'],
            'cons': ['Complex implementation', 'Many hyperparameters'],
            'default_params': {
                'learning_rate': 1e-3,
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'action_noise': None,  # Will be set based on action space
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5,
                'policy_delay': 2
            }
        },
        
        'A2C': {
            'name': 'Advantage Actor-Critic',
            'description': 'On-policy, fast training, synchronous',
            'action_space_type': 'continuous',
            'recommended_for': 'fast prototyping, resource-constrained training',
            'pros': ['Fast training', 'Low memory', 'Simple'],
            'cons': ['Less stable', 'Lower sample efficiency'],
            'default_params': {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'use_rms_prop': True,
                'rms_prop_eps': 1e-5,
                'use_sde': False,
                'sde_sample_freq': -1
            }
        }
    }
    
    # Grade-based adjustments for each algorithm
    GRADE_ADJUSTMENTS = {
        'N': {  # Novice - Fast training, basic parameters
            'SAC': {
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 64,
                'gradient_steps': 1
            },
            'PPO': {
                'n_steps': 1024,
                'batch_size': 32,
                'n_epochs': 5
            },
            'DDPG': {
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 64
            },
            'TD3': {
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 64
            },
            'A2C': {
                'n_steps': 5
            }
        },
        
        'D': {  # Developing - Balanced parameters
            'SAC': {
                'buffer_size': 100000,
                'learning_starts': 5000,
                'batch_size': 128,
                'gradient_steps': 1
            },
            'PPO': {
                'n_steps': 1024,
                'batch_size': 64,
                'n_epochs': 8
            },
            'DDPG': {
                'buffer_size': 100000,
                'learning_starts': 5000,
                'batch_size': 128
            },
            'TD3': {
                'buffer_size': 100000,
                'learning_starts': 5000,
                'batch_size': 128
            },
            'A2C': {
                'n_steps': 5
            }
        },
        
        'C': {  # Competent - Professional parameters
            'SAC': {
                'buffer_size': 250000,
                'learning_starts': 10000,
                'batch_size': 256,
                'gradient_steps': 2
            },
            'PPO': {
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10
            },
            'DDPG': {
                'buffer_size': 250000,
                'learning_starts': 10000,
                'batch_size': 256
            },
            'TD3': {
                'buffer_size': 250000,
                'learning_starts': 10000,
                'batch_size': 256
            },
            'A2C': {
                'n_steps': 5
            }
        },
        
        'B': {  # Proficient - High performance
            'SAC': {
                'buffer_size': 500000,
                'learning_starts': 10000,
                'batch_size': 256,
                'gradient_steps': 4
            },
            'PPO': {
                'n_steps': 2048,
                'batch_size': 128,
                'n_epochs': 15
            },
            'DDPG': {
                'buffer_size': 500000,
                'learning_starts': 10000,
                'batch_size': 256
            },
            'TD3': {
                'buffer_size': 500000,
                'learning_starts': 10000,
                'batch_size': 256
            },
            'A2C': {
                'n_steps': 10
            }
        },
        
        'A': {  # Advanced - Research grade
            'SAC': {
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 512,
                'gradient_steps': 8,
                'use_sde': True
            },
            'PPO': {
                'n_steps': 4096,
                'batch_size': 256,
                'n_epochs': 20
            },
            'DDPG': {
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 512
            },
            'TD3': {
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 512
            },
            'A2C': {
                'n_steps': 20
            }
        },
        
        'S': {  # Supreme - State-of-the-art
            'SAC': {
                'buffer_size': 2000000,
                'learning_starts': 20000,
                'batch_size': 512,
                'gradient_steps': 16,
                'use_sde': True,
                'learning_rate': 1e-4
            },
            'PPO': {
                'n_steps': 8192,
                'batch_size': 512,
                'n_epochs': 30,
                'learning_rate': 1e-4
            },
            'DDPG': {
                'buffer_size': 2000000,
                'learning_starts': 20000,
                'batch_size': 512,
                'learning_rate': 1e-4
            },
            'TD3': {
                'buffer_size': 2000000,
                'learning_starts': 20000,
                'batch_size': 512,
                'learning_rate': 1e-4
            },
            'A2C': {
                'n_steps': 50,
                'learning_rate': 1e-4
            }
        }
    }
    
    # Training timesteps by grade
    GRADE_TIMESTEPS = {
        'N': 50000,      # Novice
        'D': 100000,     # Developing  
        'C': 200000,     # Competent
        'B': 500000,     # Proficient
        'A': 1000000,    # Advanced
        'S': 2000000     # Supreme
    }
    
    @classmethod
    def get_algorithm_config(cls, algorithm: str, grade: str = 'C') -> Dict[str, Any]:
        """
        Get configuration for specific algorithm and grade
        
        Args:
            algorithm: Algorithm name (SAC, PPO, DDPG, TD3, A2C)
            grade: Agent grade (N, D, C, B, A, S)
            
        Returns:
            Complete configuration dictionary
        """
        if algorithm not in cls.ALGORITHM_CONFIGS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if grade not in cls.GRADE_ADJUSTMENTS:
            raise ValueError(f"Unknown grade: {grade}")
        
        # Start with base config
        config = cls.ALGORITHM_CONFIGS[algorithm].copy()
        base_params = config['default_params'].copy()
        
        # Apply grade adjustments
        grade_adjustments = cls.GRADE_ADJUSTMENTS[grade].get(algorithm, {})
        base_params.update(grade_adjustments)
        
        # Add training timesteps
        config['total_timesteps'] = cls.GRADE_TIMESTEPS[grade]
        config['default_params'] = base_params
        config['grade'] = grade
        
        return config
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithms"""
        return list(cls.ALGORITHM_CONFIGS.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> Dict[str, Any]:
        """Get information about specific algorithm"""
        if algorithm not in cls.ALGORITHM_CONFIGS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        info = cls.ALGORITHM_CONFIGS[algorithm].copy()
        info.pop('default_params', None)  # Remove params for cleaner info
        return info
    
    @classmethod
    def recommend_algorithm(cls, use_case: str = 'crypto_trading') -> str:
        """
        Recommend algorithm based on use case
        
        Args:
            use_case: Use case type
            
        Returns:
            Recommended algorithm name
        """
        recommendations = {
            'crypto_trading': 'SAC',  # Best for continuous actions
            'stable_baseline': 'PPO',  # Most stable
            'deterministic': 'TD3',    # Deterministic actions
            'fast_training': 'A2C',    # Fastest training
            'research': 'SAC'          # Research purposes
        }
        
        return recommendations.get(use_case, 'SAC')
    
    @classmethod
    def print_algorithm_comparison(cls):
        """Print comparison table of all algorithms"""
        print("🤖 Algorithm Comparison Table")
        print("=" * 80)
        print(f"{'Algorithm':<8} {'Type':<12} {'Best For':<25} {'Pros':<30}")
        print("-" * 80)
        
        for algo, config in cls.ALGORITHM_CONFIGS.items():
            name = config['name']
            best_for = config['recommended_for'][:23] + ".." if len(config['recommended_for']) > 25 else config['recommended_for']
            pros = ', '.join(config['pros'][:2])[:28] + ".." if len(', '.join(config['pros'][:2])) > 30 else ', '.join(config['pros'][:2])
            
            print(f"{algo:<8} {'Continuous':<12} {best_for:<25} {pros:<30}")
        
        print("\n💡 Recommendations:")
        print("   🎯 Crypto Trading: SAC (best balance of performance and stability)")
        print("   🏃 Fast Training: A2C (quickest results)")
        print("   🛡️ Most Stable: PPO (reliable baseline)")
        print("   🎲 Deterministic: TD3 (consistent actions)")

# Convenience functions
def get_algorithm_config(algorithm: str, grade: str = 'C') -> Dict[str, Any]:
    """Convenience function to get algorithm config"""
    return AlgorithmConfigs.get_algorithm_config(algorithm, grade)

def list_algorithms() -> List[str]:
    """Convenience function to list available algorithms"""
    return AlgorithmConfigs.get_available_algorithms()

def recommend_algorithm(use_case: str = 'crypto_trading') -> str:
    """Convenience function to get algorithm recommendation"""
    return AlgorithmConfigs.recommend_algorithm(use_case)

# Example usage
if __name__ == "__main__":
    # Print algorithm comparison
    AlgorithmConfigs.print_algorithm_comparison()
    
    # Example configurations
    print(f"\n📊 Example Configurations:")
    print("-" * 50)
    
    for grade in ['N', 'C', 'A']:
        print(f"\nGrade {grade} SAC Configuration:")
        config = AlgorithmConfigs.get_algorithm_config('SAC', grade)
        print(f"   Timesteps: {config['total_timesteps']:,}")
        print(f"   Buffer Size: {config['default_params']['buffer_size']:,}")
        print(f"   Batch Size: {config['default_params']['batch_size']}")
        print(f"   Gradient Steps: {config['default_params']['gradient_steps']}")
    
    print(f"\n🎯 Recommendation for crypto trading: {recommend_algorithm('crypto_trading')}") 