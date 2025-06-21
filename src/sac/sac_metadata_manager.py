# sac_metadata_manager.py - SAC Agent Metadata and Performance Management System
"""
SAC Agent Metadata Manager สำหรับการจัดการข้อมูลประสิทธิภาพและ configuration ของ SAC agents

คุณสมบัติหลัก:
1. เก็บ configuration ตาม grade system (N, D, C, B, A, S)
2. บันทึก performance metrics ระหว่างการเทรน
3. ติดตาม training duration และ timestamps
4. สร้าง reports และ comparisons
5. จัดการ model versioning
6. Export/Import metadata เป็น JSON และ CSV
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Tuple
import uuid
import time

# Import เพื่อใช้ grade configs
try:
    from rl_agent_configs import SAC_GradeConfigs, SAC_GradeSelector
except ImportError:
    try:
        from sac_configs import SAC_GradeConfigs, SAC_GradeSelector
    except ImportError:
        print("Warning: Cannot import SAC configs. Will use default configurations.")
        SAC_GradeConfigs = None
        SAC_GradeSelector = None

class SAC_AgentMetadata:
    """Class สำหรับเก็บข้อมูล metadata ของ SAC agent แต่ละตัว"""
    
    def __init__(self, agent_id: str = None, grade: str = 'C'):
        self.agent_id = agent_id or self._generate_agent_id()
        self.grade = grade
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Configuration data
        self.config = {}
        self.hyperparameters = {}
        
        # Training data
        self.training_history = []
        self.training_start_time = None
        self.training_end_time = None
        self.total_training_duration = 0  # in seconds
        self.total_timesteps_trained = 0
        
        # Performance metrics
        self.performance_metrics = {
            'final_reward': None,
            'mean_reward': None,
            'std_reward': None,
            'best_reward': None,
            'worst_reward': None,
            'success_rate': None,
            'convergence_timestep': None,
            'stability_score': None
        }
        
        # Evaluation results
        self.evaluation_results = []
        self.backtest_results = {}
        
        # Model file paths
        self.model_path = None
        self.tensorboard_log_path = None
        
        # Additional metadata
        self.environment_config = {}
        self.system_info = {}
        self.notes = ""
        
    def _generate_agent_id(self) -> str:
        """Generate unique agent ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:6].upper()
        return f"sac_agent_{timestamp}_{random_suffix}"
    
    def set_config(self, config: Dict):
        """Set agent configuration"""
        self.config = config.copy()
        self.updated_at = datetime.now()
        
        # Extract hyperparameters (exclude functions and non-serializable objects)
        self.hyperparameters = {}
        for k, v in config.items():
            if k not in ['policy', 'verbose', 'device', 'tensorboard_log']:
                # Handle function objects (like learning rate schedules)
                if callable(v):
                    self.hyperparameters[k] = f"<function: {v.__name__}>" if hasattr(v, '__name__') else "<function>"
                else:
                    self.hyperparameters[k] = v
    
    def start_training(self):
        """Mark training start time"""
        self.training_start_time = datetime.now()
        self.updated_at = datetime.now()
    
    def end_training(self):
        """Mark training end time and calculate duration"""
        if self.training_start_time:
            self.training_end_time = datetime.now()
            self.total_training_duration = (self.training_end_time - self.training_start_time).total_seconds()
        self.updated_at = datetime.now()
    
    def add_training_step(self, timestep: int, reward: float, additional_metrics: Dict = None):
        """Add training step data"""
        step_data = {
            'timestep': timestep,
            'reward': reward,
            'timestamp': datetime.now()
        }
        if additional_metrics:
            step_data.update(additional_metrics)
        
        self.training_history.append(step_data)
        self.total_timesteps_trained = max(self.total_timesteps_trained, timestep)
        self.updated_at = datetime.now()
    
    def update_performance_metrics(self, metrics: Dict):
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        self.updated_at = datetime.now()
    
    def add_evaluation_result(self, eval_data: Dict):
        """Add evaluation result"""
        eval_data['timestamp'] = datetime.now()
        self.evaluation_results.append(eval_data)
        self.updated_at = datetime.now()
    
    def calculate_performance_summary(self):
        """Calculate performance summary from training history"""
        if not self.training_history:
            return
        
        rewards = [step['reward'] for step in self.training_history]
        
        self.performance_metrics.update({
            'final_reward': rewards[-1] if rewards else None,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'best_reward': np.max(rewards),
            'worst_reward': np.min(rewards)
        })
        
        # Calculate stability score (negative coefficient of variation)
        if self.performance_metrics['mean_reward'] != 0:
            cv = self.performance_metrics['std_reward'] / abs(self.performance_metrics['mean_reward'])
            self.performance_metrics['stability_score'] = 1 / (1 + cv)
        
        self.updated_at = datetime.now()
    
    def get_training_duration_formatted(self) -> str:
        """Get formatted training duration"""
        if self.total_training_duration == 0:
            return "0 seconds"
        
        hours = int(self.total_training_duration // 3600)
        minutes = int((self.total_training_duration % 3600) // 60)
        seconds = int(self.total_training_duration % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'grade': self.grade,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'config': self.config,
            'hyperparameters': self.hyperparameters,
            'training_history': [
                {**step, 'timestamp': step['timestamp'].isoformat()} 
                for step in self.training_history
            ],
            'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
            'training_end_time': self.training_end_time.isoformat() if self.training_end_time else None,
            'total_training_duration': self.total_training_duration,
            'total_timesteps_trained': self.total_timesteps_trained,
            'performance_metrics': self.performance_metrics,
            'evaluation_results': [
                {**result, 'timestamp': result['timestamp'].isoformat()} 
                for result in self.evaluation_results
            ],
            'backtest_results': self.backtest_results,
            'model_path': self.model_path,
            'tensorboard_log_path': self.tensorboard_log_path,
            'environment_config': self.environment_config,
            'system_info': self.system_info,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create instance from dictionary"""
        obj = cls()
        obj.agent_id = data['agent_id']
        obj.grade = data['grade']
        obj.created_at = datetime.fromisoformat(data['created_at'])
        obj.updated_at = datetime.fromisoformat(data['updated_at'])
        obj.config = data['config']
        obj.hyperparameters = data['hyperparameters']
        
        # Parse training history
        obj.training_history = [
            {**step, 'timestamp': datetime.fromisoformat(step['timestamp'])} 
            for step in data['training_history']
        ]
        
        obj.training_start_time = datetime.fromisoformat(data['training_start_time']) if data['training_start_time'] else None
        obj.training_end_time = datetime.fromisoformat(data['training_end_time']) if data['training_end_time'] else None
        obj.total_training_duration = data['total_training_duration']
        obj.total_timesteps_trained = data['total_timesteps_trained']
        obj.performance_metrics = data['performance_metrics']
        
        # Parse evaluation results
        obj.evaluation_results = [
            {**result, 'timestamp': datetime.fromisoformat(result['timestamp'])} 
            for result in data['evaluation_results']
        ]
        
        obj.backtest_results = data['backtest_results']
        obj.model_path = data['model_path']
        obj.tensorboard_log_path = data['tensorboard_log_path']
        obj.environment_config = data['environment_config']
        obj.system_info = data['system_info']
        obj.notes = data['notes']
        
        return obj


class SAC_MetadataManager:
    """Manager class สำหรับจัดการ metadata ของ SAC agents ทั้งหมด"""
    
    def __init__(self, metadata_dir: str = "agents/sac/metadata"):
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        
        self.agents = {}  # Dict[agent_id, SAC_AgentMetadata]
        self.load_all_metadata()
    
    def create_agent(self, grade: str = 'C', config: Dict = None) -> SAC_AgentMetadata:
        """Create new agent metadata"""
        metadata = SAC_AgentMetadata(grade=grade)
        
        # Set config from grade if not provided
        if config is None and SAC_GradeConfigs:
            try:
                config = SAC_GradeSelector.get_config_by_grade(grade)
            except:
                config = {}
        
        if config:
            metadata.set_config(config)
        
        # Add system information
        try:
            import psutil
            import torch
            metadata.system_info = {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError:
            metadata.system_info = {'note': 'System info not available'}
        
        self.agents[metadata.agent_id] = metadata
        self.save_agent_metadata(metadata.agent_id)
        
        return metadata
    
    def get_agent(self, agent_id: str) -> Optional[SAC_AgentMetadata]:
        """Get agent metadata by ID"""
        return self.agents.get(agent_id)
    
    def list_agents(self, grade: str = None) -> List[SAC_AgentMetadata]:
        """List all agents, optionally filtered by grade"""
        agents = list(self.agents.values())
        if grade:
            agents = [agent for agent in agents if agent.grade == grade]
        return sorted(agents, key=lambda x: x.created_at, reverse=True)
    
    def save_agent_metadata(self, agent_id: str):
        """Save agent metadata to file"""
        if agent_id not in self.agents:
            return
        
        metadata = self.agents[agent_id]
        filepath = os.path.join(self.metadata_dir, f"{agent_id}_metadata.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load_agent_metadata(self, agent_id: str) -> bool:
        """Load agent metadata from file"""
        filepath = os.path.join(self.metadata_dir, f"{agent_id}_metadata.json")
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = SAC_AgentMetadata.from_dict(data)
            self.agents[agent_id] = metadata
            return True
        except Exception as e:
            print(f"Error loading metadata for {agent_id}: {e}")
            return False
    
    def load_all_metadata(self):
        """Load all metadata files"""
        if not os.path.exists(self.metadata_dir):
            return
        
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('_metadata.json'):
                agent_id = filename.replace('_metadata.json', '')
                self.load_agent_metadata(agent_id)
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent metadata"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
            filepath = os.path.join(self.metadata_dir, f"{agent_id}_metadata.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return True
        return False
    
    def get_performance_comparison(self, agent_ids: List[str] = None) -> pd.DataFrame:
        """Get performance comparison table"""
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        
        data = []
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            # Safe rounding for numeric values
            mean_reward = agent.performance_metrics.get('mean_reward')
            best_reward = agent.performance_metrics.get('best_reward')
            stability_score = agent.performance_metrics.get('stability_score')
            
            data.append({
                'Agent ID': agent_id,
                'Grade': agent.grade,
                'Created': agent.created_at.strftime('%Y-%m-%d %H:%M'),
                'Training Duration': agent.get_training_duration_formatted(),
                'Total Timesteps': agent.total_timesteps_trained,
                'Mean Reward': round(mean_reward, 4) if mean_reward is not None else 0.0,
                'Best Reward': round(best_reward, 4) if best_reward is not None else 0.0,
                'Stability Score': round(stability_score, 4) if stability_score is not None else 0.0,
                'Buffer Size': agent.hyperparameters.get('buffer_size', 'N/A'),
                'Learning Rate': agent.hyperparameters.get('learning_rate', 'N/A')
            })
        
        return pd.DataFrame(data)
    
    def get_grade_statistics(self) -> Dict:
        """Get statistics by grade"""
        stats = {}
        
        for grade in ['N', 'D', 'C', 'B', 'A', 'S']:
            grade_agents = [agent for agent in self.agents.values() if agent.grade == grade]
            
            if not grade_agents:
                stats[grade] = {'count': 0}
                continue
            
            rewards = []
            durations = []
            timesteps = []
            
            for agent in grade_agents:
                if agent.performance_metrics.get('mean_reward'):
                    rewards.append(agent.performance_metrics['mean_reward'])
                if agent.total_training_duration > 0:
                    durations.append(agent.total_training_duration)
                if agent.total_timesteps_trained > 0:
                    timesteps.append(agent.total_timesteps_trained)
            
            stats[grade] = {
                'count': len(grade_agents),
                'avg_reward': np.mean(rewards) if rewards else 0,
                'avg_duration_hours': np.mean(durations) / 3600 if durations else 0,
                'avg_timesteps': np.mean(timesteps) if timesteps else 0,
                'best_reward': max(rewards) if rewards else 0
            }
        
        return stats
    
    def export_to_csv(self, filepath: str = None):
        """Export all agent data to CSV"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"sac_agents_export_{timestamp}.csv"
        
        df = self.get_performance_comparison()
        df.to_csv(filepath, index=False)
        return filepath
    
    def get_best_agents_by_grade(self, top_n: int = 3) -> Dict:
        """Get best performing agents by grade"""
        results = {}
        
        for grade in ['N', 'D', 'C', 'B', 'A', 'S']:
            grade_agents = [agent for agent in self.agents.values() if agent.grade == grade]
            
            # Sort by mean reward (descending)
            grade_agents.sort(
                key=lambda x: x.performance_metrics.get('mean_reward', -float('inf')), 
                reverse=True
            )
            
            results[grade] = grade_agents[:top_n]
        
        return results


# Utility functions
def create_sac_metadata_manager() -> SAC_MetadataManager:
    """Create and return SAC metadata manager instance"""
    return SAC_MetadataManager()

def print_agent_summary(metadata: SAC_AgentMetadata):
    """Print agent summary"""
    print(f"\n=== SAC Agent Summary ===")
    print(f"Agent ID: {metadata.agent_id}")
    print(f"Grade: {metadata.grade}")
    print(f"Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training Duration: {metadata.get_training_duration_formatted()}")
    print(f"Total Timesteps: {metadata.total_timesteps_trained:,}")
    
    if metadata.performance_metrics.get('mean_reward'):
        print(f"Mean Reward: {metadata.performance_metrics['mean_reward']:.4f}")
        print(f"Best Reward: {metadata.performance_metrics['best_reward']:.4f}")
        print(f"Stability Score: {metadata.performance_metrics.get('stability_score', 0):.4f}")
    
    print(f"Buffer Size: {metadata.hyperparameters.get('buffer_size', 'N/A')}")
    print(f"Learning Rate: {metadata.hyperparameters.get('learning_rate', 'N/A')}")
    
    if metadata.notes:
        print(f"Notes: {metadata.notes}")

# Example usage
if __name__ == "__main__":
    # สร้าง metadata manager
    manager = create_sac_metadata_manager()
    
    # สร้าง agent ใหม่
    agent_metadata = manager.create_agent(grade='B')
    print_agent_summary(agent_metadata)
    
    # แสดง comparison table
    print("\n=== Performance Comparison ===")
    df = manager.get_performance_comparison()
    print(df)
    
    # แสดง statistics by grade
    print("\n=== Grade Statistics ===")
    stats = manager.get_grade_statistics()
    for grade, data in stats.items():
        if data['count'] > 0:
            print(f"Grade {grade}: {data['count']} agents, "
                  f"avg reward: {data['avg_reward']:.4f}, "
                  f"avg duration: {data['avg_duration_hours']:.1f}h") 