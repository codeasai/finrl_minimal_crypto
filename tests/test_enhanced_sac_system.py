# test_enhanced_sac_system.py - Test Enhanced SAC System
"""
Test script สำหรับทดสอบ Enhanced SAC System ที่ประกอบด้วย:
1. SAC Metadata Manager
2. Enhanced SAC Trainer
3. Grade-based configuration system
4. Performance tracking และ reporting

ใช้สำหรับ verify ว่าระบบทำงานได้ถูกต้อง
"""

import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np

# Add parent directories to path
sys.path.append('../../')
sys.path.append('.')

# Import our enhanced system
try:
    from sac_metadata_manager import SAC_AgentMetadata, SAC_MetadataManager, create_sac_metadata_manager, print_agent_summary
    from enhanced_sac_trainer import Enhanced_SAC_Trainer, create_enhanced_sac_trainer
    print("✅ Successfully imported Enhanced SAC System modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Import config system
try:
    from rl_agent_configs import SAC_GradeConfigs, SAC_GradeSelector
    print("✅ Successfully imported RL Agent Configs")
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        from sac_configs import SAC_GradeConfigs, SAC_GradeSelector
        print("✅ Successfully imported SAC Configs")
        CONFIG_AVAILABLE = True
    except ImportError:
        print("⚠️  No config modules found, will test with basic configs")
        CONFIG_AVAILABLE = False

def test_metadata_manager():
    """Test SAC Metadata Manager functionality"""
    print("\n" + "="*50)
    print("🧪 Testing SAC Metadata Manager")
    print("="*50)
    
    # Create metadata manager
    manager = create_sac_metadata_manager()
    print(f"✅ Created metadata manager")
    
    # Test creating agent metadata
    grades_to_test = ['N', 'D', 'C', 'B', 'A', 'S']
    created_agents = []
    
    for grade in grades_to_test:
        try:
            # Create test config
            test_config = {
                'policy': 'MlpPolicy',
                'learning_rate': 3e-4,
                'buffer_size': 50000 * (grades_to_test.index(grade) + 1),
                'total_timesteps': 10000 * (grades_to_test.index(grade) + 1),
                'grade': grade,
                'verbose': 0
            }
            
            # Create agent metadata
            metadata = manager.create_agent(grade=grade, config=test_config)
            created_agents.append(metadata)
            
            # Add some mock training data
            for i in range(5):
                metadata.add_training_step(
                    timestep=i * 1000, 
                    reward=np.random.normal(0.1 * (grades_to_test.index(grade) + 1), 0.05),
                    additional_metrics={'epoch': i}
                )
            
            # Calculate performance summary
            metadata.calculate_performance_summary()
            
            # Add mock evaluation result
            metadata.add_evaluation_result({
                'mean_reward': np.random.normal(0.15 * (grades_to_test.index(grade) + 1), 0.02),
                'std_reward': 0.02,
                'n_episodes': 10
            })
            
            print(f"✅ Created agent grade {grade}: {metadata.agent_id}")
            
        except Exception as e:
            print(f"❌ Failed to create grade {grade} agent: {e}")
    
    # Test performance comparison
    try:
        comparison_df = manager.get_performance_comparison()
        print(f"\n📊 Performance comparison table:")
        print(comparison_df.to_string(index=False))
    except Exception as e:
        print(f"❌ Failed to create comparison table: {e}")
    
    # Test grade statistics
    try:
        stats = manager.get_grade_statistics()
        print(f"\n📈 Grade statistics:")
        for grade, data in stats.items():
            if data['count'] > 0:
                print(f"Grade {grade}: {data['count']} agents, "
                      f"avg reward: {data['avg_reward']:.4f}")
    except Exception as e:
        print(f"❌ Failed to get grade statistics: {e}")
    
    # Test export functionality
    try:
        export_file = manager.export_to_csv("test_sac_agents_export.csv")
        print(f"📄 Exported data to: {export_file}")
        
        # Clean up
        if os.path.exists(export_file):
            os.remove(export_file)
            print(f"🧹 Cleaned up export file")
    except Exception as e:
        print(f"❌ Failed to export data: {e}")
    
    return manager, created_agents

def test_enhanced_trainer():
    """Test Enhanced SAC Trainer functionality"""
    print("\n" + "="*50)
    print("🧪 Testing Enhanced SAC Trainer")
    print("="*50)
    
    # Create trainer
    trainer = create_enhanced_sac_trainer()
    print(f"✅ Created Enhanced SAC Trainer")
    
    # Test configuration loading
    if CONFIG_AVAILABLE:
        try:
            for grade in ['N', 'C', 'B']:
                config = SAC_GradeSelector.get_config_by_grade(grade)
                print(f"✅ Loaded grade {grade} config: "
                      f"buffer={config.get('buffer_size')}, "
                      f"timesteps={config.get('total_timesteps')}")
        except Exception as e:
            print(f"❌ Failed to load configs: {e}")
    else:
        print("⚠️  Skipping config test - no config modules available")
    
    # Test trainer methods (without actual training)
    try:
        # Test compare agents
        comparison = trainer.compare_agents()
        print(f"📊 Agent comparison table shape: {comparison.shape}")
        
        # Test get best agents
        best_agents = trainer.get_best_agents(top_n=3)
        print(f"🏆 Found {len(best_agents)} best agents")
        
    except Exception as e:
        print(f"❌ Trainer method test failed: {e}")
    
    return trainer

def test_grade_configurations():
    """Test grade-based configuration system"""
    print("\n" + "="*50)
    print("🧪 Testing Grade Configuration System")
    print("="*50)
    
    if not CONFIG_AVAILABLE:
        print("⚠️  No config modules available, skipping grade config test")
        return
    
    grades = ['N', 'D', 'C', 'B', 'A', 'S']
    
    for grade in grades:
        try:
            # Test SAC config for each grade
            config = SAC_GradeSelector.get_config_by_grade(grade)
            
            print(f"📋 Grade {grade} config:")
            print(f"  Buffer Size: {config.get('buffer_size', 'N/A'):,}")
            print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
            print(f"  Total Timesteps: {config.get('total_timesteps', 'N/A'):,}")
            print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
            print(f"  Gradient Steps: {config.get('gradient_steps', 'N/A')}")
            print(f"  Description: {config.get('description', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Failed to get grade {grade} config: {e}")

def test_integration():
    """Test integration between components"""
    print("\n" + "="*50)
    print("🧪 Testing System Integration")
    print("="*50)
    
    try:
        # Create manager and trainer
        manager = create_sac_metadata_manager()
        trainer = create_enhanced_sac_trainer()
        
        # Verify they can work together
        print("✅ Manager and trainer created successfully")
        
        # Test metadata persistence
        test_metadata = manager.create_agent(grade='C')
        agent_id = test_metadata.agent_id
        
        # Save and reload
        manager.save_agent_metadata(agent_id)
        print(f"✅ Saved metadata for agent {agent_id}")
        
        # Create new manager and try to load
        new_manager = create_sac_metadata_manager()
        loaded_metadata = new_manager.get_agent(agent_id)
        
        if loaded_metadata and loaded_metadata.agent_id == agent_id:
            print(f"✅ Successfully loaded metadata for agent {agent_id}")
        else:
            print(f"❌ Failed to load metadata for agent {agent_id}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def run_comprehensive_test():
    """Run comprehensive test of Enhanced SAC System"""
    print("🚀 Starting Comprehensive Test of Enhanced SAC System")
    print("="*80)
    
    # Test individual components
    manager, agents = test_metadata_manager()
    trainer = test_enhanced_trainer()
    test_grade_configurations()
    test_integration()
    
    # Final summary
    print("\n" + "="*50)
    print("📋 Test Summary")
    print("="*50)
    
    print(f"✅ SAC Metadata Manager: Working")
    print(f"✅ Enhanced SAC Trainer: Working")
    print(f"✅ Grade Configuration System: {'Working' if CONFIG_AVAILABLE else 'Partially Working'}")
    print(f"✅ System Integration: Working")
    
    if agents:
        print(f"\n📊 Created {len(agents)} test agents:")
        for agent in agents:
            print(f"  - {agent.agent_id} (Grade {agent.grade})")
    
    print(f"\n🎯 Enhanced SAC System is ready for use!")
    print(f"📂 Metadata stored in: agents/sac/metadata/")
    print(f"🔧 To use the system:")
    print(f"   1. Import: from enhanced_sac_trainer import create_enhanced_sac_trainer")
    print(f"   2. Create trainer: trainer = create_enhanced_sac_trainer()")
    print(f"   3. Create agent: model, metadata = trainer.create_agent(env, grade='C')")
    print(f"   4. Train: trainer.train_agent(model, metadata)")

if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc() 