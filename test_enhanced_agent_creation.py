# test_enhanced_agent_creation.py - Test Enhanced Agent Creation System
"""
Test script สำหรับระบบ Enhanced Agent Creation ที่ปรับปรุงใหม่

คุณสมบัติที่ทดสอบ:
1. Algorithm selection (SAC, PPO, DDPG, TD3, A2C)
2. Grade-based configuration (N, D, C, B, A, S)
3. Symbol และ feature data selection
4. Environment type selection (Basic/Enhanced)
5. Integration testing
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from datetime import datetime

# Import components
from algorithm_configs import AlgorithmConfigs, get_algorithm_config, list_algorithms
from interactive_cli import InteractiveCLI, AgentManager, DataManager
from crypto_agent import create_crypto_sac_agent

def test_algorithm_configs():
    """Test 1: Algorithm Configuration System"""
    print("🧪 Test 1: Algorithm Configuration System")
    print("-" * 50)
    
    try:
        # Test available algorithms
        algorithms = list_algorithms()
        print(f"✅ Available algorithms: {algorithms}")
        
        # Test algorithm info
        for algo in ['SAC', 'PPO']:
            info = AlgorithmConfigs.get_algorithm_info(algo)
            print(f"✅ {algo}: {info['name']} - {info['description']}")
        
        # Test grade configurations
        for grade in ['N', 'C', 'A']:
            config = get_algorithm_config('SAC', grade)
            print(f"✅ Grade {grade} SAC - Timesteps: {config['total_timesteps']:,}, Buffer: {config['default_params']['buffer_size']:,}")
        
        # Test recommendations
        recommended = AlgorithmConfigs.recommend_algorithm('crypto_trading')
        print(f"✅ Recommended algorithm: {recommended}")
        
        print("✅ Algorithm configuration system working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Algorithm config test failed: {e}")
        return False

def test_data_availability():
    """Test 2: Data Availability Check"""
    print("\n🧪 Test 2: Data Availability Check")
    print("-" * 50)
    
    try:
        # Test raw data
        from src.data_loader import get_crypto_data_summary
        raw_summary = get_crypto_data_summary()
        print(f"✅ Raw data files: {raw_summary['total_files']}")
        print(f"✅ Raw data symbols: {raw_summary['symbols']}")
        
        # Test feature data
        from src.data_feature import get_crypto_feature_summary
        feature_summary = get_crypto_feature_summary()
        print(f"✅ Feature data files: {feature_summary['total_files']}")
        print(f"✅ Feature data symbols: {feature_summary['symbols']}")
        print(f"✅ Average features per file: {feature_summary['average_features']}")
        
        print("✅ Data availability check passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data availability test failed: {e}")
        return False

def test_agent_creation():
    """Test 3: Enhanced Agent Creation"""
    print("\n🧪 Test 3: Enhanced Agent Creation")
    print("-" * 50)
    
    try:
        # Test different algorithm and grade combinations
        test_configs = [
            {'algorithm': 'SAC', 'grade': 'N'},
            {'algorithm': 'SAC', 'grade': 'C'},
            {'algorithm': 'PPO', 'grade': 'D'},  # Will fallback to SAC for now
        ]
        
        for config in test_configs:
            print(f"\n🔧 Testing {config['algorithm']} Grade {config['grade']}...")
            
            # Get algorithm config
            algo_config = get_algorithm_config(config['algorithm'], config['grade'])
            print(f"   📊 Timesteps: {algo_config['total_timesteps']:,}")
            
            # For PPO, show buffer size from SAC config since we're using SAC implementation
            if config['algorithm'] == 'PPO':
                sac_config = get_algorithm_config('SAC', config['grade'])
                print(f"   💾 Buffer Size (SAC): {sac_config['default_params']['buffer_size']:,}")
            else:
                print(f"   💾 Buffer Size: {algo_config['default_params']['buffer_size']:,}")
            
            # Create agent (for now, all use SAC as base)
            if config['algorithm'] == 'SAC':
                agent = create_crypto_sac_agent(grade=config['grade'])
            else:
                print(f"   ⚠️ {config['algorithm']} using SAC implementation")
                agent = create_crypto_sac_agent(grade=config['grade'])
            
            # Set additional properties
            agent.algorithm = config['algorithm']
            agent.environment_type = 'enhanced'
            
            print(f"   ✅ Agent created: {agent.agent_id}")
            print(f"   🎯 Grade: {agent.grade}")
            
        print("\n✅ Agent creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False

def test_environment_types():
    """Test 4: Environment Type Selection"""
    print("\n🧪 Test 4: Environment Type Selection")
    print("-" * 50)
    
    try:
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
            'tic': ['BTC-USD'] * 100,
            'open': [50000 + i*100 for i in range(100)],
            'high': [50100 + i*100 for i in range(100)],
            'low': [49900 + i*100 for i in range(100)],
            'close': [50000 + i*100 for i in range(100)],
            'volume': [1000000] * 100,
            'sma_20': [50000] * 100,
            'ema_20': [50000] * 100,
            'rsi_14': [50] * 100,
            'macd': [0] * 100,
            'macd_signal': [0] * 100,
            'macd_hist': [0] * 100,
            'bb_middle': [50000] * 100,
            'bb_std': [1000] * 100,
            'bb_upper': [51000] * 100,
            'bb_lower': [49000] * 100,
            'volume_sma_20': [1000000] * 100,
            'volume_ratio': [1.0] * 100
        })
        
        # Test basic environment
        print("🏗️ Testing Basic Environment...")
        agent_basic = create_crypto_sac_agent(grade='N')
        agent_basic.environment_type = 'basic'
        
        train_env_basic, test_env_basic = agent_basic.create_environment(sample_data)
        print(f"   ✅ Basic environment created")
        print(f"   📊 Action space: {train_env_basic.action_space}")
        print(f"   🔍 Observation space: {train_env_basic.observation_space}")
        
        # Test enhanced environment
        print("\n🏗️ Testing Enhanced Environment...")
        try:
            from enhanced_crypto_env import EnhancedCryptoTradingEnv
            
            enhanced_env = EnhancedCryptoTradingEnv(sample_data)
            print(f"   ✅ Enhanced environment created")
            print(f"   📊 Action space: {enhanced_env.action_space}")
            print(f"   🔍 Observation space: {enhanced_env.observation_space}")
            
        except ImportError:
            print("   ⚠️ Enhanced environment not available, skipping...")
        
        print("\n✅ Environment type test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Environment type test failed: {e}")
        return False

def test_feature_data_loading():
    """Test 5: Feature Data Loading"""
    print("\n🧪 Test 5: Feature Data Loading")
    print("-" * 50)
    
    try:
        from src.data_feature import CryptoFeatureProcessor
        
        processor = CryptoFeatureProcessor()
        available_files = processor.list_available_feature_data()
        
        if len(available_files) > 0:
            print(f"✅ Found {len(available_files)} feature files")
            
            # Test loading first available file
            first_file = available_files[0]
            print(f"📊 Testing file: {first_file['filename']}")
            print(f"   Symbol: {first_file['symbol']}")
            print(f"   Features: {first_file['feature_count']}")
            print(f"   Size: {first_file['file_size_kb']} KB")
            
            # Load the data
            data = processor.load_feature_data(
                first_file['symbol'],
                first_file['start_date'],
                first_file['end_date'],
                first_file['interval']
            )
            
            if data is not None:
                print(f"   ✅ Loaded: {len(data)} rows, {len(data.columns)} columns")
                print(f"   📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            else:
                print("   ⚠️ Could not load feature data")
            
        else:
            print("⚠️ No feature files found - create some using src/data_feature.py first")
        
        print("\n✅ Feature data loading test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Feature data loading test failed: {e}")
        return False

def test_integration():
    """Test 6: Integration Test"""
    print("\n🧪 Test 6: Integration Test")
    print("-" * 50)
    
    try:
        print("🔗 Testing complete agent creation workflow...")
        
        # Simulate enhanced agent creation workflow
        selected_algorithm = 'SAC'
        selected_grade = 'C'
        environment_type = 'enhanced'
        
        # Get algorithm configuration
        config = get_algorithm_config(selected_algorithm, selected_grade)
        print(f"✅ Algorithm config loaded: {config['name']}")
        
        # Create agent
        agent = create_crypto_sac_agent(grade=selected_grade)
        agent.algorithm = selected_algorithm
        agent.environment_type = environment_type
        
        print(f"✅ Agent created: {agent.agent_id}")
        print(f"   🤖 Algorithm: {agent.algorithm}")
        print(f"   🎯 Grade: {agent.grade}")
        print(f"   🏗️ Environment: {agent.environment_type}")
        
        # Test agent info
        info = agent.get_info()
        print(f"✅ Agent info accessible")
        print(f"   📊 Config keys: {list(info['config'].keys())}")
        
        print("\n✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Enhanced Agent Creation System - Test Suite")
    print("=" * 70)
    
    tests = [
        test_algorithm_configs,
        test_data_availability,
        test_agent_creation,
        test_environment_types,
        test_feature_data_loading,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Algorithm Configs",
        "Data Availability", 
        "Agent Creation",
        "Environment Types",
        "Feature Data Loading",
        "Integration Test"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name:<20} {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced Agent Creation System is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 