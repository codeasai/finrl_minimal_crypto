#!/usr/bin/env python3
"""
Test GPU Requirement System

ทดสอบระบบ GPU requirement ที่เพิ่มเข้าไปใน:
1. config.py - GPU detection และ device configuration
2. main.py - GPU check ใน train_mode
3. crypto_agent.py - GPU check ใน train method
4. interactive_cli.py - GPU check ใน train workflows

Usage:
    python test_gpu_requirement.py
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestGPURequirement(unittest.TestCase):
    """Test GPU requirement system"""
    
    def setUp(self):
        """Setup test environment"""
        print(f"\n{'='*50}")
        print(f"Testing GPU Requirement System")
        print(f"{'='*50}")
    
    def test_config_gpu_detection(self):
        """Test GPU detection in config.py"""
        print("\n🔍 Test 1: Config GPU Detection")
        print("-" * 30)
        
        try:
            # Mock torch.cuda for testing
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch.cuda.device_count', return_value=0):
                    from config.config import check_gpu_availability, get_device_config
                    
                    # Test GPU check
                    available, message = check_gpu_availability()
                    print(f"   GPU Available: {available}")
                    print(f"   Message: {message}")
                    
                    self.assertFalse(available)
                    self.assertIn("CUDA is not available", message)
                    
                    # Test device config
                    device, gpu_available = get_device_config()
                    print(f"   Device: {device}")
                    print(f"   GPU Available: {gpu_available}")
                    
                    self.assertFalse(gpu_available)
                    
            print("   ✅ Config GPU detection working correctly")
            
        except Exception as e:
            print(f"   ❌ Config GPU detection failed: {e}")
            self.fail(f"Config GPU detection failed: {e}")
    
    def test_main_train_mode_gpu_check(self):
        """Test GPU check in main.py train_mode"""
        print("\n🔍 Test 2: Main Train Mode GPU Check")
        print("-" * 30)
        
        try:
            # Mock config to simulate no GPU
            with patch('config.config.GPU_AVAILABLE', False):
                with patch('config.config.DEVICE', None):
                    from main import train_mode
                    
                    # Test train mode with no GPU
                    result = train_mode(
                        grade='N',
                        algorithm='SAC',
                        environment='basic',
                        verbose=1
                    )
                    
                    print(f"   Train Result: {result}")
                    self.assertFalse(result, "Training should fail without GPU")
                    
            print("   ✅ Main train mode GPU check working correctly")
            
        except Exception as e:
            print(f"   ❌ Main train mode GPU check failed: {e}")
            self.fail(f"Main train mode GPU check failed: {e}")
    
    def test_crypto_agent_train_gpu_check(self):
        """Test GPU check in CryptoSACAgent.train"""
        print("\n🔍 Test 3: CryptoSACAgent Train GPU Check")
        print("-" * 30)
        
        try:
            # Mock config to simulate no GPU
            with patch('config.config.GPU_AVAILABLE', False):
                with patch('config.config.DEVICE', None):
                    from src.agents.crypto_agent import CryptoSACAgent
                    
                    # Create agent
                    agent = CryptoSACAgent(grade='N')
                    print(f"   Agent Created: {agent.agent_id}")
                    
                    # Try to train without GPU (should fail)
                    with self.assertRaises(RuntimeError) as context:
                        agent.train(verbose=1)
                    
                    error_message = str(context.exception)
                    print(f"   Expected Error: {error_message}")
                    self.assertIn("GPU required", error_message)
                    
            print("   ✅ CryptoSACAgent train GPU check working correctly")
            
        except Exception as e:
            print(f"   ❌ CryptoSACAgent train GPU check failed: {e}")
            self.fail(f"CryptoSACAgent train GPU check failed: {e}")
    
    def test_interactive_cli_gpu_check(self):
        """Test GPU check in interactive CLI workflows"""
        print("\n🔍 Test 4: Interactive CLI GPU Check")
        print("-" * 30)
        
        try:
            # Mock config to simulate no GPU
            with patch('config.config.GPU_AVAILABLE', False):
                with patch('config.config.DEVICE', None):
                    from src.cli.interactive_cli import InteractiveCLI, AgentManager
                    
                    # Create CLI instance
                    cli = InteractiveCLI()
                    print(f"   CLI Created: {type(cli).__name__}")
                    
                    # Mock agent for testing
                    mock_agent = MagicMock()
                    mock_agent.agent_id = "test_agent"
                    mock_agent.grade = "N"
                    mock_agent.config = {'total_timesteps': 50000}
                    mock_agent.env = MagicMock()  # Mock environment
                    
                    cli.agent_manager.current_agent = mock_agent
                    cli.current_data = MagicMock()  # Mock data
                    
                    # Test train workflow (should detect no GPU and return early)
                    with patch('builtins.input', return_value=''):
                        with patch('builtins.print') as mock_print:
                            cli.train_agent_workflow()
                            
                            # Check if GPU error message was printed
                            print_calls = [str(call) for call in mock_print.call_args_list]
                            gpu_error_found = any("GPU is required" in call for call in print_calls)
                            
                            print(f"   GPU Error Message Found: {gpu_error_found}")
                            self.assertTrue(gpu_error_found, "GPU error message should be displayed")
                    
            print("   ✅ Interactive CLI GPU check working correctly")
            
        except Exception as e:
            print(f"   ❌ Interactive CLI GPU check failed: {e}")
            self.fail(f"Interactive CLI GPU check failed: {e}")
    
    def test_info_mode_gpu_status(self):
        """Test GPU status display in info mode"""
        print("\n🔍 Test 5: Info Mode GPU Status")
        print("-" * 30)
        
        try:
            # Mock config to simulate no GPU
            with patch('config.config.GPU_AVAILABLE', False):
                with patch('config.config.DEVICE', None):
                    with patch('torch.cuda.is_available', return_value=False):
                        from main import info_mode
                        
                        # Test info mode
                        with patch('builtins.print') as mock_print:
                            result = info_mode(verbose=1)
                            
                            print(f"   Info Mode Result: {result}")
                            self.assertTrue(result, "Info mode should complete successfully")
                            
                            # Check if GPU status is displayed correctly
                            print_calls = [str(call) for call in mock_print.call_args_list]
                            training_ready_found = any("Training Ready: ❌ NO" in call for call in print_calls)
                            
                            print(f"   Training Ready Status Found: {training_ready_found}")
                            self.assertTrue(training_ready_found, "Training ready status should be displayed")
            
            print("   ✅ Info mode GPU status working correctly")
            
        except Exception as e:
            print(f"   ❌ Info mode GPU status failed: {e}")
            self.fail(f"Info mode GPU status failed: {e}")
    
    def test_gpu_available_scenario(self):
        """Test scenario when GPU is available"""
        print("\n🔍 Test 6: GPU Available Scenario")
        print("-" * 30)
        
        try:
            # Mock torch.cuda to simulate GPU available
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.device_count', return_value=1):
                    with patch('torch.cuda.get_device_name', return_value="Test GPU"):
                        with patch('torch.cuda.get_device_properties') as mock_props:
                            # Mock GPU properties
                            mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                            
                            # Mock torch.zeros and tensor operations
                            with patch('torch.zeros') as mock_zeros:
                                with patch('torch.cuda.empty_cache'):
                                    mock_tensor = MagicMock()
                                    mock_zeros.return_value = mock_tensor
                                    
                                    from config.config import check_gpu_availability, get_device_config
                                    
                                    # Test GPU check with GPU available
                                    available, message = check_gpu_availability()
                                    print(f"   GPU Available: {available}")
                                    print(f"   Message: {message}")
                                    
                                    self.assertTrue(available)
                                    self.assertIn("GPU available", message)
                                    
                                    # Test device config with GPU available
                                    device, gpu_available = get_device_config()
                                    print(f"   Device: {device}")
                                    print(f"   GPU Available: {gpu_available}")
                                    
                                    self.assertTrue(gpu_available)
                                    self.assertEqual(device, 'cuda')
            
            print("   ✅ GPU available scenario working correctly")
            
        except Exception as e:
            print(f"   ❌ GPU available scenario failed: {e}")
            self.fail(f"GPU available scenario failed: {e}")

def run_tests():
    """Run all GPU requirement tests"""
    print("🧪 GPU Requirement Test Suite")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGPURequirement)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Results Summary")
    print(f"{'='*60}")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback}")
    
    if result.errors:
        print(f"\n⚠️ Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n🎉 All tests passed! GPU requirement system is working correctly.")
        print(f"✅ Training will be properly blocked when GPU is not available")
        print(f"✅ All components check GPU requirements before training")
        print(f"✅ User-friendly error messages are displayed")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 