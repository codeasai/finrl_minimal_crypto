# tests/__init__.py - Test Package
"""
Test package for finrl_minimal_crypto

Contains all test files:
- test_enhanced_agent_creation.py: Enhanced agent creation system tests
- test_data_feature.py: Feature engineering tests
- test_data_loader.py: Data loading tests
- test_rl_agent_configs.py: RL agent configuration tests
- test_sac_results.py: SAC results analysis tests
- test_enhanced_environment.py: Enhanced environment tests
- test_enhanced_vs_original.py: Environment comparison tests
- test_enhanced_sac_system.py: Enhanced SAC system tests
"""

# Test discovery helper
def discover_tests():
    """Discover all test modules in this package"""
    import os
    import glob
    
    test_dir = os.path.dirname(__file__)
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    
    test_modules = []
    for test_file in test_files:
        module_name = os.path.basename(test_file)[:-3]  # Remove .py
        test_modules.append(module_name)
    
    return sorted(test_modules)

# Available test modules
TEST_MODULES = [
    'test_enhanced_agent_creation',
    'test_data_feature', 
    'test_data_loader',
    'test_rl_agent_configs',
    'test_sac_results',
    'test_enhanced_environment',
    'test_enhanced_vs_original',
    'test_enhanced_sac_system'
] 