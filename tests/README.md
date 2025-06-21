# Tests Directory

This directory contains all test files for the finrl_minimal_crypto project.

## Test Files

### Core System Tests
- **`test_enhanced_agent_creation.py`** - Enhanced agent creation system tests
  - Algorithm configuration testing
  - Data availability checks
  - Agent creation with different algorithms and grades
  - Environment type testing
  - Feature data loading
  - Integration testing

### Data Pipeline Tests
- **`test_data_feature.py`** - Feature engineering system tests
  - 151 technical indicators testing
  - Feature data processing
  - Data validation and quality checks
  - File operations and naming conventions

- **`test_data_loader.py`** - Data loading system tests
  - Yahoo Finance data download
  - Multiple symbols and timeframes
  - File naming and storage
  - Data integrity checks

### Algorithm Tests
- **`test_rl_agent_configs.py`** - RL agent configuration tests
  - Grade-based configuration testing
  - Parameter validation
  - Configuration recommendations

- **`test_sac_results.py`** - SAC results analysis tests
  - Interactive SAC agent browser
  - Performance metrics analysis
  - Agent comparison utilities

### Environment Tests
- **`test_enhanced_environment.py`** - Enhanced environment tests
  - Environment creation and validation
  - Action and observation space testing
  - Reward function testing

- **`test_enhanced_vs_original.py`** - Environment comparison tests
  - Basic vs Enhanced environment comparison
  - Performance benchmarking
  - Feature comparison

### Advanced System Tests
- **`test_enhanced_sac_system.py`** - Enhanced SAC system tests
  - Metadata management testing
  - Enhanced trainer testing
  - Integration with grade system

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v

# Or run individual test files
python tests/test_enhanced_agent_creation.py
```

### Run Specific Test Categories
```bash
# Data pipeline tests
python tests/test_data_feature.py
python tests/test_data_loader.py

# Algorithm tests
python tests/test_rl_agent_configs.py
python tests/test_sac_results.py

# Environment tests
python tests/test_enhanced_environment.py
python tests/test_enhanced_vs_original.py

# System integration tests
python tests/test_enhanced_agent_creation.py
python tests/test_enhanced_sac_system.py
```

## Test Coverage

The test suite covers:
- ✅ Algorithm configuration system (5 algorithms, 6 grades)
- ✅ Data loading and feature engineering (151 features)
- ✅ Agent creation and management
- ✅ Environment types (Basic/Enhanced)
- ✅ Trading performance analysis
- ✅ System integration and workflows

## Test Results Example

```
🚀 Enhanced Agent Creation System - Test Suite
======================================================================
✅ Algorithm Configs     - Algorithm configuration system
✅ Data Availability     - Raw and feature data checks
✅ Agent Creation        - Multi-algorithm agent creation
✅ Environment Types     - Basic vs Enhanced environments
✅ Feature Data Loading  - 151 features loading
✅ Integration Test      - Complete workflow testing

🎯 Overall Result: 6/6 tests passed
🎉 All tests passed! System is working correctly.
```

## Adding New Tests

When adding new functionality, create corresponding test files following the naming convention:
- `test_<module_name>.py` for module-specific tests
- Include comprehensive test cases covering normal and edge cases
- Add integration tests for new workflows
- Update this README with new test descriptions 