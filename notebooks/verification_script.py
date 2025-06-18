# verify_system.py
# System verification script for Crypto Trading RL Agent

import os
import sys
import importlib
import subprocess
from datetime import datetime

def check_dependencies():
    """
    ตรวจสอบ dependencies ที่จำเป็น
    """
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'yfinance',
        'torch',
        'stable_baselines3',
        'finrl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All dependencies available")
        return True

def check_config():
    """
    ตรวจสอบไฟล์ config
    """
    print("\n🔍 Checking configuration...")
    
    try:
        import config
        config.validate_config()
        config.print_config_summary()
        return True
    except ImportError:
        print("❌ config.py not found")
        return False
    except Exception as e:
        print(f"❌ Config error: {str(e)}")
        return False

def check_directories():
    """
    ตรวจสอบและสร้างโฟลเดอร์ที่จำเป็น
    """
    print("\n🔍 Checking directories...")
    
    required_dirs = [
        'data',
        'processed_data', 
        'models',
        'agents',
        'evaluation',
        'reports',
        'trading',
        'backtest',
        'paper_trading',
        'live_trading',
        'logs',
        'tensorboard_logs'
    ]
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  📁 Created: {dir_name}")
        else:
            print(f"  ✅ Exists: {dir_name}")
    
    return True

def check_notebook_files():
    """
    ตรวจสอบไฟล์ notebook
    """
    print("\n🔍 Checking notebook files...")
    
    required_notebooks = [
        '1_data_preparation.ipynb',
        '2_agent_creation.ipynb', 
        '3_agent_training.ipynb',
        '4_agent_evaluation.ipynb',
        '5_trading_implementation.ipynb'
    ]
    
    all_exist = True
    
    for notebook in required_notebooks:
        if os.path.exists(notebook):
            print(f"  ✅ {notebook}")
        else:
            print(f"  ❌ {notebook}")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """
    ทดสอบการทำงานพื้นฐาน
    """
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test data download
        print("  📊 Testing data download...")
        import yfinance as yf
        
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period="5d")
        
        if len(data) > 0:
            print("    ✅ Data download working")
        else:
            print("    ❌ No data retrieved")
            return False
        
        # Test FinRL import
        print("  🤖 Testing FinRL...")
        from finrl.agents.stablebaselines3.models import DRLAgent
        print("    ✅ FinRL import successful")
        
        # Test model import
        print("  🧠 Testing model imports...")
        from stable_baselines3 import PPO, A2C
        print("    ✅ Stable Baselines3 import successful")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return False

def create_sample_data():
    """
    สร้างข้อมูลตัวอย่างสำหรับทดสอบ
    """
    print("\n📊 Creating sample data...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # สร้างข้อมูลตัวอย่าง
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        
        sample_data = []
        for symbol in ['BTC-USD', 'ETH-USD']:
            for date in dates:
                sample_data.append({
                    'timestamp': date,
                    'tic': symbol,
                    'open': np.random.uniform(40000, 50000) if symbol == 'BTC-USD' else np.random.uniform(2500, 3500),
                    'high': np.random.uniform(40000, 50000) if symbol == 'BTC-USD' else np.random.uniform(2500, 3500),
                    'low': np.random.uniform(40000, 50000) if symbol == 'BTC-USD' else np.random.uniform(2500, 3500),
                    'close': np.random.uniform(40000, 50000) if symbol == 'BTC-USD' else np.random.uniform(2500, 3500),
                    'volume': np.random.uniform(1000000, 5000000)
                })
        
        df = pd.DataFrame(sample_data)
        
        # บันทึกไฟล์ตัวอย่าง
        sample_file = os.path.join('data', 'sample_data.csv')
        df.to_csv(sample_file, index=False)
        
        print(f"  ✅ Sample data created: {sample_file}")
        print(f"  📊 {len(df)} rows, {len(df['tic'].unique())} symbols")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating sample data: {str(e)}")
        return False

def run_system_verification():
    """
    รันการตรวจสอบระบบทั้งหมด
    """
    print("🚀 Starting system verification...")
    print("=" * 60)
    
    results = []
    
    # ตรวจสอบ dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # ตรวจสอบ config
    results.append(("Configuration", check_config()))
    
    # ตรวจสอบ directories
    results.append(("Directories", check_directories()))
    
    # ตรวจสอบ notebook files
    results.append(("Notebook Files", check_notebook_files()))
    
    # ทดสอบการทำงานพื้นฐาน
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # สร้างข้อมูลตัวอย่าง
    results.append(("Sample Data", create_sample_data()))
    
    # สรุปผลการตรวจสอบ
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\n✅ System is ready for use")
        print("\n📚 Next steps:")
        print("1. Run 1_data_preparation.ipynb")
        print("2. Run 2_agent_creation.ipynb") 
        print("3. Run 3_agent_training.ipynb")
        print("4. Run 4_agent_evaluation.ipynb")
        print("5. Run 5_trading_implementation.ipynb")
        
        print("\n⚠️ Important reminders:")
        print("- Always test with paper trading first")
        print("- Start with small amounts for live trading")
        print("- Monitor performance continuously")
        print("- Use proper risk management")
        
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\n💡 Please fix the issues before proceeding")
        print("- Install missing dependencies")
        print("- Fix configuration errors")
        print("- Ensure all files are present")
    
    return all_passed

def create_quick_start_guide():
    """
    สร้างคู่มือเริ่มต้นใช้งาน
    """
    guide = """
# 🚀 Crypto Trading RL Agent - Quick Start Guide

## 📋 Prerequisites
1. Python 3.8+ installed
2. Required packages installed (run verify_system.py to check)
3. Stable internet connection for data download

## 🔧 Setup Steps

### Step 1: System Verification
```bash
python verify_system.py
```

### Step 2: Data Preparation
```bash
jupyter notebook 1_data_preparation.ipynb
```
- Downloads crypto price data
- Adds technical indicators
- Normalizes data for ML

### Step 3: Agent Creation
```bash
jupyter notebook 2_agent_creation.ipynb
```
- Creates trading environment
- Sets up RL agent architecture
- Configures hyperparameters

### Step 4: Agent Training
```bash
jupyter notebook 3_agent_training.ipynb
```
- Trains the RL agent
- Validates performance
- Saves trained model

### Step 5: Agent Evaluation
```bash
jupyter notebook 4_agent_evaluation.ipynb
```
- Evaluates trained model
- Compares with baselines
- Generates performance reports

### Step 6: Trading Implementation
```bash
jupyter notebook 5_trading_implementation.ipynb
```
- Backtesting framework
- Paper trading simulation
- Live trading preparation

## ⚠️ Important Safety Notes

### Before Live Trading:
1. **Always start with paper trading**
2. **Test thoroughly with small amounts**
3. **Set proper stop-losses**
4. **Monitor performance continuously**
5. **Never risk more than you can afford to lose**

### Risk Management:
- Maximum 2% of portfolio per trade
- Daily loss limit: 2% of portfolio
- Position size limit: 20% per asset
- Stop loss: 5% per position

## 📊 Expected Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| Setup & Data Prep | 1-2 hours | Get system ready |
| Agent Training | 2-4 hours | Train RL model |
| Evaluation | 1 hour | Analyze performance |
| Paper Trading | 1-4 weeks | Test in simulation |
| Live Trading | Ongoing | Real trading (optional) |

## 🔧 Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
pip install finrl stable-baselines3 yfinance pandas numpy matplotlib
```

**2. CUDA/GPU Issues**
- System will automatically fall back to CPU
- GPU is recommended but not required

**3. Data Download Failures**
- Check internet connection
- Try different date ranges
- Some symbols may be unavailable

**4. Training Convergence Issues**
- Reduce learning rate
- Increase training timesteps
- Try different model architectures

## 📚 Additional Resources

- [FinRL Documentation](https://finrl.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Cryptocurrency Trading Best Practices](https://academy.binance.com/)

## 🆘 Support

If you encounter issues:
1. Check the verification script output
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Check file permissions and paths

## 📈 Performance Expectations

**Realistic Expectations:**
- RL agents may not always outperform buy-and-hold
- Performance varies significantly with market conditions
- Backtesting results don't guarantee future performance
- Start with conservative position sizes

**Success Metrics:**
- Positive Sharpe ratio (>0.5 good, >1.0 excellent)
- Maximum drawdown <10%
- Consistent performance across different market conditions

---

**⚠️ DISCLAIMER:** This is for educational purposes only. 
Cryptocurrency trading involves significant risk. Past performance 
does not guarantee future results. Always do your own research 
and consider consulting with financial advisors.
"""
    
    guide_file = "QUICK_START_GUIDE.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"📚 Quick start guide created: {guide_file}")
    return guide_file

def generate_system_report():
    """
    สร้างรายงานสถานะระบบ
    """
    print("\n📊 Generating system report...")
    
    import platform
    import psutil
    
    report = []
    report.append("# System Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # System information
    report.append("## System Information")
    report.append(f"- OS: {platform.system()} {platform.release()}")
    report.append(f"- Python: {platform.python_version()}")
    report.append(f"- CPU Cores: {psutil.cpu_count()}")
    report.append(f"- RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    report.append("")
    
    # Package versions
    report.append("## Package Versions")
    
    packages_to_check = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'yfinance', 'torch', 'stable_baselines3'
    ]
    
    for package in packages_to_check:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            report.append(f"- {package}: {version}")
        except ImportError:
            report.append(f"- {package}: Not installed")
    
    report.append("")
    
    # GPU information
    report.append("## GPU Information")
    try:
        import torch
        if torch.cuda.is_available():
            report.append(f"- CUDA Available: Yes")
            report.append(f"- GPU Count: {torch.cuda.device_count()}")
            report.append(f"- GPU Name: {torch.cuda.get_device_name(0)}")
            report.append(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            report.append("- CUDA Available: No")
    except:
        report.append("- GPU Check: Failed")
    
    report.append("")
    
    # Directory structure
    report.append("## Directory Structure")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        report.append(f"{indent}{os.path.basename(root)}/")
        
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Limit to first 5 files per directory
            if file.endswith(('.py', '.ipynb', '.md', '.txt', '.json')):
                report.append(f"{subindent}{file}")
        
        if len(files) > 5:
            report.append(f"{subindent}... and {len(files) - 5} more files")
    
    # Write report
    report_content = "\n".join(report)
    report_file = "system_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📊 System report saved: {report_file}")
    return report_file

if __name__ == "__main__":
    """
    Main execution
    """
    print("🔍 Crypto Trading RL Agent - System Verification")
    print("=" * 60)
    
    # Run verification
    success = run_system_verification()
    
    # Create additional resources
    create_quick_start_guide()
    generate_system_report()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 SYSTEM VERIFICATION COMPLETED SUCCESSFULLY!")
        print("\n📚 Resources created:")
        print("- QUICK_START_GUIDE.md")
        print("- system_report.md")
        
        print("\n🚀 You're ready to start!")
        print("Begin with: jupyter notebook 1_data_preparation.ipynb")
        
    else:
        print("❌ SYSTEM VERIFICATION FAILED")
        print("Please fix the issues before proceeding")
    
    print("\n💡 Tip: Run this script anytime to check system status")
    print("=" * 60)