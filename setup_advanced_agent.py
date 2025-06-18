#!/usr/bin/env python3
# setup_advanced_agent.py
"""
Setup Script สำหรับ Advanced Crypto Agent
รันไฟล์นี้เพื่อติดตั้ง dependencies ที่จำเป็นทั้งหมด
"""

import subprocess
import sys
import os

def install_package(package):
    """ติดตั้ง package โดยใช้ pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {package}: {e}")
        return False

def check_package(package_name):
    """ตรวจสอบว่า package ถูกติดตั้งแล้วหรือไม่"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("🚀 Setting up Advanced Crypto Agent")
    print("=" * 50)
    
    # Core packages ที่จำเป็น
    core_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "torch>=1.9.0",
        "scikit-learn>=1.0.0"
    ]
    
    # FinRL
    finrl_packages = ["finrl"]
    
    # Technical indicators packages
    ta_packages = ["pandas_ta>=0.3.14b"]
    
    print("\n📦 Installing core packages...")
    for package in core_packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"Installing {package_name}...")
        if not install_package(package):
            print(f"⚠️ Failed to install {package_name}, continuing...")
    
    print("\n🤖 Installing FinRL...")
    for package in finrl_packages:
        print(f"Installing {package}...")
        if not install_package(package):
            print(f"⚠️ Failed to install {package}")
    
    print("\n📈 Installing Technical Indicators...")
    ta_success = False
    
    # ลองติดตั้ง pandas_ta ก่อน
    print("Trying to install pandas_ta...")
    if install_package("pandas_ta"):
        ta_success = True
        print("✅ pandas_ta installed successfully")
    else:
        print("⚠️ pandas_ta installation failed")
        
        # ถ้า pandas_ta ไม่ได้ ลอง TA-Lib
        print("Trying to install TA-Lib...")
        if sys.platform.startswith('win'):
            # Windows
            if install_package("TA-Lib-Binary"):
                ta_success = True
                print("✅ TA-Lib-Binary installed successfully")
        else:
            # Linux/Mac
            if install_package("TA-Lib"):
                ta_success = True
                print("✅ TA-Lib installed successfully")
    
    if not ta_success:
        print("⚠️ No technical indicators library installed.")
        print("💡 The agent will use manual calculations instead.")
    
    print("\n🔍 Verifying installation...")
    
    # ตรวจสอบ core packages
    core_check = {
        "pandas": "pandas",
        "numpy": "numpy", 
        "matplotlib": "matplotlib",
        "yfinance": "yfinance",
        "torch": "torch",
        "sklearn": "scikit-learn"
    }
    
    for import_name, display_name in core_check.items():
        if check_package(import_name):
            print(f"✅ {display_name}")
        else:
            print(f"❌ {display_name} - ติดตั้งไม่สำเร็จ")
    
    # ตรวจสอบ FinRL
    if check_package("finrl"):
        print("✅ finrl")
    else:
        print("❌ finrl - ติดตั้งไม่สำเร็จ")
    
    # ตรวจสอบ technical indicators
    if check_package("pandas_ta"):
        print("✅ pandas_ta")
    elif check_package("talib"):
        print("✅ talib")
    else:
        print("⚠️ No technical indicators library - will use manual calculations")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. รันคำสั่ง: python advanced_crypto_agent.py")
    print("2. หรือดู README_advanced_agent.md สำหรับข้อมูลเพิ่มเติม")
    
    # สร้างโฟลเดอร์ที่จำเป็น
    os.makedirs("data", exist_ok=True)
    os.makedirs("advanced_models", exist_ok=True)
    print("\n📁 Created necessary directories")

if __name__ == "__main__":
    main() 