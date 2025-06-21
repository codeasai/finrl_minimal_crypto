# test_data_loader.py - Simple Test Script for Data Loader
"""
ไฟล์ทดสอบระบบ Data Loader อย่างง่าย

ทดสอบ:
1. การดาวน์โหลดข้อมูล single symbol
2. การดาวน์โหลดข้อมูล multiple symbols
3. การแสดงรายการไฟล์ที่มีอยู่
4. การแสดงสรุปข้อมูล
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from data_loader import YahooDataLoader, download_crypto_data, get_crypto_data_summary

def test_data_loader():
    """ทดสอบ Data Loader"""
    print("🧪 Testing Yahoo Finance Data Loader")
    print("=" * 50)
    
    # Test 1: Single symbol download
    print("\n📊 Test 1: Single Symbol Download")
    try:
        # Download BTC data for last 30 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"Downloading BTC-USD from {start_date} to {end_date}")
        btc_data = download_crypto_data('BTC-USD', start_date, end_date, '1d')
        
        print(f"✅ Success: Downloaded {len(btc_data)} rows")
        print(f"Columns: {list(btc_data.columns)}")
        print(f"Date range: {btc_data['timestamp'].min()} to {btc_data['timestamp'].max()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Multiple symbols download
    print("\n📊 Test 2: Multiple Symbols Download")
    try:
        symbols = ['ETH-USD', 'ADA-USD', 'DOT-USD']
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Downloading {symbols} from {start_date} to {end_date}")
        batch_data = download_crypto_data(symbols, start_date, end_date, '1d')
        
        print(f"✅ Success: Downloaded {len(batch_data)} symbols")
        for symbol, data in batch_data.items():
            print(f"   {symbol}: {len(data)} rows")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: List available data
    print("\n📋 Test 3: Available Data Files")
    try:
        loader = YahooDataLoader()
        available = loader.list_available_data()
        
        print(f"📁 Found {len(available)} data files:")
        for file_info in available:
            print(f"   {file_info['symbol']} ({file_info['interval']}) - {file_info['file_size_kb']} KB - {file_info['modified_time']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Data summary
    print("\n📈 Test 4: Data Summary")
    try:
        summary = get_crypto_data_summary()
        print(f"📊 Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   Symbols: {summary['symbols']}")
        print(f"   Intervals: {summary['intervals']}")
        print(f"   Total size: {summary['total_size_mb']} MB")
        print(f"   Raw directory: {summary['raw_directory']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Load existing data
    print("\n📥 Test 5: Load Existing Data")
    try:
        loader = YahooDataLoader()
        
        # Try to load BTC data if it exists
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        existing_data = loader.load_symbol('BTC-USD', start_date, end_date, '1d')
        
        if existing_data is not None:
            print(f"✅ Loaded existing BTC data: {len(existing_data)} rows")
            print(f"Price range: ${existing_data['close'].min():.2f} - ${existing_data['close'].max():.2f}")
        else:
            print("ℹ️  No existing BTC data found for this date range")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Data Loader Testing Completed!")

if __name__ == "__main__":
    test_data_loader() 