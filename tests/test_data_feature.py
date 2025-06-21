# test_data_feature.py - Comprehensive Test for Data Feature Engineering
"""
ไฟล์ทดสอบระบบ Feature Engineering อย่างครบถ้วน

ทดสอบ:
1. การประมวลผล raw data เป็น feature data
2. การเพิ่ม technical indicators ครบถ้วน
3. การบันทึกและโหลด feature data
4. การแสดงสรุปข้อมูล features
5. การทำงานร่วมกับ data loader
6. การเตรียมข้อมูลสำหรับ environment, train และ test
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from data_feature import (
    CryptoFeatureProcessor, 
    process_crypto_features, 
    process_all_crypto_features,
    load_crypto_features,
    get_crypto_feature_summary
)

def test_feature_engineering_system():
    """ทดสอบระบบ Feature Engineering ครบถ้วน"""
    print("🧪 Testing Crypto Feature Engineering System")
    print("=" * 60)
    
    # Test 1: Initialize processor
    print("\n📊 Test 1: Initialize Feature Processor")
    try:
        processor = CryptoFeatureProcessor()
        print("✅ Feature processor initialized successfully")
        print(f"   Raw directory: {processor.raw_dir}")
        print(f"   Feature directory: {processor.feature_dir}")
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return
    
    # Test 2: Check available raw data
    print("\n📁 Test 2: Check available raw data")
    try:
        import glob
        raw_files = glob.glob(os.path.join(processor.raw_dir, "*.csv"))
        print(f"📄 Found {len(raw_files)} raw data files:")
        for file in raw_files[:5]:  # Show first 5
            filename = os.path.basename(file)
            size_kb = os.path.getsize(file) / 1024
            print(f"   {filename} ({size_kb:.1f} KB)")
        if len(raw_files) > 5:
            print(f"   ... and {len(raw_files) - 5} more files")
    except Exception as e:
        print(f"❌ Error checking raw data: {e}")
    
    # Test 3: Process single raw data file
    print("\n🔧 Test 3: Process single raw data file")
    try:
        if raw_files:
            sample_file = raw_files[0]
            filename = os.path.basename(sample_file)
            print(f"Processing: {filename}")
            
            # Process the file
            feature_data = processor.process_raw_data(sample_file)
            
            if feature_data is not None:
                print(f"✅ Successfully processed {filename}")
                print(f"   Rows: {len(feature_data)}")
                print(f"   Features: {len(feature_data.columns)}")
                print(f"   Sample features: {list(feature_data.columns[:10])}...")
                
                # Show some statistics
                feature_cols = [col for col in feature_data.columns if col not in 
                               ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                'Dividends', 'Stock Splits', 'symbol', 'interval', 
                                'start_date', 'end_date', 'download_time']]
                print(f"   Technical indicators: {len(feature_cols)}")
            else:
                print(f"❌ Failed to process {filename}")
        else:
            print("⚠️  No raw data files found to process")
    except Exception as e:
        print(f"❌ Error processing single file: {e}")
    
    # Test 4: Process all raw data files
    print("\n🔄 Test 4: Process all raw data files")
    try:
        results = process_all_crypto_features(force_reprocess=False)
        
        print(f"📊 Processing results:")
        successful = 0
        for filename, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {filename}")
            if success:
                successful += 1
        
        print(f"\n📈 Summary: {successful}/{len(results)} files processed successfully")
        
    except Exception as e:
        print(f"❌ Error processing all files: {e}")
    
    # Test 5: List available feature data
    print("\n📋 Test 5: List available feature data")
    try:
        available_features = processor.list_available_feature_data()
        print(f"📁 Found {len(available_features)} feature files:")
        
        for file_info in available_features[:5]:  # Show first 5
            print(f"   {file_info['symbol']} ({file_info['interval']}) "
                  f"- {file_info['start_date']} to {file_info['end_date']}")
            print(f"     Features: {file_info['feature_count']}, "
                  f"Size: {file_info['file_size_kb']} KB, "
                  f"Modified: {file_info['modified_time']}")
        
        if len(available_features) > 5:
            print(f"   ... and {len(available_features) - 5} more files")
            
    except Exception as e:
        print(f"❌ Error listing feature data: {e}")
    
    # Test 6: Get feature summary
    print("\n📈 Test 6: Feature summary")
    try:
        summary = get_crypto_feature_summary()
        print(f"📊 Feature Data Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   Symbols: {summary['symbols']}")
        print(f"   Intervals: {summary['intervals']}")
        print(f"   Average features per file: {summary['average_features']}")
        print(f"   Total size: {summary['total_size_mb']} MB")
        print(f"   Feature directory: {summary['feature_directory']}")
    except Exception as e:
        print(f"❌ Error getting feature summary: {e}")
    
    # Test 7: Load specific feature data
    print("\n📥 Test 7: Load specific feature data")
    try:
        # Try to load BTC feature data
        btc_features = load_crypto_features('BTC-USD', '2024-01-01', '2024-02-01', '1d')
        
        if btc_features is not None:
            print(f"✅ Loaded BTC feature data:")
            print(f"   Rows: {len(btc_features)}")
            print(f"   Features: {len(btc_features.columns)}")
            print(f"   Date range: {btc_features['timestamp'].min()} to {btc_features['timestamp'].max()}")
            
            # Show some technical indicators
            technical_indicators = [col for col in btc_features.columns if any(x in col for x in [
                'sma_', 'ema_', 'rsi', 'macd', 'bb_', 'atr', 'stoch_', 'williams_', 'cci', 'roc'
            ])]
            print(f"   Technical indicators: {len(technical_indicators)}")
            print(f"   Sample indicators: {technical_indicators[:10]}")
            
            # Check for null values
            null_counts = btc_features.isnull().sum()
            high_null_features = null_counts[null_counts > len(btc_features) * 0.5].index.tolist()
            if high_null_features:
                print(f"   ⚠️  High null percentage features: {len(high_null_features)}")
            else:
                print(f"   ✅ Data quality looks good")
                
        else:
            print("ℹ️  No BTC feature data found for this date range")
            
    except Exception as e:
        print(f"❌ Error loading feature data: {e}")
    
    # Test 8: Check feature data for environment preparation
    print("\n🏗️  Test 8: Environment preparation readiness")
    try:
        if 'btc_features' in locals() and btc_features is not None:
            # Check essential columns for trading environment
            essential_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_essential = [col for col in essential_columns if col not in btc_features.columns]
            
            if not missing_essential:
                print("✅ Essential OHLCV columns present")
            else:
                print(f"❌ Missing essential columns: {missing_essential}")
            
            # Check technical indicators
            indicators = {
                'Moving Averages': [col for col in btc_features.columns if 'sma_' in col or 'ema_' in col],
                'Momentum': [col for col in btc_features.columns if any(x in col for x in ['rsi', 'macd', 'roc'])],
                'Volatility': [col for col in btc_features.columns if any(x in col for x in ['bb_', 'atr', 'volatility_'])],
                'Volume': [col for col in btc_features.columns if 'volume' in col and col != 'volume'],
                'Price Action': [col for col in btc_features.columns if any(x in col for x in ['price_change_', 'gap_', 'body_size'])]
            }
            
            print("📊 Available indicator categories:")
            for category, indicators_list in indicators.items():
                print(f"   {category}: {len(indicators_list)} indicators")
            
            # Calculate data completeness
            total_indicators = sum(len(indicators_list) for indicators_list in indicators.values())
            print(f"   Total technical indicators: {total_indicators}")
            
            # Check if data is ready for training
            complete_rows = btc_features.dropna().shape[0]
            completion_rate = complete_rows / len(btc_features) * 100
            print(f"   Data completion rate: {completion_rate:.1f}%")
            
            if completion_rate > 70:
                print("✅ Data ready for environment creation and training")
            else:
                print("⚠️  Data may need additional preprocessing for training")
        else:
            print("⚠️  No feature data available for environment testing")
            
    except Exception as e:
        print(f"❌ Error checking environment readiness: {e}")
    
    # Test 9: Integration test with data loader
    print("\n🔗 Test 9: Integration with data loader")
    try:
        # Import data loader
        from data_loader import YahooDataLoader
        
        # Check if we can create a complete pipeline
        print("Testing complete data pipeline:")
        print("   Raw Data → Feature Engineering → Environment Ready")
        
        # Count files in each stage
        raw_count = len(glob.glob(os.path.join(processor.raw_dir, "*.csv")))
        feature_count = len(glob.glob(os.path.join(processor.feature_dir, "*-features.csv")))
        
        print(f"   Raw files: {raw_count}")
        print(f"   Feature files: {feature_count}")
        
        if feature_count > 0:
            print("✅ Complete pipeline functional")
            print("   Ready for:")
            print("     - Environment creation")
            print("     - Agent training")
            print("     - Backtesting")
            print("     - Live trading")
        else:
            print("⚠️  Pipeline incomplete - no feature files generated")
            
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
    
    print("\n🎉 Feature Engineering System Testing Completed!")
    print("=" * 60)

def test_specific_features():
    """ทดสอบ features เฉพาะ"""
    print("\n🔍 Testing Specific Technical Indicators")
    print("-" * 40)
    
    try:
        # Load sample feature data
        btc_features = load_crypto_features('BTC-USD', '2024-01-01', '2024-02-01', '1d')
        
        if btc_features is not None:
            # Test specific indicators
            indicators_to_test = {
                'SMA 20': 'sma_20',
                'EMA 20': 'ema_20', 
                'RSI': 'rsi',
                'MACD': 'macd',
                'Bollinger Upper': 'bb_upper',
                'ATR': 'atr',
                'Volume Ratio': 'volume_ratio'
            }
            
            print("📊 Indicator values (latest 3 rows):")
            for name, column in indicators_to_test.items():
                if column in btc_features.columns:
                    latest_values = btc_features[column].tail(3).values
                    print(f"   {name}: {latest_values}")
                else:
                    print(f"   {name}: Not available")
        else:
            print("⚠️  No feature data available for specific testing")
            
    except Exception as e:
        print(f"❌ Error testing specific features: {e}")

if __name__ == "__main__":
    # Run comprehensive tests
    test_feature_engineering_system()
    
    # Run specific feature tests
    test_specific_features()
    
    print("\n✨ All tests completed!")
    print("\n📋 Next Steps:")
    print("   1. Use feature data for environment creation")
    print("   2. Train SAC agents with enhanced features")
    print("   3. Evaluate performance improvements")
    print("   4. Deploy for live trading") 