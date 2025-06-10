import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def check_data_quality(df, symbol):
    """ตรวจสอบคุณภาพของข้อมูล"""
    print(f"\n🔍 ตรวจสอบข้อมูล {symbol}")
    print("-" * 50)
    
    # 1. ตรวจสอบข้อมูลพื้นฐาน
    print("\n📊 ข้อมูลพื้นฐาน:")
    print(f"จำนวนแถว: {len(df):,}")
    print(f"ช่วงเวลา: {df['timestamp'].min()} ถึง {df['timestamp'].max()}")
    print(f"ความถี่ข้อมูล: {df['timestamp'].diff().mode()[0]}")
    
    # 2. ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
    else:
        print("✅ มีคอลัมน์ที่จำเป็นครบถ้วน")
    
    # 3. ตรวจสอบค่า NaN
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        print(f"❌ พบค่า NaN ในคอลัมน์: {', '.join(nan_counts[nan_counts > 0].index)}")
    else:
        print("✅ ไม่พบค่า NaN")
    
    # 4. ตรวจสอบค่า inf
    inf_counts = df[required_columns].isin([np.inf, -np.inf]).sum()
    if inf_counts.any():
        print(f"❌ พบค่า inf ในคอลัมน์: {', '.join(inf_counts[inf_counts > 0].index)}")
    else:
        print("✅ ไม่พบค่า inf")
    
    # 5. ตรวจสอบช่วงราคา
    price_cols = ['open', 'high', 'low', 'close']
    price_issues = []
    for col in price_cols:
        if (df[col] <= 0).any():
            price_issues.append(f"{col}: พบ {len(df[df[col] <= 0])} ค่า 0 หรือติดลบ")
        if df[col].isna().any():
            price_issues.append(f"{col}: พบ {df[col].isna().sum()} ค่า NaN")
    
    if price_issues:
        print("❌ พบปัญหากับราคา:")
        for issue in price_issues:
            print(f"  - {issue}")
    else:
        print("✅ ราคาทั้งหมดถูกต้อง")
    
    # 6. ตรวจสอบความถูกต้องของราคา
    logic_issues = []
    if (df['high'] < df[['open', 'close', 'low']].max(axis=1)).any():
        logic_issues.append("high ต่ำกว่า open, close, หรือ low")
    if (df['low'] > df[['open', 'close', 'high']].min(axis=1)).any():
        logic_issues.append("low สูงกว่า open, close, หรือ high")
    
    if logic_issues:
        print("❌ พบความไม่ถูกต้องของราคา:")
        for issue in logic_issues:
            print(f"  - {issue}")
    else:
        print("✅ ความสัมพันธ์ของราคาถูกต้อง")
    
    # 7. ตรวจสอบ volume
    volume_issues = []
    if (df['volume'] < 0).any():
        volume_issues.append(f"พบ {len(df[df['volume'] < 0])} ค่า volume ติดลบ")
    if df['volume'].isna().any():
        volume_issues.append(f"พบ {df['volume'].isna().sum()} ค่า volume เป็น NaN")
    
    if volume_issues:
        print("❌ พบปัญหากับ volume:")
        for issue in volume_issues:
            print(f"  - {issue}")
    else:
        print("✅ volume ถูกต้อง")
    
    # 8. ตรวจสอบความต่อเนื่องของข้อมูล
    time_diff = df['timestamp'].diff()
    expected_diff = time_diff.mode()[0]
    gaps = time_diff[time_diff != expected_diff]
    
    if not gaps.empty:
        print(f"⚠️ พบช่องว่างในข้อมูล {len(gaps)} จุด")
        print("ตัวอย่างช่องว่าง:")
        for i, (idx, diff) in enumerate(gaps.head(3).items()):
            print(f"  - {df.loc[idx-1, 'timestamp']} ถึง {df.loc[idx, 'timestamp']} (ห่าง {diff})")
    else:
        print("✅ ข้อมูลต่อเนื่อง")
    
    # 9. สรุปสถิติพื้นฐาน
    print("\n📈 สถิติพื้นฐาน:")
    print(df[price_cols + ['volume']].describe())
    
    return {
        'rows': len(df),
        'start_date': df['timestamp'].min(),
        'end_date': df['timestamp'].max(),
        'frequency': df['timestamp'].diff().mode()[0],
        'has_issues': bool(price_issues or logic_issues or volume_issues)
    }

def main():
    # ตรวจสอบข้อมูลทั้ง 3 ไฟล์
    files = [
        'data/BTC_USD-1d-20230601-20250609.csv',
        'data/BTC_USDT-5m-20250511-20250610.csv',
        'data/ADA_USDT-1d-20250511-20250610.csv'
    ]
    
    results = {}
    for file in files:
        try:
            # อ่านข้อมูล
            df = pd.read_csv(file)
            
            # แปลง timestamp เป็น datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # ตรวจสอบข้อมูล
            symbol = file.split('/')[-1].split('-')[0]
            results[symbol] = check_data_quality(df, symbol)
            
        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาดในการตรวจสอบ {file}: {str(e)}")
    
    # สรุปผลการตรวจสอบ
    print("\n📋 สรุปผลการตรวจสอบ")
    print("-" * 50)
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  - จำนวนข้อมูล: {result['rows']:,} แถว")
        print(f"  - ช่วงเวลา: {result['start_date']} ถึง {result['end_date']}")
        print(f"  - ความถี่ข้อมูล: {result['frequency']}")
        print(f"  - มีปัญหา: {'❌' if result['has_issues'] else '✅'}")

if __name__ == "__main__":
    main() 