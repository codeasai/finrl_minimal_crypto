import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CRYPTO_SYMBOLS

def load_test_data(
    start_date: str = None,
    end_date: str = None,
    num_rows: int = 10,
    symbols: list = None,
    time_interval: str = '1D'
) -> pd.DataFrame:
    """
    โหลดข้อมูลทดสอบจาก Yahoo Finance
    
    Parameters:
    -----------
    start_date : str, optional
        วันที่เริ่มต้นในรูปแบบ 'YYYY-MM-DD'
        ถ้าไม่ระบุจะใช้ 30 วันก่อนวันนี้
    end_date : str, optional
        วันที่สิ้นสุดในรูปแบบ 'YYYY-MM-DD'
        ถ้าไม่ระบุจะใช้วันนี้
    num_rows : int, optional
        จำนวนแถวที่ต้องการโหลด (default: 10)
    symbols : list, optional
        รายการสัญลักษณ์คริปโตที่ต้องการโหลด
        ถ้าไม่ระบุจะใช้จาก config.py
    time_interval : str, optional
        ช่วงเวลาของข้อมูล (default: '1D' = รายวัน)
    
    Returns:
    --------
    pd.DataFrame
        ข้อมูลที่โหลดมา
    """
    try:
        # กำหนดค่าเริ่มต้นถ้าไม่มีการระบุ
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if symbols is None:
            symbols = CRYPTO_SYMBOLS
        
        print(f"🔍 Loading test data:")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"📊 Symbols: {symbols}")
        print(f"📈 Number of rows: {num_rows}")
        
        # ดาวน์โหลดข้อมูลโดยตรงจาก yfinance
        df_list = []
        for symbol in symbols:
            print(f"📥 Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=time_interval,
                auto_adjust=True  # ใช้ auto_adjust แทน Adj Close
            )
            if len(df) == 0:
                print(f"⚠️ Warning: No data found for {symbol}")
                continue
                
            # เพิ่มคอลัมน์ที่จำเป็น
            df['tic'] = symbol
            df['timestamp'] = df.index
            df_list.append(df)
        
        if not df_list:
            raise ValueError(f"ไม่พบข้อมูลในช่วงวันที่ {start_date} ถึง {end_date}")
        
        # รวมข้อมูล
        df = pd.concat(df_list, axis=0)
        df = df.reset_index(drop=True)
        
        # เลือกจำนวนแถวที่ต้องการ
        df = df.head(num_rows)
        print(f"✅ Downloaded {len(df)} rows")
        
        # แสดงข้อมูลตัวอย่าง
        print("\n📊 Data Preview:")
        print("="*80)
        print(df)
        print("="*80)
        
        # แสดงข้อมูลสถิติพื้นฐาน
        print("\n📈 Basic Statistics:")
        print("="*80)
        print(df.describe())
        print("="*80)
        
        # แสดงข้อมูลเกี่ยวกับคอลัมน์
        print("\n📋 Column Information:")
        print("="*80)
        print(df.info())
        print("="*80)
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Tips:")
        print("1. ตรวจสอบว่าวันที่ที่ระบุถูกต้อง")
        print("2. ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
        print("3. ตรวจสอบว่าสัญลักษณ์คริปโตที่ระบุมีอยู่จริง")
        raise

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    print("🧪 Testing data loader...")
    
    # ทดสอบโหลดข้อมูล 5 แถวของ BTC-USD ในช่วง 7 วันที่ผ่านมา
    test_df = load_test_data(
        start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        num_rows=5,
        symbols=["BTC-USD"]
    ) 