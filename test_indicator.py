import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CRYPTO_SYMBOLS

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ technical indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        ข้อมูลราคาที่มีคอลัมน์ ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
    --------
    pd.DataFrame
        ข้อมูลที่มี technical indicators เพิ่มเติม
    """
    # สร้างสำเนาข้อมูล
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # 5. Volume Indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    return df

def plot_indicators(df: pd.DataFrame, symbol: str):
    """
    สร้างกราฟแสดง indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        ข้อมูลที่มี indicators
    symbol : str
        ชื่อสัญลักษณ์คริปโต
    """
    # สร้าง subplot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. Price and Moving Averages
    ax1.plot(df.index, df['Close'], label='Price', color='blue')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='red')
    ax1.plot(df.index, df['EMA_20'], label='EMA 20', color='green')
    ax1.plot(df.index, df['BB_Upper'], label='BB Upper', color='gray', linestyle='--')
    ax1.plot(df.index, df['BB_Lower'], label='BB Lower', color='gray', linestyle='--')
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_title(f'{symbol} Price and Indicators')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # 2. MACD
    ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax2.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
    ax2.bar(df.index, df['MACD_Hist'], label='Histogram', color='green', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # 3. RSI
    ax3.plot(df.index, df['RSI_14'], label='RSI 14', color='purple')
    ax3.axhline(y=70, color='red', linestyle='--')
    ax3.axhline(y=30, color='green', linestyle='--')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_indicators(
    symbol: str = "BTC-USD",
    start_date: str = None,
    end_date: str = None,
    days: int = 100
):
    """
    ทดสอบการคำนวณและแสดงผล indicators
    
    Parameters:
    -----------
    symbol : str
        ชื่อสัญลักษณ์คริปโต
    start_date : str
        วันที่เริ่มต้น (YYYY-MM-DD)
    end_date : str
        วันที่สิ้นสุด (YYYY-MM-DD)
    days : int
        จำนวนวันย้อนหลัง (ถ้าไม่ระบุ start_date)
    """
    # กำหนดวันที่
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"🔍 Testing indicators for {symbol}")
    print(f"📅 Date range: {start_date} to {end_date}")
    
    # ดาวน์โหลดข้อมูล
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if len(df) == 0:
        print(f"❌ No data found for {symbol}")
        return
    
    # คำนวณ indicators
    df = calculate_indicators(df)
    
    # แสดงข้อมูลตัวอย่าง
    print("\n📊 Data Preview with Indicators:")
    print("="*80)
    print(df.tail())
    print("="*80)
    
    # แสดงข้อมูลสถิติ
    print("\n📈 Indicator Statistics:")
    print("="*80)
    print(df.describe())
    print("="*80)
    
    # สร้างกราฟ
    plot_indicators(df, symbol)

if __name__ == "__main__":
    # ทดสอบกับ BTC-USD
    test_indicators(
        symbol="BTC-USD",
        days=100  # 100 วันย้อนหลัง
    ) 