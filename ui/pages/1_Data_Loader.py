import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
import ccxt
import time
import numpy as np

st.set_page_config(
    page_title="Data Loader",
    page_icon="📊",
    layout="wide"
)

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

# กำหนด path ของโฟลเดอร์ data
DATA_DIR = os.path.join(root_path, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# กำหนด timeframe ที่รองรับ
TIMEFRAMES = {
    "5m": "5 minutes",
    "15m": "15 minutes", 
    "30m": "30 minutes",
    "1h": "1 hour",
    "1d": "1 day",
    "1w": "1 week",
    "1M": "1 month",
    "3M": "3 months",
    "6M": "6 months"
}

def get_filename(symbol, timeframe, start_date, end_date):
    """สร้างชื่อไฟล์ตามรูปแบบที่กำหนด"""
    symbol = symbol.replace('/', '_')
    return f"{symbol}-{timeframe}-{start_date}-{end_date}.csv"

def download_crypto_data(exchange_id, symbols, start_date, end_date, timeframe='1h'):
    """โหลดข้อมูลจาก exchange"""
    try:
        exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_data = []
        for symbol in symbols:
            st.info(f"⏳ กำลังโหลดข้อมูล {symbol} ({TIMEFRAMES[timeframe]})...")
            
            if not exchange.has['fetchOHLCV']:
                st.error(f"⚠️ {exchange_id} ไม่รองรับการโหลดข้อมูล OHLCV")
                continue
                
            current_timestamp = start_timestamp
            while current_timestamp < end_timestamp:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_timestamp, limit=1000)
                    if not ohlcv:
                        break
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # ตรวจสอบและแก้ไขข้อมูล
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].isna().any():
                            df[col] = df[col].fillna(df[col].mean())
                    
                    # แก้ไขราคาที่ผิดปกติ
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        mask = (df[col] <= 0)
                        if mask.any():
                            df.loc[mask, col] = df.loc[mask, price_cols].mean(axis=1)
                    
                    df.loc[df['volume'] < 0, 'volume'] = 0
                    
                    # ตรวจสอบความถูกต้องของราคา OHLC
                    df['high'] = df[['high', 'open', 'close', 'low']].max(axis=1)
                    df['low'] = df[['low', 'open', 'close', 'high']].min(axis=1)
                    df['open'] = df[['open', 'high', 'low']].apply(lambda x: min(max(x['open'], x['low']), x['high']), axis=1)
                    df['close'] = df[['close', 'high', 'low']].apply(lambda x: min(max(x['close'], x['low']), x['high']), axis=1)
                    
                    df['symbol'] = symbol
                    df['timeframe'] = timeframe
                    all_data.append(df)
                    
                    current_timestamp = ohlcv[-1][0] + 1
                    progress = (current_timestamp - start_timestamp) / (end_timestamp - start_timestamp) * 100
                    st.progress(min(progress, 100))
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลดข้อมูล {symbol}: {str(e)}")
                    break
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='ms')
            final_df = final_df.sort_values('timestamp').drop_duplicates()
            
            data_ready, message = check_data_quality(final_df)
            if not data_ready:
                st.warning(f"⚠️ {message}")
            
            return final_df
        else:
            st.error("⚠️ ไม่พบข้อมูลที่ต้องการ")
            return None
            
    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาดในการเชื่อมต่อกับ {exchange_id}: {str(e)}")
        return None

def check_data_quality(df):
    """ตรวจสอบคุณภาพของข้อมูล"""
    try:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}"
        
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            return False, f"พบค่า NaN ในคอลัมน์: {', '.join(nan_counts[nan_counts > 0].index)}"
        
        inf_counts = df[required_columns].isin([np.inf, -np.inf]).sum()
        if inf_counts.any():
            return False, f"พบค่า inf ในคอลัมน์: {', '.join(inf_counts[inf_counts > 0].index)}"
        
        if len(df) < 100:
            return False, "ข้อมูลน้อยเกินไป (น้อยกว่า 100 แถว)"
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                return False, f"พบราคา 0 หรือติดลบในคอลัมน์ {col}"
        
        return True, "ข้อมูลมีคุณภาพดี"
        
    except Exception as e:
        return False, f"เกิดข้อผิดพลาดในการตรวจสอบข้อมูล: {str(e)}"

def show_data_info():
    """แสดงข้อมูลไฟล์ในโฟลเดอร์ data"""
    st.subheader("📁 ไฟล์ข้อมูลที่มีอยู่")
    
    if not os.path.exists(DATA_DIR):
        st.warning("⚠️ ไม่พบโฟลเดอร์ data")
        return
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        st.info("📄 ยังไม่มีไฟล์ข้อมูล")
        return
    
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {file}")
        with col2:
            st.write(f"{file_size:.2f} MB")
        with col3:
            if st.button("🗑️", key=f"delete_{file}"):
                try:
                    os.remove(file_path)
                    st.success(f"ลบไฟล์ {file} เรียบร้อย")
                    st.rerun()
                except Exception as e:
                    st.error(f"ไม่สามารถลบไฟล์ได้: {str(e)}")

def show_data_preview(df):
    """แสดงตัวอย่างข้อมูล"""
    st.subheader("👀 ตัวอย่างข้อมูล")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 สถิติข้อมูล")
    st.dataframe(df.describe(), use_container_width=True)

# Main UI
def data_loader_ui():
    """UI สำหรับการโหลดข้อมูล"""
    st.header("📊 Data Loader")

    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown("""
**ขั้นตอนที่ 1 จาก 6** — โหลดข้อมูลราคา OHLCV จาก exchange

1. เลือก Exchange → ใส่สกุลเงิน (เช่น `BTC/USDT,ETH/USDT`) → เลือก Timeframe
2. ตั้งช่วงวันที่ — แนะนำอย่างน้อย **6 เดือน** และ timeframe **1d**
3. กด **เริ่มโหลดข้อมูล** → ไฟล์จะบันทึกใน `data/`
4. นำไฟล์ที่ได้ไปใช้ต่อที่หน้า **🔧 Data Prepare**

> ดูคำแนะนำเพิ่มเติมที่หน้า **📖 Help**
""")

    # แสดงข้อมูลไฟล์ที่มีอยู่
    show_data_info()
    
    st.markdown("---")
    
    # ฟอร์มสำหรับโหลดข้อมูลใหม่
    st.subheader("🔄 โหลดข้อมูลใหม่")
    
    with st.form("data_loader_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            exchange = st.selectbox(
                "🏢 เลือก Exchange",
                ["binance", "bybit", "okx"],
                help="เลือก cryptocurrency exchange ที่ต้องการโหลดข้อมูล"
            )
            
            symbols_input = st.text_input(
                "💱 สกุลเงิน (คั่นด้วย comma)",
                value="BTC/USDT,ETH/USDT",
                help="ใส่คู่เทรดที่ต้องการ เช่น BTC/USDT,ETH/USDT"
            )
            
            timeframe = st.selectbox(
                "⏰ Timeframe",
                list(TIMEFRAMES.keys()),
                index=3,  # default to 1h
                format_func=lambda x: f"{x} ({TIMEFRAMES[x]})"
            )
        
        with col2:
            start_date = st.date_input(
                "📅 วันที่เริ่มต้น",
                value=datetime.now() - timedelta(days=30)
            )
            
            end_date = st.date_input(
                "📅 วันที่สิ้นสุด",
                value=datetime.now()
            )
            
            filename_prefix = st.text_input(
                "📄 คำนำหน้าชื่อไฟล์ (ไม่บังคับ)",
                help="ใส่คำนำหน้าชื่อไฟล์ถ้าต้องการ"
            )
        
        submit_button = st.form_submit_button("🚀 เริ่มโหลดข้อมูล", type="primary")
    
    # ย้าย logic การประมวลผลออกมาข้างนอก form
    if submit_button:
        if start_date >= end_date:
            st.error("⚠️ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
        else:
            symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
            
            if not symbols:
                st.error("⚠️ กรุณาใส่สกุลเงินที่ต้องการ")
            else:
                progress_container = st.container()
                
                with st.spinner("🔄 กำลังโหลดข้อมูล..."):
                    df = download_crypto_data(
                        exchange,
                        symbols,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        timeframe
                    )
                    
                    if df is not None:
                        # สร้างชื่อไฟล์
                        if filename_prefix:
                            filename = f"{filename_prefix}_{get_filename('_'.join(symbols), timeframe, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))}"
                        else:
                            filename = get_filename('_'.join(symbols), timeframe, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
                        
                        file_path = os.path.join(DATA_DIR, filename)
                        
                        # บันทึกไฟล์
                        df.to_csv(file_path, index=False)
                        
                        st.success(f"✅ โหลดข้อมูลสำเร็จ! บันทึกเป็น: {filename}")
                        st.info(f"📊 จำนวนข้อมูล: {len(df):,} แถว")
                        
                        # แสดงตัวอย่างข้อมูล
                        show_data_preview(df)
                        
                        # ปุ่มดาวน์โหลด (ย้ายออกมาข้างนอก form)
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="💾 ดาวน์โหลดไฟล์ CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv"
                        )

# เรียกใช้ UI
data_loader_ui() 