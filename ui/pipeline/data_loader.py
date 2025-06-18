import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
import ccxt
import time
import numpy as np

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

# กำหนด path ของโฟลเดอร์ data
DATA_DIR = os.path.join(root_path, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    st.info(f"📁 สร้างโฟลเดอร์ data ที่: {DATA_DIR}")

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
    # แปลงสัญลักษณ์ / เป็น _ ในชื่อสกุลเงิน
    symbol = symbol.replace('/', '_')
    return f"{symbol}-{timeframe}-{start_date}-{end_date}.csv"

def download_crypto_data(exchange_id, symbols, start_date, end_date, timeframe='1h'):
    """โหลดข้อมูลจาก exchange"""
    try:
        # สร้าง exchange instance
        exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
        # แปลงวันที่เป็น timestamp
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_data = []
        for symbol in symbols:
            st.info(f"⏳ กำลังโหลดข้อมูล {symbol} ({TIMEFRAMES[timeframe]})...")
            
            # ตรวจสอบว่ามีคู่เทรดนี้หรือไม่
            if not exchange.has['fetchOHLCV']:
                st.error(f"⚠️ {exchange_id} ไม่รองรับการโหลดข้อมูล OHLCV")
                continue
                
            # โหลดข้อมูลทีละช่วง
            current_timestamp = start_timestamp
            while current_timestamp < end_timestamp:
                try:
                    ohlcv = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=current_timestamp,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                        
                    # แปลงข้อมูลเป็น DataFrame
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # ตรวจสอบและแก้ไขข้อมูลที่ไม่ถูกต้อง
                    # 1. แปลงเป็น float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 2. แทนที่ค่า NaN ด้วยค่าเฉลี่ย
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if df[col].isna().any():
                            df[col] = df[col].fillna(df[col].mean())
                    
                    # 3. แทนที่ราคา 0 หรือติดลบ
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        mask = (df[col] <= 0)
                        if mask.any():
                            # ใช้ค่าเฉลี่ยของคอลัมน์อื่นๆ
                            df.loc[mask, col] = df.loc[mask, price_cols].mean(axis=1)
                    
                    # 4. แทนที่ volume ติดลบด้วย 0
                    df.loc[df['volume'] < 0, 'volume'] = 0
                    
                    # 5. ตรวจสอบความถูกต้องของราคา
                    # high ต้องมากกว่าหรือเท่ากับ open, close, low
                    df['high'] = df[['high', 'open', 'close', 'low']].max(axis=1)
                    # low ต้องน้อยกว่าหรือเท่ากับ open, close, high
                    df['low'] = df[['low', 'open', 'close', 'high']].min(axis=1)
                    # open, close ต้องอยู่ระหว่าง high และ low
                    df['open'] = df[['open', 'high', 'low']].apply(
                        lambda x: min(max(x['open'], x['low']), x['high']), axis=1
                    )
                    df['close'] = df[['close', 'high', 'low']].apply(
                        lambda x: min(max(x['close'], x['low']), x['high']), axis=1
                    )
                    
                    # เพิ่มคอลัมน์ symbol และ timeframe
                    df['symbol'] = symbol
                    df['timeframe'] = timeframe
                    
                    all_data.append(df)
                    
                    # อัพเดท timestamp สำหรับรอบถัดไป
                    current_timestamp = ohlcv[-1][0] + 1
                    
                    # แสดงความคืบหน้า
                    progress = (current_timestamp - start_timestamp) / (end_timestamp - start_timestamp) * 100
                    st.progress(min(progress, 100))
                    
                    # รอเพื่อไม่ให้เกิน rate limit
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลดข้อมูล {symbol}: {str(e)}")
                    break
        
        if all_data:
            # รวมข้อมูลทั้งหมด
            final_df = pd.concat(all_data, ignore_index=True)
            
            # แปลง timestamp เป็น datetime
            final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='ms')
            
            # เรียงข้อมูลตาม timestamp
            final_df = final_df.sort_values('timestamp')
            
            # ลบข้อมูลซ้ำ
            final_df = final_df.drop_duplicates()
            
            # ตรวจสอบข้อมูลสุดท้าย
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
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}"
        
        # ตรวจสอบค่า NaN
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            return False, f"พบค่า NaN ในคอลัมน์: {', '.join(nan_counts[nan_counts > 0].index)}"
        
        # ตรวจสอบค่า inf
        inf_counts = df[required_columns].isin([np.inf, -np.inf]).sum()
        if inf_counts.any():
            return False, f"พบค่า inf ในคอลัมน์: {', '.join(inf_counts[inf_counts > 0].index)}"
        
        # ตรวจสอบจำนวนข้อมูล
        if len(df) < 100:
            return False, "ข้อมูลน้อยเกินไป (น้อยกว่า 100 แถว)"
        
        # ตรวจสอบช่วงราคา
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                return False, f"พบราคา 0 หรือติดลบในคอลัมน์ {col}"
        
        # ตรวจสอบความถูกต้องของราคา
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            # ตรวจสอบว่า high >= open, close, low
            if (symbol_data['high'] < symbol_data[['open', 'close', 'low']].max(axis=1)).any():
                return False, f"พบราคา high ต่ำกว่า open, close, หรือ low ใน {symbol}"
            
            # ตรวจสอบว่า low <= open, close, high
            if (symbol_data['low'] > symbol_data[['open', 'close', 'high']].min(axis=1)).any():
                return False, f"พบราคา low สูงกว่า open, close, หรือ high ใน {symbol}"
        
        # ตรวจสอบ volume
        if (df['volume'] < 0).any():
            return False, "พบ volume ติดลบ"
        
        return True, "ข้อมูลมีคุณภาพดี"
    except Exception as e:
        return False, f"เกิดข้อผิดพลาดในการตรวจสอบข้อมูล: {str(e)}"

def show_data_info():
    """แสดงข้อมูลเกี่ยวกับข้อมูลที่โหลด"""
    # แสดงรายการไฟล์ข้อมูลที่มีอยู่
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if data_files:
        # สร้างข้อมูลสำหรับตาราง
        table_data = []
        for file in data_files:
            file_path = os.path.join(DATA_DIR, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            try:
                df = pd.read_csv(file_path)
                # แยกข้อมูลจากชื่อไฟล์
                parts = file.replace('.csv', '').split('-')
                symbol = parts[0].replace('_', '/')
                timeframe = parts[1]
                start_date = parts[2]
                end_date = parts[3]
                
                # แปลงวันที่ให้อ่านง่าย
                start_date = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
                
                table_data.append({
                    "📄 ไฟล์": file,
                    "💱 สกุลเงิน": symbol,
                    "⏱️ Timeframe": f"{timeframe} ({TIMEFRAMES[timeframe]})",
                    "📅 วันที่เริ่มต้น": start_date,
                    "📅 วันที่สิ้นสุด": end_date,
                    "📊 จำนวนข้อมูล": f"{len(df):,} แถว",
                    "💾 ขนาดไฟล์": f"{file_size:.1f} KB"
                })
            except Exception as e:
                st.error(f"⚠️ เกิดข้อผิดพลาดในการอ่านไฟล์ {file}: {str(e)}")
        
        # แสดงผลในรูปแบบตาราง
        st.markdown("### 📊 ข้อมูลไฟล์ทั้งหมด")
        st.dataframe(
            pd.DataFrame(table_data),
            hide_index=True,
            use_container_width=True
        )
        
        # แสดงผลในรูปแบบ Card
        st.markdown("### 📁 รายละเอียดไฟล์")
        cols = st.columns(3)  # แสดง 3 คอลัมน์
        for idx, data in enumerate(table_data):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"""
                    <div style='padding: 1rem; border: 1px solid #e0e0e0; border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <h4 style='margin: 0; color: #1E88E5;'>{data['💱 สกุลเงิน']}</h4>
                        <p style='margin: 0.5rem 0;'><b>Timeframe:</b> {data['⏱️ Timeframe']}</p>
                        <p style='margin: 0.5rem 0;'><b>ช่วงเวลา:</b> {data['📅 วันที่เริ่มต้น']} ถึง {data['📅 วันที่สิ้นสุด']}</p>
                        <p style='margin: 0.5rem 0;'><b>จำนวนข้อมูล:</b> {data['📊 จำนวนข้อมูล']}</p>
                        <p style='margin: 0.5rem 0;'><b>ขนาดไฟล์:</b> {data['💾 ขนาดไฟล์']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูลในโฟลเดอร์ data")

def show_data_preview(df):
    """แสดงตัวอย่างข้อมูล"""
    with st.expander("👀 ตัวอย่างข้อมูล", expanded=True):
        st.dataframe(df.head())
        st.download_button(
            "📥 ดาวน์โหลดข้อมูล",
            df.to_csv(index=False).encode('utf-8'),
            "crypto_data.csv",
            "text/csv",
            key='download-csv'
        )

def show_data_visualization(df):
    """แสดงกราฟข้อมูล"""
    with st.expander("📈 แสดงกราฟ", expanded=True):
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['timestamp', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"⚠️ ไม่พบคอลัมน์ที่จำเป็นสำหรับการแสดงกราฟ: {', '.join(missing_columns)}")
            return
            
        # เลือกสกุลเงิน (ถ้ามี)
        if 'symbol' in df.columns:
            selected_symbol = st.selectbox(
                "เลือกสกุลเงิน",
                df['symbol'].unique()
            )
            selected_data = df[df['symbol'] == selected_symbol]
        else:
            selected_data = df
        
        # Plot price
        st.subheader("📈 กราฟราคา")
        st.line_chart(selected_data.set_index('timestamp')['close'])
        
        # Plot volume
        st.subheader("📊 กราฟปริมาณการเทรด")
        st.bar_chart(selected_data.set_index('timestamp')['volume'])

def data_loader_ui():
    """UI สำหรับจัดการข้อมูล"""
    st.header("📥 Data Management")
    
    # แสดงข้อมูลปัจจุบัน
    show_data_info()
    
    # ส่วนการโหลดข้อมูลใหม่
    with st.expander("🔄 โหลดข้อมูลใหม่", expanded=True):
        st.markdown("""
        ### 📋 คำแนะนำการโหลดข้อมูล
        1. เลือก exchange ที่ต้องการ
        2. เลือก timeframe ที่ต้องการ
        3. เลือกช่วงเวลาที่ต้องการ
        4. เลือกสกุลเงินที่ต้องการ
        5. กดปุ่มโหลดข้อมูล
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            # เลือก exchange
            exchange = st.selectbox(
                "เลือก Exchange",
                ["binance", "bybit", "okx"],
                help="เลือก exchange ที่ต้องการโหลดข้อมูล"
            )
            
            # เลือก timeframe
            timeframe = st.selectbox(
                "เลือก Timeframe",
                list(TIMEFRAMES.keys()),
                format_func=lambda x: f"{x} ({TIMEFRAMES[x]})",
                help="เลือกช่วงเวลาของข้อมูล"
            )
            
        with col2:
            start_date = st.date_input(
                "วันที่เริ่มต้น",
                value=datetime.now() - timedelta(days=30)
            )
            end_date = st.date_input(
                "วันที่สิ้นสุด",
                value=datetime.now()
            )
            
        symbols = st.multiselect(
            "เลือกสกุลเงิน",
            ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"],
            default=["BTC/USDT", "ETH/USDT"]
        )
        
        if st.button("📥 โหลดข้อมูล"):
            if not symbols:
                st.error("⚠️ กรุณาเลือกสกุลเงินอย่างน้อย 1 สกุล")
                return
                
            if start_date >= end_date:
                st.error("⚠️ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
                return
                
            # โหลดข้อมูล
            new_df = download_crypto_data(
                exchange,
                symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                timeframe
            )
            
            if new_df is not None:
                # บันทึกข้อมูลแยกตามสกุลเงิน
                for symbol in symbols:
                    symbol_data = new_df[new_df['symbol'] == symbol]
                    if not symbol_data.empty:
                        filename = get_filename(
                            symbol,
                            timeframe,
                            start_date.strftime('%Y%m%d'),
                            end_date.strftime('%Y%m%d')
                        )
                        file_path = os.path.join(DATA_DIR, filename)
                        symbol_data.to_csv(file_path, index=False)
                        st.success(f"✅ บันทึกข้อมูล {symbol} สำเร็จที่: {file_path}")
                
                # Refresh data display
                show_data_info()
    
    # แสดงข้อมูลและกราฟของไฟล์ที่เลือก
    selected_file = st.selectbox(
        "เลือกไฟล์เพื่อดูข้อมูล",
        [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')],
        format_func=lambda x: f"📄 {x}"
    )
    
    if selected_file:
        file_path = os.path.join(DATA_DIR, selected_file)
        try:
            df = pd.read_csv(file_path)
            show_data_preview(df)
            show_data_visualization(df)
        except Exception as e:
            st.error(f"⚠️ เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")
