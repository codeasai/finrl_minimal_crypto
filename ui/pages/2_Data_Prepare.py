import streamlit as st

# st.set_page_config ต้องอยู่บรรทัดแรกสุด
st.set_page_config(
    page_title="Data Prepare",
    page_icon="🔧",
    layout="wide"
)

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib, fall back to manual calculations if not available
try:
    import talib as ta
    TALIB_AVAILABLE = True
    st.success("✅ TA-Lib พร้อมใช้งาน (Advanced Indicators)")
except ImportError:
    TALIB_AVAILABLE = False
    st.warning("⚠️ TA-Lib ไม่พร้อมใช้งาน - ใช้ indicators พื้นฐาน")
    st.info("💡 ติดตั้ง TA-Lib ด้วย: `conda install ta-lib -c conda-forge`")

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

# กำหนด directories
DATA_DIR = os.path.join(root_path, "data")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "data_prepare")

# สร้างโฟลเดอร์ถ้ายังไม่มี
if not os.path.exists(PREPARED_DATA_DIR):
    os.makedirs(PREPARED_DATA_DIR)

def find_data_files():
    """ค้นหาไฟล์ข้อมูล CSV ทั้งหมด"""
    data_files = []
    
    if not os.path.exists(DATA_DIR):
        return data_files
    
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            file_path = os.path.join(DATA_DIR, file)
            try:
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size_mb': file_size
                })
            except:
                continue
    
    return sorted(data_files, key=lambda x: x['size_mb'], reverse=True)

def load_data(file_path):
    """โหลดข้อมูลจากไฟล์"""
    try:
        df = pd.read_csv(file_path)
        
        # ตรวจสอบและแปลงคอลัมน์
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"⚠️ ไม่พบคอลัมน์ที่จำเป็น: {missing_cols}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ได้: {str(e)}")
        return None

def add_technical_indicators(df):
    """เพิ่ม Technical Indicators"""
    st.info("🔧 กำลังคำนวณ Technical Indicators...")
    
    df = df.copy()
    progress_bar = st.progress(0)
    
    # จัดกลุ่มตาม symbol ถ้ามี
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        total_symbols = len(symbols)
        
        # ใช้ list เก็บ dataframes แทนการแก้ไข in-place
        symbol_dfs = []
        
        for i, symbol in enumerate(symbols):
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy().sort_values('timestamp').reset_index(drop=True)
            
            # คำนวณ indicators สำหรับแต่ละ symbol
            with st.spinner(f"กำลังคำนวณ indicators สำหรับ {symbol}..."):
                symbol_data = calculate_indicators_for_symbol(symbol_data)
            symbol_dfs.append(symbol_data)
            
            progress_bar.progress((i + 1) / total_symbols)
        
        # รวม dataframes กลับ
        df = pd.concat(symbol_dfs, ignore_index=True)
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    else:
        # ถ้าไม่มี symbol คำนวณทั้งหมดเลย
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = calculate_indicators_for_symbol(df)
        progress_bar.progress(1.0)
    
    # ตรวจสอบว่า indicators ถูกเพิ่มแล้ว
    exclude_cols = ['timestamp', 'date', 'symbol', 'tic', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
    indicators = [col for col in df.columns if col not in exclude_cols]
    
    if indicators:
        st.success(f"✅ Technical Indicators คำนวณเสร็จสิ้น! เพิ่ม {len(indicators)} indicators")
    else:
        st.warning("⚠️ ไม่มี indicators ถูกเพิ่ม")
    
    return df

def calculate_indicators_for_symbol(data):
    """คำนวณ indicators สำหรับข้อมูล 1 symbol"""
    data = data.copy()
    
    # ตรวจสอบข้อมูลพื้นฐาน
    if len(data) < 50:  # ต้องมีข้อมูลเพียงพอ
        st.warning(f"⚠️ ข้อมูลไม่เพียงพอสำหรับคำนวณ indicators (มี {len(data)} แถว)")
        return data
    
    # แปลงข้อมูลให้เป็น numeric และตรวจสอบ
    close = pd.to_numeric(data['close'], errors='coerce').values
    high = pd.to_numeric(data['high'], errors='coerce').values
    low = pd.to_numeric(data['low'], errors='coerce').values
    volume = pd.to_numeric(data['volume'], errors='coerce').values
    open_price = pd.to_numeric(data['open'], errors='coerce').values
    
    # เติมค่า NaN ที่อาจเกิดจากการแปลง
    close = np.nan_to_num(close)
    high = np.nan_to_num(high)
    low = np.nan_to_num(low)
    volume = np.nan_to_num(volume)
    open_price = np.nan_to_num(open_price)
    
    # เก็บจำนวนคอลัมน์เดิมเพื่อเปรียบเทียบ
    original_cols = len(data.columns)
    
    try:
        if TALIB_AVAILABLE:
            # ใช้ TA-Lib
            data = calculate_talib_indicators(data, close, high, low, volume, open_price)
        else:
            # ใช้ manual calculation
            data = calculate_manual_indicators(data, close, high, low, volume, open_price)
            
    except Exception as e:
        st.error(f"⚠️ ไม่สามารถคำนวณ indicators ได้: {str(e)}")
        st.write(f"Error details: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        
        # ถ้า TA-Lib ล้มเหลว ใช้ manual calculation
        try:
            st.warning("🔄 ลองใช้ manual calculation...")
            data = calculate_manual_indicators(data, close, high, low, volume, open_price)
            st.success("✅ Manual calculation สำเร็จ")
        except Exception as e2:
            st.error(f"⚠️ Manual calculation ก็ล้มเหลว: {str(e2)}")
            st.code(traceback.format_exc())
    
    # เติมค่า NaN ด้วยวิธีการต่างๆ
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    try:
        # ใช้ method ใหม่สำหรับ pandas เวอร์ชันใหม่
        data[numeric_cols] = data[numeric_cols].ffill().bfill().fillna(0)
    except:
        # fallback สำหรับ pandas เวอร์ชันเก่า
        try:
            data[numeric_cols] = data[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        except:
            # ถ้าทั้งสองวิธีไม่ได้ ใช้แค่ fillna(0)
            data[numeric_cols] = data[numeric_cols].fillna(0)
    
    return data

def calculate_talib_indicators(data, close, high, low, volume, open_price):
    """คำนวณ indicators ด้วย TA-Lib"""
    # 1. Moving Averages
    data['sma_5'] = ta.SMA(close, timeperiod=5)
    data['sma_10'] = ta.SMA(close, timeperiod=10)
    data['sma_20'] = ta.SMA(close, timeperiod=20)
    data['sma_50'] = ta.SMA(close, timeperiod=50)
    data['ema_12'] = ta.EMA(close, timeperiod=12)
    data['ema_26'] = ta.EMA(close, timeperiod=26)
    data['ema_20'] = ta.EMA(close, timeperiod=20)
    
    # 2. Momentum Indicators
    data['rsi_14'] = ta.RSI(close, timeperiod=14)
    data['rsi_21'] = ta.RSI(close, timeperiod=21)
    
    # 3. MACD
    macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data['macd'] = macd
    data['macd_signal'] = macd_signal
    data['macd_hist'] = macd_hist
    
    # 4. Bollinger Bands
    bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['bb_upper'] = bb_upper
    data['bb_middle'] = bb_middle
    data['bb_lower'] = bb_lower
    data['bb_width'] = bb_upper - bb_lower
    data['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    
    # 5. Volume Indicators
    data['volume_sma_20'] = ta.SMA(volume.astype(float), timeperiod=20)
    data['volume_ratio'] = volume / data['volume_sma_20']
    data['ad'] = ta.AD(high, low, close, volume.astype(float))
    data['obv'] = ta.OBV(close, volume.astype(float))
    
    # 6. Volatility Indicators
    data['atr'] = ta.ATR(high, low, close, timeperiod=14)
    data['natr'] = ta.NATR(high, low, close, timeperiod=14)
    
    # 7. Stochastic
    slowk, slowd = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    data['stoch_k'] = slowk
    data['stoch_d'] = slowd
    
    # 8. Williams %R
    data['willr'] = ta.WILLR(high, low, close, timeperiod=14)
    
    # 9. CCI
    data['cci'] = ta.CCI(high, low, close, timeperiod=14)
    
    # 10. ADX
    data['adx'] = ta.ADX(high, low, close, timeperiod=14)
    
    # เพิ่ม Price Features
    data = add_price_features(data, close, high, low, open_price)
    
    return data

def calculate_manual_indicators(data, close, high, low, volume, open_price):
    """คำนวณ indicators แบบ manual (ไม่ใช้ TA-Lib)"""
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    volume_series = pd.Series(volume)
    
    # 1. Moving Averages
    data['sma_5'] = close_series.rolling(window=5).mean()
    data['sma_10'] = close_series.rolling(window=10).mean()
    data['sma_20'] = close_series.rolling(window=20).mean()
    data['sma_50'] = close_series.rolling(window=50).mean()
    
    # EMA calculation
    data['ema_12'] = close_series.ewm(span=12).mean()
    data['ema_26'] = close_series.ewm(span=26).mean()
    data['ema_20'] = close_series.ewm(span=20).mean()
    
    # 2. RSI
    data['rsi_14'] = calculate_rsi(close_series, 14)
    data['rsi_21'] = calculate_rsi(close_series, 21)
    
    # 3. MACD
    macd = data['ema_12'] - data['ema_26']
    macd_signal = macd.ewm(span=9).mean()
    data['macd'] = macd
    data['macd_signal'] = macd_signal
    data['macd_hist'] = macd - macd_signal
    
    # 4. Bollinger Bands
    bb_middle = close_series.rolling(window=20).mean()
    bb_std = close_series.rolling(window=20).std()
    data['bb_upper'] = bb_middle + (bb_std * 2)
    data['bb_middle'] = bb_middle
    data['bb_lower'] = bb_middle - (bb_std * 2)
    data['bb_width'] = data['bb_upper'] - data['bb_lower']
    data['bb_position'] = (close_series - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # 5. Volume Indicators
    data['volume_sma_20'] = volume_series.rolling(window=20).mean()
    data['volume_ratio'] = volume_series / data['volume_sma_20']
    
    # OBV manual calculation
    price_change = close_series.diff()
    obv = []
    obv_value = 0
    for i in range(len(volume_series)):
        if i == 0:
            obv_value = volume_series.iloc[i]
        elif price_change.iloc[i] > 0:
            obv_value += volume_series.iloc[i]
        elif price_change.iloc[i] < 0:
            obv_value -= volume_series.iloc[i]
        obv.append(obv_value)
    data['obv'] = obv
    
    # A/D Line manual calculation
    money_flow_multiplier = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series)
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    money_flow_volume = money_flow_multiplier * volume_series
    data['ad'] = money_flow_volume.cumsum()
    
    # 6. ATR
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr'] = tr.rolling(window=14).mean()
    data['natr'] = (data['atr'] / close_series) * 100
    
    # 7. Stochastic
    lowest_low = low_series.rolling(window=5).min()
    highest_high = high_series.rolling(window=5).max()
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    data['stoch_k'] = k_percent.rolling(window=3).mean()
    data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
    
    # 8. Williams %R
    data['willr'] = -100 * ((highest_high - close_series) / (highest_high - lowest_low))
    
    # 9. CCI
    tp = (high_series + low_series + close_series) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.mean())))
    data['cci'] = (tp - sma_tp) / (0.015 * mad)
    
    # 10. ADX (simplified version)
    # Calculate True Range first (already done above in ATR section)
    dm_plus = np.maximum(high_series.diff(), 0)
    dm_minus = np.maximum(-low_series.diff(), 0)
    
    # Smooth the DM values
    dm_plus_smooth = dm_plus.rolling(window=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / data['atr'])
    di_minus = 100 * (dm_minus_smooth / data['atr'])
    
    # Calculate DX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # Calculate ADX (smoothed DX)
    data['adx'] = dx.rolling(window=14).mean()
    
    # เพิ่ม Price Features
    data = add_price_features(data, close, high, low, open_price)
    
    return data

def calculate_rsi(close_series, period):
    """คำนวณ RSI"""
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_price_features(data, close, high, low, open_price):
    """เพิ่ม Price Features"""
    close_series = pd.Series(close)
    
    # Price Features
    data['price_change'] = (close - open_price) / open_price
    data['price_range'] = (high - low) / close
    data['gap'] = (open_price - close_series.shift(1)) / close_series.shift(1)
    
    # Returns และ Volatility
    data['returns'] = close_series.pct_change()
    data['volatility'] = data['returns'].rolling(window=20).std()
    data['log_returns'] = np.log(close_series / close_series.shift(1))
    
    # Support/Resistance
    data['resistance'] = close_series.rolling(window=20).max()
    data['support'] = close_series.rolling(window=20).min()
    data['distance_to_resistance'] = (data['resistance'] - close_series) / close_series
    data['distance_to_support'] = (close_series - data['support']) / close_series
    
    return data

def normalize_data(df, method='standard'):
    """Normalize ข้อมูล"""
    st.info(f"📊 กำลัง Normalize ข้อมูลด้วยวิธี: {method}")
    
    df = df.copy()
    
    # คอลัมน์ที่ต้อง normalize
    exclude_cols = ['timestamp', 'date', 'symbol', 'tic', 'timeframe']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        st.error("⚠️ วิธี normalize ไม่ถูกต้อง")
        return df
    
    # Normalize แยกตาม symbol ถ้ามี
    if 'symbol' in df.columns:
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            df.loc[mask, numeric_cols] = scaler.fit_transform(df.loc[mask, numeric_cols])
    else:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    st.success("✅ Normalize ข้อมูลเสร็จสิ้น!")
    return df

def prepare_for_environment(df):
    """เตรียมข้อมูลสำหรับ Environment"""
    st.info("🏗️ กำลังเตรียมข้อมูลสำหรับ Trading Environment...")
    
    df = df.copy()
    
    # สร้างคอลัมน์ที่จำเป็นสำหรับ FinRL
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    
    if 'tic' not in df.columns and 'symbol' in df.columns:
        df['tic'] = df['symbol']
    
    # เรียงข้อมูลตามวันที่และ symbol
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # ตรวจสอบและแก้ไขข้อมูล
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    st.success("✅ เตรียมข้อมูลสำหรับ Environment เสร็จสิ้น!")
    return df

def show_data_statistics(df):
    """แสดงสถิติข้อมูล"""
    st.subheader("📊 สถิติข้อมูลหลังการเตรียม")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("จำนวนแถว", f"{len(df):,}")
    with col2:
        st.metric("จำนวนคอลัมน์", len(df.columns))
    with col3:
        if 'symbol' in df.columns or 'tic' in df.columns:
            symbol_col = 'symbol' if 'symbol' in df.columns else 'tic'
            st.metric("จำนวนสกุลเงิน", df[symbol_col].nunique())
    with col4:
        missing_count = df.isnull().sum().sum()
        st.metric("ค่า Missing", missing_count)
    
    # แสดงข้อมูลตัวอย่าง
    st.subheader("👀 ตัวอย่างข้อมูลที่เตรียมแล้ว")
    st.dataframe(df.head(10), use_container_width=True)
    
    # แสดงรายชื่อ indicators
    exclude_cols = ['timestamp', 'date', 'symbol', 'tic', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
    indicators = [col for col in df.columns if col not in exclude_cols]
    
    if indicators:
        st.subheader("📈 Technical Indicators ที่เพิ่ม")
        st.write(f"จำนวน indicators: {len(indicators)}")
        
        # แสดงใน columns
        cols = st.columns(4)
        for i, indicator in enumerate(indicators):
            with cols[i % 4]:
                st.write(f"• {indicator}")

# Main UI
def data_prepare_ui():
    """UI หลักสำหรับการเตรียมข้อมูล"""
    st.header("🔧 Data Prepare")
    st.markdown("### เตรียมข้อมูลสำหรับการเทรน AI Agent")

    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown("""
**ขั้นตอนที่ 2 จาก 6** — เพิ่ม Technical Indicators ให้ข้อมูล OHLCV

1. เลือกไฟล์จาก **Data Loader** (ไฟล์ใน `data/`)
2. กด **เพิ่ม Technical Indicators** — ระบบจะเพิ่ม ~30 indicators อัตโนมัติ
3. กด **Export** เพื่อบันทึกไปยัง `data/data_prepare/`
4. นำไฟล์ที่ Export ไปใช้ที่หน้า **🎯 Train Agent**

> **สำคัญ:** ต้องใช้ไฟล์จาก `data/data_prepare/` เท่านั้นในการเทรน
> ดูรายละเอียด indicators ทั้งหมดที่หน้า **📖 Help**
""")
    
    # ค้นหาไฟล์ข้อมูล
    data_files = find_data_files()
    
    if not data_files:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูล กรุณาโหลดข้อมูลจากหน้า Data Loader ก่อน")
        return
    
    # เลือกไฟล์
    st.subheader("📁 เลือกไฟล์ข้อมูล")
    
    selected_file = st.selectbox(
        "เลือกไฟล์ข้อมูลที่ต้องการเตรียม",
        options=data_files,
        format_func=lambda x: f"{x['name']} ({x['size_mb']:.2f} MB)"
    )
    
    if not selected_file:
        return
    
    # แสดงการตั้งค่า
    st.subheader("⚙️ การตั้งค่าการเตรียมข้อมูล")
    
    col1, col2 = st.columns(2)
    
    with col1:
        add_indicators = st.checkbox("✅ เพิ่ม Technical Indicators", value=True)
        normalize_enabled = st.checkbox("✅ Normalize ข้อมูล", value=True)
        
        if normalize_enabled:
            normalize_method = st.selectbox(
                "วิธี Normalization",
                ["standard", "minmax", "robust"],
                index=0,
                help="Standard: z-score, MinMax: 0-1, Robust: median-based"
            )
    
    with col2:
        prepare_env = st.checkbox("✅ เตรียมสำหรับ Environment", value=True)
        
        # ตั้งชื่อไฟล์
        base_name = selected_file['name'].replace('.csv', '')
        output_filename = st.text_input(
            "ชื่อไฟล์ผลลัพธ์",
            value=f"prepared_{base_name}.csv",
            help="ไฟล์จะถูกบันทึกใน data/data_prepare/"
        )
    
    # ปุ่มเริ่มประมวลผล
    if st.button("🚀 เริ่มเตรียมข้อมูล", type="primary"):
        
        # โหลดข้อมูล
        with st.spinner("📂 กำลังโหลดข้อมูล..."):
            df = load_data(selected_file['path'])
        
        if df is None:
            return
        
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")
        
        # แสดงข้อมูลต้นฉบับ
        with st.expander("👀 ข้อมูลต้นฉบับ"):
            st.dataframe(df.head(), use_container_width=True)
        
        # เพิ่ม Technical Indicators
        if add_indicators:
            df = add_technical_indicators(df)
        
        # Normalize ข้อมูล
        if normalize_enabled:
            df = normalize_data(df, normalize_method)
        
        # เตรียมสำหรับ Environment
        if prepare_env:
            df = prepare_for_environment(df)
        
        # แสดงสถิติผลลัพธ์
        show_data_statistics(df)
        
        # Debug: แสดงคอลัมน์ทั้งหมดก่อนบันทึก
        st.write("🔍 Debug - คอลัมน์ทั้งหมดในข้อมูล:")
        st.write(f"จำนวนคอลัมน์ทั้งหมด: {len(df.columns)}")
        st.write("รายชื่อคอลัมน์:", list(df.columns))
        
        # บันทึกไฟล์
        output_path = os.path.join(PREPARED_DATA_DIR, output_filename)
        
        try:
            df.to_csv(output_path, index=False)
            st.success(f"✅ บันทึกไฟล์สำเร็จ: {output_path}")
            
            # แสดงข้อมูลไฟล์
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            st.info(f"📄 ขนาดไฟล์: {file_size:.2f} MB")
            
            # ตรวจสอบไฟล์ที่บันทึกแล้ว
            saved_df = pd.read_csv(output_path)
            st.write(f"🔍 ไฟล์ที่บันทึกมี {len(saved_df.columns)} คอลัมน์")
            st.write("คอลัมน์ในไฟล์ที่บันทึก:", list(saved_df.columns))
            
            # ปุ่มดาวน์โหลด
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="💾 ดาวน์โหลดไฟล์",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"⚠️ ไม่สามารถบันทึกไฟล์ได้: {str(e)}")
    
    # แสดงไฟล์ที่เตรียมแล้ว
    st.markdown("---")
    st.subheader("📚 ไฟล์ข้อมูลที่เตรียมแล้ว")
    
    if os.path.exists(PREPARED_DATA_DIR):
        prepared_files = [f for f in os.listdir(PREPARED_DATA_DIR) if f.endswith('.csv')]
        
        if prepared_files:
            for file in prepared_files:
                file_path = os.path.join(PREPARED_DATA_DIR, file)
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file}")
                with col2:
                    st.write(f"{file_size:.2f} MB")
                with col3:
                    if st.button("🗑️", key=f"delete_prepared_{file}"):
                        try:
                            os.remove(file_path)
                            st.success(f"ลบไฟล์ {file} เรียบร้อย")
                            st.rerun()
                        except Exception as e:
                            st.error(f"ไม่สามารถลบไฟล์ได้: {str(e)}")
        else:
            st.info("📄 ยังไม่มีไฟล์ข้อมูลที่เตรียมแล้ว")

# เรียกใช้ UI
data_prepare_ui() 