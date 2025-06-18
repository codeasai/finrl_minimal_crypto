# advanced_crypto_agent.py
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import yfinance as yf
import torch
try:
    import talib as ta
    TA_AVAILABLE = True
    print("✅ Using TA-Lib for technical indicators")
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ TA-Lib not found, using manual calculations")
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

# FinRL imports
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# Import config
from config import *

# สร้างโฟลเดอร์สำหรับเก็บข้อมูล
DATA_DIR = "data"
ADVANCED_MODEL_DIR = "models"  # ใช้ models directory เดียวกัน
for dir_name in [DATA_DIR, ADVANCED_MODEL_DIR]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def setup_device():
    """ตรวจสอบและตั้งค่าการใช้งาน GPU/CPU"""
    print("\n🔍 ตรวจสอบการใช้งาน GPU/CPU")
    print("-" * 50)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ พบ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 จำนวน GPU: {torch.cuda.device_count()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("ℹ️ ไม่พบ GPU ใช้ CPU แทน")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else "-1"
    return device

def download_crypto_data_advanced(symbols=None, lookback_days=365*2, force_download=False):
    """
    ดาวน์โหลดข้อมูล crypto ขั้นสูงพร้อมข้อมูลเพิ่มเติม
    """
    if symbols is None:
        symbols = CRYPTO_SYMBOLS + ['ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD', 'MATIC-USD']
    
    # คำนวณวันที่ย้อนหลัง
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    data_file = os.path.join(DATA_DIR, "advanced_crypto_data.csv")
    
    if os.path.exists(data_file) and not force_download:
        print("📂 Loading existing advanced data...")
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"✅ Loaded {len(df)} rows of data")
            print(f"📈 Symbols: {df['tic'].unique()}")
            return df
        except Exception as e:
            print(f"⚠️ Error loading existing data: {str(e)}")
    
    print(f"📊 Downloading advanced crypto data for {len(symbols)} symbols...")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    df_list = []
    for symbol in symbols:
        print(f"📥 Downloading {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            
            # ดาวน์โหลดข้อมูลราคา
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D',
                auto_adjust=True
            )
            
            if len(df) == 0:
                print(f"⚠️ No data for {symbol}")
                continue
            
            df['tic'] = symbol
            df['timestamp'] = df.index
            
            # แปลงชื่อคอลัมน์
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # เพิ่มข้อมูล market cap และ features อื่นๆ
            try:
                info = ticker.info
                df['market_cap'] = info.get('marketCap', 0)
                df['circulating_supply'] = info.get('circulatingSupply', 0)
            except:
                df['market_cap'] = 0
                df['circulating_supply'] = 0
            
            df_list.append(df)
            print(f"✅ Downloaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            continue
    
    if not df_list:
        raise ValueError("ไม่พบข้อมูลใดๆ")
    
    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    # บันทึกข้อมูล
    df.to_csv(data_file, index=False)
    print(f"💾 Saved data to {data_file}")
    print(f"✅ Downloaded {len(df)} rows total")
    
    return df

def add_advanced_technical_indicators(df):
    """
    เพิ่ม technical indicators ขั้นสูงมากขึ้น
    """
    print("📈 Adding advanced technical indicators...")
    
    df = df.copy()
    
    # แปลงชื่อคอลัมน์
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    # เรียงข้อมูลก่อนคำนวณ indicators
    df = df.sort_values(['tic', 'timestamp']).reset_index(drop=True)
    
    # คำนวณ indicators แยกตาม symbol
    for symbol in df['tic'].unique():
        mask = df['tic'] == symbol
        symbol_data = df[mask].copy()
        
        if len(symbol_data) < 50:  # ข้ามถ้าข้อมูลน้อยเกินไป
            continue
            
        try:
            if TA_AVAILABLE:
                print(f"📊 Using TA-Lib for {symbol}")
                # ใช้ TA-Lib
                symbol_df = symbol_data.copy()
                symbol_df = symbol_df.set_index('timestamp')
                
                # 1. Moving Averages
                try:
                    close_values = symbol_df['close'].values
                    
                    df.loc[mask, 'sma_5'] = ta.SMA(close_values, timeperiod=5)
                    df.loc[mask, 'sma_10'] = ta.SMA(close_values, timeperiod=10)
                    df.loc[mask, 'sma_20'] = ta.SMA(close_values, timeperiod=20)
                    df.loc[mask, 'sma_50'] = ta.SMA(close_values, timeperiod=50)
                    df.loc[mask, 'sma_200'] = ta.SMA(close_values, timeperiod=200)
                    
                    df.loc[mask, 'ema_5'] = ta.EMA(close_values, timeperiod=5)
                    df.loc[mask, 'ema_10'] = ta.EMA(close_values, timeperiod=10)
                    df.loc[mask, 'ema_20'] = ta.EMA(close_values, timeperiod=20)
                    df.loc[mask, 'ema_50'] = ta.EMA(close_values, timeperiod=50)
                except Exception as e:
                    print(f"⚠️ MA error for {symbol}: {e}")
                    # Default values if error
                    close_values = symbol_df['close'].values
                    df.loc[mask, 'sma_5'] = close_values
                    df.loc[mask, 'sma_10'] = close_values
                    df.loc[mask, 'sma_20'] = close_values
                    df.loc[mask, 'sma_50'] = close_values
                    df.loc[mask, 'sma_200'] = close_values
                    df.loc[mask, 'ema_5'] = close_values
                    df.loc[mask, 'ema_10'] = close_values
                    df.loc[mask, 'ema_20'] = close_values
                    df.loc[mask, 'ema_50'] = close_values
                
                # 2. Momentum Indicators
                try:
                    close_values = symbol_df['close'].values
                    high_values = symbol_df['high'].values
                    low_values = symbol_df['low'].values
                    
                    df.loc[mask, 'rsi_14'] = ta.RSI(close_values, timeperiod=14)
                    df.loc[mask, 'rsi_21'] = ta.RSI(close_values, timeperiod=21)
                    
                    stoch_k, stoch_d = ta.STOCH(high_values, low_values, close_values)
                    df.loc[mask, 'stoch_k'] = stoch_k
                    df.loc[mask, 'stoch_d'] = stoch_d
                    
                    df.loc[mask, 'willr'] = ta.WILLR(high_values, low_values, close_values)
                except Exception as e:
                    print(f"⚠️ Momentum error for {symbol}: {e}")
                    # Default values if error
                    df.loc[mask, 'rsi_14'] = np.full(len(symbol_data), 50)
                    df.loc[mask, 'rsi_21'] = np.full(len(symbol_data), 50)
                    df.loc[mask, 'stoch_k'] = np.full(len(symbol_data), 50)
                    df.loc[mask, 'stoch_d'] = np.full(len(symbol_data), 50)
                    df.loc[mask, 'willr'] = np.full(len(symbol_data), -50)
                
                # 3. MACD
                try:
                    close_values = symbol_df['close'].values
                    macd, macd_signal, macd_hist = ta.MACD(close_values)
                    df.loc[mask, 'macd'] = macd
                    df.loc[mask, 'macd_signal'] = macd_signal
                    df.loc[mask, 'macd_hist'] = macd_hist
                except Exception as e:
                    print(f"⚠️ MACD error for {symbol}: {e}")
                    # Default values if error
                    df.loc[mask, 'macd'] = np.zeros(len(symbol_data))
                    df.loc[mask, 'macd_signal'] = np.zeros(len(symbol_data))
                    df.loc[mask, 'macd_hist'] = np.zeros(len(symbol_data))
                
                # 4. Bollinger Bands
                try:
                    close_values = symbol_df['close'].values
                    bb_upper, bb_middle, bb_lower = ta.BBANDS(close_values)
                    
                    df.loc[mask, 'bb_upper'] = bb_upper
                    df.loc[mask, 'bb_middle'] = bb_middle
                    df.loc[mask, 'bb_lower'] = bb_lower
                    df.loc[mask, 'bb_width'] = bb_upper - bb_lower
                    
                    # คำนวณ bb_position
                    bb_width = bb_upper - bb_lower
                    bb_width = np.where(bb_width == 0, 1, bb_width)  # avoid division by zero
                    df.loc[mask, 'bb_position'] = (close_values - bb_lower) / bb_width
                except Exception as e:
                    print(f"⚠️ Bollinger Bands error for {symbol}: {e}")
                    # Default values if error
                    close_values = symbol_df['close'].values
                    df.loc[mask, 'bb_upper'] = close_values
                    df.loc[mask, 'bb_middle'] = close_values
                    df.loc[mask, 'bb_lower'] = close_values
                    df.loc[mask, 'bb_width'] = np.zeros(len(symbol_data))
                    df.loc[mask, 'bb_position'] = np.full(len(symbol_data), 0.5)
                
                # 5. Volume Indicators
                try:
                    high_values = symbol_df['high'].values
                    low_values = symbol_df['low'].values
                    close_values = symbol_df['close'].values
                    volume_values = symbol_df['volume'].values
                    
                    df.loc[mask, 'ad'] = ta.AD(high_values, low_values, close_values, volume_values)
                    df.loc[mask, 'obv'] = ta.OBV(close_values, volume_values)
                    df.loc[mask, 'volume_sma_20'] = ta.SMA(volume_values, timeperiod=20)
                except Exception as e:
                    print(f"⚠️ Volume indicators error for {symbol}: {e}")
                    # Default values if error
                    df.loc[mask, 'ad'] = np.zeros(len(symbol_data))
                    df.loc[mask, 'obv'] = np.cumsum(symbol_df['volume'].values)  # simplified OBV
                    df.loc[mask, 'volume_sma_20'] = np.full(len(symbol_data), symbol_df['volume'].mean())
                
                # 6. Volatility Indicators
                try:
                    high_values = symbol_df['high'].values
                    low_values = symbol_df['low'].values
                    close_values = symbol_df['close'].values
                    
                    df.loc[mask, 'atr'] = ta.ATR(high_values, low_values, close_values)
                    df.loc[mask, 'natr'] = ta.NATR(high_values, low_values, close_values)
                except Exception as e:
                    print(f"⚠️ Volatility indicators error for {symbol}: {e}")
                    # Default values if error
                    df.loc[mask, 'atr'] = np.zeros(len(symbol_data))
                    df.loc[mask, 'natr'] = np.zeros(len(symbol_data))
                
                # 7. Trend Indicators
                try:
                    high_values = symbol_df['high'].values
                    low_values = symbol_df['low'].values
                    close_values = symbol_df['close'].values
                    
                    df.loc[mask, 'adx'] = ta.ADX(high_values, low_values, close_values)
                    df.loc[mask, 'cci'] = ta.CCI(high_values, low_values, close_values)
                except Exception as e:
                    print(f"⚠️ Trend indicators error for {symbol}: {e}")
                    # Default values if error
                    df.loc[mask, 'adx'] = np.full(len(symbol_data), 25)
                    df.loc[mask, 'cci'] = np.zeros(len(symbol_data))
                
            else:
                print(f"📊 Using manual calculations for {symbol}")
                # Manual calculations
                close_series = pd.Series(symbol_data['close'].values)
                high_series = pd.Series(symbol_data['high'].values)
                low_series = pd.Series(symbol_data['low'].values)
                volume_series = pd.Series(symbol_data['volume'].values)
                
                # 1. Moving Averages
                df.loc[mask, 'sma_5'] = close_series.rolling(5).mean().fillna(close_series).values
                df.loc[mask, 'sma_10'] = close_series.rolling(10).mean().fillna(close_series).values
                df.loc[mask, 'sma_20'] = close_series.rolling(20).mean().fillna(close_series).values
                df.loc[mask, 'sma_50'] = close_series.rolling(50).mean().fillna(close_series).values
                df.loc[mask, 'sma_200'] = close_series.rolling(200).mean().fillna(close_series).values
                
                df.loc[mask, 'ema_5'] = close_series.ewm(span=5).mean().fillna(close_series).values
                df.loc[mask, 'ema_10'] = close_series.ewm(span=10).mean().fillna(close_series).values
                df.loc[mask, 'ema_20'] = close_series.ewm(span=20).mean().fillna(close_series).values
                df.loc[mask, 'ema_50'] = close_series.ewm(span=50).mean().fillna(close_series).values
                
                # 2. RSI
                delta = close_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                df.loc[mask, 'rsi_14'] = rsi.fillna(50).values
                
                gain_21 = (delta.where(delta > 0, 0)).rolling(window=21).mean()
                loss_21 = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
                rs_21 = gain_21 / (loss_21 + 1e-8)
                rsi_21 = 100 - (100 / (1 + rs_21))
                df.loc[mask, 'rsi_21'] = rsi_21.fillna(50).values
                
                # 3. MACD
                exp1 = close_series.ewm(span=12).mean()
                exp2 = close_series.ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                hist = macd - signal
                df.loc[mask, 'macd'] = macd.fillna(0).values
                df.loc[mask, 'macd_signal'] = signal.fillna(0).values
                df.loc[mask, 'macd_hist'] = hist.fillna(0).values
                
                # 4. Bollinger Bands
                sma_20 = close_series.rolling(20).mean()
                std_20 = close_series.rolling(20).std()
                df.loc[mask, 'bb_upper'] = (sma_20 + (std_20 * 2)).fillna(close_series).values
                df.loc[mask, 'bb_middle'] = sma_20.fillna(close_series).values
                df.loc[mask, 'bb_lower'] = (sma_20 - (std_20 * 2)).fillna(close_series).values
                df.loc[mask, 'bb_width'] = (std_20 * 4).fillna(0).values
                
                bb_width = (std_20 * 4).fillna(1)
                bb_width = bb_width.where(bb_width > 0, 1)
                df.loc[mask, 'bb_position'] = ((close_series - (sma_20 - (std_20 * 2))) / bb_width).fillna(0.5).values
                
                # 5. Volume indicators
                df.loc[mask, 'volume_sma_20'] = volume_series.rolling(20).mean().fillna(volume_series.mean()).values
                df.loc[mask, 'ad'] = np.zeros(len(symbol_data))  # simplified
                df.loc[mask, 'obv'] = volume_series.cumsum().values  # simplified OBV
                
                # 6. ATR
                tr1 = high_series - low_series
                tr2 = abs(high_series - close_series.shift())
                tr3 = abs(low_series - close_series.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df.loc[mask, 'atr'] = tr.rolling(14).mean().fillna(0).values
                df.loc[mask, 'natr'] = (tr.rolling(14).mean() / (close_series + 1e-8) * 100).fillna(0).values
                
                # 7. Default values for others
                df.loc[mask, 'stoch_k'] = np.full(len(symbol_data), 50)
                df.loc[mask, 'stoch_d'] = np.full(len(symbol_data), 50)
                df.loc[mask, 'willr'] = np.full(len(symbol_data), -50)
                df.loc[mask, 'adx'] = np.full(len(symbol_data), 25)
                df.loc[mask, 'cci'] = np.zeros(len(symbol_data))
            
            # Volume ratio (always calculated manually)
            if 'volume_sma_20' in df.columns:
                volume_sma_20 = df.loc[mask, 'volume_sma_20'].values
                volume_sma_20 = np.where(volume_sma_20 == 0, 1, volume_sma_20)
                df.loc[mask, 'volume_ratio'] = symbol_data['volume'].values / volume_sma_20
            
            # Always calculate these manually
            close = symbol_data['close'].values
            high = symbol_data['high'].values
            low = symbol_data['low'].values
            open_price = symbol_data['open'].values
            
            # 8. Support/Resistance Levels
            rolling_max = pd.Series(close).rolling(window=20).max()
            rolling_min = pd.Series(close).rolling(window=20).min()
            df.loc[mask, 'resistance'] = rolling_max.fillna(close).values
            df.loc[mask, 'support'] = rolling_min.fillna(close).values
            df.loc[mask, 'distance_to_resistance'] = ((rolling_max - close) / (close + 1e-8)).fillna(0).values
            df.loc[mask, 'distance_to_support'] = ((close - rolling_min) / (close + 1e-8)).fillna(0).values
            
            # 9. Price Action Features
            df.loc[mask, 'price_change'] = (close / (open_price + 1e-8)) - 1
            df.loc[mask, 'high_low_ratio'] = high / (low + 1e-8)
            df.loc[mask, 'body_size'] = abs(close - open_price) / (open_price + 1e-8)
            df.loc[mask, 'upper_shadow'] = (high - np.maximum(close, open_price)) / (open_price + 1e-8)
            df.loc[mask, 'lower_shadow'] = (np.minimum(close, open_price) - low) / (open_price + 1e-8)
            
            # 10. Pattern recognition
            if TA_AVAILABLE:
                try:
                    # ใช้ TA-Lib candlestick patterns
                    df.loc[mask, 'cdl_doji'] = ta.CDLDOJI(open_price, high, low, close)
                    df.loc[mask, 'cdl_hammer'] = ta.CDLHAMMER(open_price, high, low, close)
                    df.loc[mask, 'cdl_engulfing'] = ta.CDLENGULFING(open_price, high, low, close)
                except:
                    # Fallback to simple pattern recognition
                    body = abs(close - open_price)
                    total_range = high - low + 1e-8
                    df.loc[mask, 'cdl_doji'] = (body / total_range < 0.1).astype(int)
                    df.loc[mask, 'cdl_hammer'] = np.zeros(len(symbol_data), dtype=int)
                    df.loc[mask, 'cdl_engulfing'] = np.zeros(len(symbol_data), dtype=int)
            else:
                # Simple pattern recognition
                body = abs(close - open_price)
                upper_shadow = high - np.maximum(close, open_price)
                lower_shadow = np.minimum(close, open_price) - low
                total_range = high - low + 1e-8
                
                df.loc[mask, 'cdl_doji'] = (body / total_range < 0.1).astype(int)
                is_hammer = (body / total_range < 0.3) & (lower_shadow > 2 * body) & (upper_shadow < body)
                df.loc[mask, 'cdl_hammer'] = is_hammer.astype(int)
                prev_body = np.roll(body, 1)
                df.loc[mask, 'cdl_engulfing'] = (body > 1.5 * prev_body).astype(int)
            
        except Exception as e:
            print(f"⚠️ Error calculating indicators for {symbol}: {str(e)}")
            continue
    
    # 11. Cross-asset features
    print("🔗 Adding cross-asset features...")
    btc_data = df[df['tic'] == 'BTC-USD']['close'].values if 'BTC-USD' in df['tic'].values else None
    
    if btc_data is not None and len(btc_data) > 0:
        for symbol in df['tic'].unique():
            if symbol != 'BTC-USD':
                mask = df['tic'] == symbol
                symbol_close = df[mask]['close'].values
                
                if len(symbol_close) == len(btc_data):
                    try:
                        correlation = pd.Series(symbol_close).rolling(window=30).corr(pd.Series(btc_data))
                        df.loc[mask, 'btc_correlation'] = correlation.fillna(0).values
                        
                        returns_symbol = pd.Series(symbol_close).pct_change()
                        returns_btc = pd.Series(btc_data).pct_change()
                        cov = returns_symbol.rolling(window=30).cov(returns_btc)
                        var = returns_btc.rolling(window=30).var()
                        beta = cov / (var + 1e-8)
                        df.loc[mask, 'btc_beta'] = beta.fillna(1).values
                    except:
                        df.loc[mask, 'btc_correlation'] = np.zeros(len(symbol_close))
                        df.loc[mask, 'btc_beta'] = np.ones(len(symbol_close))
    
    # 12. Market Regime Features
    print("📊 Adding market regime features...")
    for symbol in df['tic'].unique():
        mask = df['tic'] == symbol
        close = df[mask]['close'].values
        
        if len(close) > 50:
            try:
                sma_20 = pd.Series(close).rolling(window=20).mean()
                sma_50 = pd.Series(close).rolling(window=50).mean()
                df.loc[mask, 'trend_strength'] = ((sma_20 - sma_50) / (sma_50 + 1e-8)).fillna(0).values
                
                returns = pd.Series(close).pct_change()
                vol_20 = returns.rolling(window=20).std()
                vol_50 = returns.rolling(window=50).std()
                df.loc[mask, 'volatility_regime'] = (vol_20 / (vol_50 + 1e-8)).fillna(1).values
            except:
                df.loc[mask, 'trend_strength'] = np.zeros(len(close))
                df.loc[mask, 'volatility_regime'] = np.ones(len(close))
    
    # Normalize ข้อมูล
    print("🔄 Normalizing features...")
    feature_columns = [col for col in df.columns if col not in ['tic', 'timestamp', 'date']]
    
    scaler = RobustScaler()
    
    for symbol in df['tic'].unique():
        mask = df['tic'] == symbol
        symbol_data = df[mask][feature_columns]
        
        if len(symbol_data) > 0:
            try:
                scaled_data = scaler.fit_transform(symbol_data.fillna(0))
                df.loc[mask, feature_columns] = scaled_data
            except Exception as e:
                print(f"⚠️ Error normalizing {symbol}: {str(e)}")
    
    # แทนที่ค่า inf และ nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # เพิ่ม outlier detection
    print("🔍 Detecting outliers...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    
    for symbol in df['tic'].unique():
        mask = df['tic'] == symbol
        symbol_data = df[mask][feature_columns]
        
        if len(symbol_data) > 50:
            try:
                outlier_labels = isolation_forest.fit_predict(symbol_data)
                df.loc[mask, 'is_outlier'] = (outlier_labels == -1).astype(int)
            except:
                df.loc[mask, 'is_outlier'] = 0
    
    print(f"✅ Added {len(feature_columns)} advanced features")
    print(f"📊 Final data shape: {df.shape}")
    
    return df

class AdvancedTradingEnv(StockTradingEnv):
    """
    Advanced Trading Environment พร้อม reward function ที่ปรับปรุงแล้ว
    """
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.previous_portfolio_value = self.initial_amount
        
    def step(self, actions):
        """Override step function เพื่อเพิ่ม advanced reward calculation"""
        # เรียก parent step function
        state, reward, done, info = super().step(actions)
        
        # คำนวณ advanced reward
        current_portfolio_value = self.asset_memory[-1]
        
        # 1. Return-based reward
        portfolio_return = (current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        
        # 2. Risk-adjusted reward (Sharpe-like)
        if len(self.asset_memory) > 30:
            returns = np.diff(self.asset_memory[-30:]) / self.asset_memory[-31:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if np.std(returns) > 0 else 0
            risk_adjusted_reward = sharpe_ratio * 0.1
        else:
            risk_adjusted_reward = 0
        
        # 3. Drawdown penalty
        peak_value = max(self.asset_memory)
        drawdown = (peak_value - current_portfolio_value) / peak_value if peak_value > 0 else 0
        drawdown_penalty = -drawdown * 0.5 if drawdown > 0.1 else 0  # penalty เมื่อ drawdown > 10%
        
        # 4. Diversification reward
        total_holdings = sum(abs(holding) for holding in self.state_memory[-1][self.stock_dim+1:2*self.stock_dim+1])
        num_assets_held = sum(1 for holding in self.state_memory[-1][self.stock_dim+1:2*self.stock_dim+1] if abs(holding) > 0.01)
        diversification_reward = 0.01 * min(num_assets_held / len(CRYPTO_SYMBOLS), 1.0) if total_holdings > 0 else 0
        
        # 5. Transaction cost consideration
        if len(self.actions_memory) > 1:
            prev_actions = self.actions_memory[-2]
            current_actions = actions
            action_changes = sum(abs(curr - prev) for curr, prev in zip(current_actions, prev_actions))
            transaction_penalty = -action_changes * 0.001  # penalty สำหรับการเปลี่ยน position บ่อย
        else:
            transaction_penalty = 0
        
        # รวม reward ทั้งหมด
        advanced_reward = (portfolio_return * 100 +  # scale up return reward
                          risk_adjusted_reward + 
                          drawdown_penalty + 
                          diversification_reward + 
                          transaction_penalty)
        
        self.previous_portfolio_value = current_portfolio_value
        
        return state, advanced_reward, done, info

def create_advanced_environment(df):
    """
    สร้าง advanced trading environment
    """
    print("🏛️ Creating advanced trading environment...")
    
    # แปลงชื่อคอลัมน์
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    # เรียงข้อมูลและเตรียม timestamp
    df = df.sort_values(['timestamp', 'tic']).reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # หาเฉพาะ symbols ที่มีข้อมูลครบ
    symbol_counts = df['tic'].value_counts()
    min_data_points = symbol_counts.quantile(0.8)  # ใช้ symbols ที่มีข้อมูลอย่างน้อย 80% ของ max
    valid_symbols = symbol_counts[symbol_counts >= min_data_points].index.tolist()
    df = df[df['tic'].isin(valid_symbols)].reset_index(drop=True)
    
    print(f"📊 Using {len(valid_symbols)} symbols: {valid_symbols}")
    
    # แบ่งข้อมูลแบบเรียบง่าย train/test = 80/20 (เหมือน main.py ที่ทำงานได้)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    
    # สร้าง validation จาก 20% ของ training data
    val_size = int(len(train_df) * 0.2)
    val_df = train_df.iloc[-val_size:].copy().reset_index(drop=True)
    train_df = train_df.iloc[:-val_size].copy().reset_index(drop=True)
    
    print(f"📚 Training: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"📝 Validation: {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"🧪 Testing: {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # กำหนด indicators ที่ใช้ (ลดจำนวนให้เหมือน main.py ที่ทำงานได้)
    indicators = [
        'sma_20', 'ema_20', 'rsi_14', 
        'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_upper', 'bb_lower',
        'volume_sma_20', 'volume_ratio'
    ]
    
    # กรองเฉพาะ indicators ที่มีในข้อมูล
    available_indicators = [ind for ind in indicators if ind in df.columns]
    print(f"📈 Using {len(available_indicators)} indicators")
    
    # เตรียมข้อมูลให้เป็นรูปแบบที่ FinRL ต้องการ (แบบปลอดภัยจาก numpy AttributeError)
    # ใช้วิธีการเดียวกับ main.py ที่ทำงานได้ดี
    def prepare_data_for_finrl(data):
        """แปลงข้อมูลให้เข้ากับ FinRL โดยหลีกเลี่ยง numpy scalar AttributeError"""
        data = data.copy()
        
        # แปลง timestamp และ date เป็น string (เหมือน main.py)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')  # ใช้ string แทน date object
        
        # เรียงข้อมูลตามวันที่และ symbol
        data = data.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # แปลงข้อมูลตัวเลขให้เป็น pandas Series ชัดเจน (เหมือน main.py)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume'] + available_indicators
        
        for col in numeric_columns:
            if col in data.columns:
                # แปลงเป็น pandas Series ชัดเจน และแทนที่ NaN ด้วย 0 (เหมือน main.py)
                data[col] = pd.Series(data[col]).astype('float64').fillna(0.0)
        
        # ตรวจสอบและแก้ไข inf values (เหมือน main.py)
        data = data.replace([np.inf, -np.inf], 0.0)
        
        # ตรวจสอบให้แน่ใจว่าไม่มี NaN (เหมือน main.py)
        data = data.fillna(0.0)
        
        # ตรวจสอบให้แน่ใจว่ามีคอลัมน์ที่จำเป็น
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"⚠️ Missing required column: {col}")
        
        return data
    
    train_df = prepare_data_for_finrl(train_df)
    val_df = prepare_data_for_finrl(val_df)
    test_df = prepare_data_for_finrl(test_df)
    
    # สร้าง environment kwargs
    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * len(valid_symbols),
        "buy_cost_pct": [TRANSACTION_COST_PCT] * len(valid_symbols),
        "sell_cost_pct": [TRANSACTION_COST_PCT] * len(valid_symbols),
        "state_space": 1 + 2 * len(valid_symbols) + len(valid_symbols) * len(available_indicators),
        "stock_dim": len(valid_symbols),
        "tech_indicator_list": available_indicators,
        "action_space": len(valid_symbols),
        "reward_scaling": 1e-2,
        "print_verbosity": 1
    }
    
    # ตรวจสอบข้อมูลก่อนส่งให้ FinRL (เพิ่มการป้องกันเพิ่มเติม)
    def validate_finrl_dataframe(df_to_check, name=""):
        """ตรวจสอบและแก้ไขปัญหา DataFrame สำหรับ FinRL"""
        print(f"🔍 Validating {name} DataFrame...")
        
        # ตรวจสอบว่าเป็น DataFrame
        if not isinstance(df_to_check, pd.DataFrame):
            raise ValueError(f"{name} is not a DataFrame")
        
        # ตรวจสอบว่ามีข้อมูล
        if len(df_to_check) == 0:
            raise ValueError(f"{name} is empty")
        
        # ตรวจสอบ required columns
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df_to_check.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        
        # ตรวจสอบว่า close column เป็น Series
        if not isinstance(df_to_check['close'], pd.Series):
            print(f"⚠️ Converting {name} close to Series")
            df_to_check['close'] = pd.Series(df_to_check['close'].values)
        
        # ตรวจสอบ data types อื่นๆ
        for col in required_cols[2:]:  # skip date และ tic
            if not pd.api.types.is_numeric_dtype(df_to_check[col]):
                print(f"⚠️ Converting {name} {col} to numeric")
                df_to_check[col] = pd.to_numeric(df_to_check[col], errors='coerce').fillna(0.0)
        
        print(f"✅ {name} DataFrame validated")
        return df_to_check
    
    # Validate data frames
    train_df = validate_finrl_dataframe(train_df, "Training")
    val_df = validate_finrl_dataframe(val_df, "Validation") 
    test_df = validate_finrl_dataframe(test_df, "Testing")
    
    # สร้าง environments
    try:
        train_env = AdvancedTradingEnv(df=train_df, **env_kwargs)
        val_env = AdvancedTradingEnv(df=val_df, **env_kwargs)
        test_env = AdvancedTradingEnv(df=test_df, **env_kwargs)
        
        print("✅ Advanced environments created successfully")
        
        return train_env, val_env, test_env, train_df, val_df, test_df, valid_symbols
        
    except Exception as e:
        print(f"❌ Error creating environments: {str(e)}")
        print("🔧 Trying with standard StockTradingEnv...")
        
        # ลองใช้ StockTradingEnv มาตรฐานแทน
        train_env = StockTradingEnv(df=train_df, **env_kwargs)
        val_env = StockTradingEnv(df=val_df, **env_kwargs)
        test_env = StockTradingEnv(df=test_df, **env_kwargs)
        
        print("✅ Standard environments created successfully")
        
        return train_env, val_env, test_env, train_df, val_df, test_df, valid_symbols

def train_ensemble_agents(train_env, val_env):
    """
    เทรน ensemble ของหลาย models
    """
    print("🤖 Training ensemble of advanced agents...")
    
    device = setup_device()
    agent = DRLAgent(env=train_env)
    
    # กำหนด models และ parameters
    models_config = {
        'ppo_conservative': {
            'learning_rate': 1e-5,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.1,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'device': device
        },
        'ppo_aggressive': {
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 128,
            'n_epochs': 4,
            'gamma': 0.95,
            'gae_lambda': 0.9,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 1.0,
            'device': device
        },
        'ppo_balanced': {
            'learning_rate': 1e-4,
            'n_steps': 1536,
            'batch_size': 96,
            'n_epochs': 6,
            'gamma': 0.98,
            'gae_lambda': 0.92,
            'clip_range': 0.15,
            'ent_coef': 0.02,
            'vf_coef': 0.5,
            'max_grad_norm': 0.75,
            'device': device
        }
    }
    
    trained_models = {}
    
    for model_name, params in models_config.items():
        print(f"\n🧠 Training {model_name}...")
        print(f"Parameters: {params}")
        
        try:
            model = agent.get_model("ppo", model_kwargs=params)
            
            # เทรน model
            trained_model = agent.train_model(
                model=model,
                tb_log_name=f"advanced_crypto_{model_name}",
                total_timesteps=200000
            )
            
            # บันทึก model
            model_path = os.path.join(ADVANCED_MODEL_DIR, f"advanced_crypto_{model_name}")
            trained_model.save(model_path)
            print(f"💾 {model_name} saved to {model_path}")
            
            # ทดสอบบน validation set
            print(f"📊 Validating {model_name}...")
            df_val_account, _ = DRLAgent.DRL_prediction(
                model=trained_model,
                environment=val_env
            )
            
            val_return = (df_val_account['account_value'].iloc[-1] / INITIAL_AMOUNT - 1) * 100
            print(f"📈 {model_name} validation return: {val_return:.2f}%")
            
            trained_models[model_name] = {
                'model': trained_model,
                'val_return': val_return,
                'path': model_path
            }
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    print(f"\n✅ Trained {len(trained_models)} models successfully")
    
    # เลือก best model
    if trained_models:
        best_model_name = max(trained_models.keys(), key=lambda x: trained_models[x]['val_return'])
        best_model = trained_models[best_model_name]['model']
        print(f"🏆 Best model: {best_model_name} (validation return: {trained_models[best_model_name]['val_return']:.2f}%)")
        
        return best_model, trained_models
    else:
        raise ValueError("ไม่สามารถเทรน model ใดได้สำเร็จ")

def test_ensemble_agents(trained_models, test_env):
    """
    ทดสอบ ensemble และสร้าง prediction แบบ weighted average
    """
    print("📊 Testing ensemble agents...")
    
    predictions = {}
    account_values = {}
    
    # ทดสอบแต่ละ model
    for model_name, model_info in trained_models.items():
        print(f"🧪 Testing {model_name}...")
        
        df_account, df_actions = DRLAgent.DRL_prediction(
            model=model_info['model'],
            environment=test_env
        )
        
        predictions[model_name] = df_actions
        account_values[model_name] = df_account
        
        final_return = (df_account['account_value'].iloc[-1] / INITIAL_AMOUNT - 1) * 100
        print(f"📈 {model_name} test return: {final_return:.2f}%")
    
    # สร้าง ensemble prediction (weighted by validation performance)
    print("🔮 Creating ensemble prediction...")
    
    # คำนวณ weights จาก validation performance
    val_returns = [info['val_return'] for info in trained_models.values()]
    min_return = min(val_returns)
    adjusted_returns = [ret - min_return + 1 for ret in val_returns]  # shift to positive
    total_weight = sum(adjusted_returns)
    weights = [ret / total_weight for ret in adjusted_returns]
    
    print(f"📊 Ensemble weights: {dict(zip(trained_models.keys(), weights))}")
    
    # เลือก best single model
    best_model_name = max(trained_models.keys(), key=lambda x: trained_models[x]['val_return'])
    best_account_value = account_values[best_model_name]
    
    return best_account_value, predictions, account_values, best_model_name

def advanced_analysis(account_values, test_df, valid_symbols):
    """
    วิเคราะห์ผลลัพธ์ขั้นสูง
    """
    print("📈 Advanced result analysis...")
    
    # คำนวณ performance metrics
    final_value = account_values['account_value'].iloc[-1]
    initial_value = INITIAL_AMOUNT
    total_return = (final_value - initial_value) / initial_value * 100
    
    # คำนวณ buy and hold benchmark
    btc_data = test_df[test_df['tic'] == 'BTC-USD']
    if len(btc_data) > 0:
        btc_return = (btc_data['close'].iloc[-1] / btc_data['close'].iloc[0] - 1) * 100
    else:
        btc_return = 0
    
    # คำนวณ equal weight portfolio return
    equal_weight_returns = []
    for symbol in valid_symbols:
        symbol_data = test_df[test_df['tic'] == symbol]
        if len(symbol_data) > 0:
            symbol_return = symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1
            equal_weight_returns.append(symbol_return)
    
    equal_weight_return = (sum(equal_weight_returns) / len(equal_weight_returns)) * 100 if equal_weight_returns else 0
    
    # คำนวณ risk metrics
    portfolio_values = account_values['account_value'].values
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Sharpe Ratio (assume risk-free rate = 0)
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)  # annualized
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252) * 100  # annualized
    
    # Calmar Ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
    
    print(f"\n📊 ADVANCED RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"💰 Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"💰 Final Portfolio Value: ${final_value:,.2f}")
    print(f"📈 Agent Total Return: {total_return:.2f}%")
    print(f"📈 BTC Buy & Hold Return: {btc_return:.2f}%")
    print(f"📈 Equal Weight Portfolio Return: {equal_weight_return:.2f}%")
    print(f"🎯 Alpha vs BTC: {total_return - btc_return:.2f}%")
    print(f"🎯 Alpha vs Equal Weight: {total_return - equal_weight_return:.2f}%")
    print(f"📊 Sharpe Ratio (Annualized): {sharpe_ratio_annualized:.3f}")
    print(f"📉 Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"📊 Volatility (Annualized): {volatility:.2f}%")
    print(f"📊 Calmar Ratio: {calmar_ratio:.3f}")
    print(f"{'='*60}")
    
    # สร้างกราฟ
    create_advanced_plots(account_values, test_df, valid_symbols)
    
    return {
        'agent_return': total_return,
        'btc_return': btc_return,
        'equal_weight_return': equal_weight_return,
        'alpha_btc': total_return - btc_return,
        'alpha_equal_weight': total_return - equal_weight_return,
        'sharpe_ratio': sharpe_ratio_annualized,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'calmar_ratio': calmar_ratio,
        'final_value': final_value
    }

def create_advanced_plots(account_values, test_df, valid_symbols):
    """
    สร้างกราฟการวิเคราะห์ขั้นสูง
    """
    print("📊 Creating advanced performance plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Portfolio value over time
    dates = pd.to_datetime(test_df['timestamp'].unique())
    portfolio_values = account_values['account_value'].values
    
    axes[0,0].plot(dates, portfolio_values, label='Agent Portfolio', linewidth=2, color='blue')
    axes[0,0].axhline(y=INITIAL_AMOUNT, color='red', linestyle='--', label='Initial Value')
    axes[0,0].set_title('Portfolio Value Over Time', fontweight='bold')
    axes[0,0].set_ylabel('Portfolio Value ($)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    
    axes[0,1].fill_between(dates, drawdown, 0, alpha=0.3, color='red')
    axes[0,1].set_title('Portfolio Drawdown', fontweight='bold')
    axes[0,1].set_ylabel('Drawdown (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling Sharpe Ratio
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    rolling_sharpe = pd.Series(returns).rolling(window=30).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0
    ) * np.sqrt(252)
    
    axes[1,0].plot(dates[1:], rolling_sharpe, linewidth=2, color='green')
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,0].set_title('Rolling Sharpe Ratio (30-day)', fontweight='bold')
    axes[1,0].set_ylabel('Sharpe Ratio')
    axes[1,0].set_xlabel('Date')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Asset comparison
    # เปรียบเทียบกับ top assets
    comparison_data = []
    comparison_labels = []
    
    for symbol in valid_symbols[:5]:  # top 5 assets
        symbol_data = test_df[test_df['tic'] == symbol]
        if len(symbol_data) > 0:
            symbol_return = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100
            comparison_data.append(symbol_return)
            comparison_labels.append(symbol.replace('-USD', ''))
    
    agent_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    comparison_data.append(agent_return)
    comparison_labels.append('Agent')
    
    colors = ['skyblue'] * len(comparison_data[:-1]) + ['orange']
    bars = axes[1,1].bar(comparison_labels, comparison_data, color=colors)
    axes[1,1].set_title('Return Comparison', fontweight='bold')
    axes[1,1].set_ylabel('Return (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # เพิ่ม value labels บน bars
    for bar, value in zip(bars, comparison_data):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ADVANCED_MODEL_DIR, 'advanced_performance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Advanced performance plots saved and displayed")

def main():
    """
    Main function สำหรับ Advanced Crypto Agent
    """
    print("🚀 Starting Advanced Crypto Agent with Enhanced Features")
    print("="*70)
    
    try:
        device = setup_device()
        
        # Step 1: Download enhanced data
        df = download_crypto_data_advanced(force_download=False)
        
        # Step 2: Add advanced technical indicators
        df = add_advanced_technical_indicators(df)
        
        # Step 3: Create advanced environments
        train_env, val_env, test_env, train_df, val_df, test_df, valid_symbols = create_advanced_environment(df)
        
        # Step 4: Train ensemble agents
        best_model, trained_models = train_ensemble_agents(train_env, val_env)
        
        # Step 5: Test ensemble
        best_account_value, predictions, all_account_values, best_model_name = test_ensemble_agents(trained_models, test_env)
        
        # Step 6: Advanced analysis
        results = advanced_analysis(best_account_value, test_df, valid_symbols)
        
        print(f"\n🎉 Advanced Crypto Agent completed successfully!")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📈 Agent achieved {results['agent_return']:.2f}% return")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
        
        if results['alpha_btc'] > 0:
            print(f"🎯 Excellent! Agent outperformed BTC by {results['alpha_btc']:.2f}%")
        
        if results['alpha_equal_weight'] > 0:
            print(f"🎯 Great! Agent outperformed equal-weight portfolio by {results['alpha_equal_weight']:.2f}%")
        
        # เก็บ summary
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_model': best_model_name,
            'results': results,
            'valid_symbols': valid_symbols
        }
        
        # บันทึก summary
        import json
        with open(os.path.join(ADVANCED_MODEL_DIR, 'advanced_agent_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Summary saved to {ADVANCED_MODEL_DIR}/advanced_agent_summary.json")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        print(f"🔍 Full error trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 