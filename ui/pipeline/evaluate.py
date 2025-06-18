import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import torch
from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import shutil

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR, INITIAL_AMOUNT, CRYPTO_SYMBOLS, DATA_DIR, TRANSACTION_COST_PCT, HMAX
from ui.pipeline.train import add_technical_indicators, prepare_data_for_training

def organize_all_models():
    """จัดระเบียบโมเดลทั้งหมดให้มารวมไว้ใน root models directory"""
    try:
        # ค้นหา model directories ทั้งหมดยกเว้น root
        all_model_dirs = find_all_model_directories()
        model_dirs = [(name, path) for name, path in all_model_dirs if path != MODEL_DIR]
        
        moved_count = 0
        skipped_count = 0
        
        for dir_name, dir_path in model_dirs:
            if os.path.exists(dir_path):
                model_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
                
                for model_file in model_files:
                    source_path = os.path.join(dir_path, model_file)
                    dest_path = os.path.join(MODEL_DIR, model_file)
                    
                    if not os.path.exists(dest_path):
                        shutil.copy2(source_path, dest_path)
                        moved_count += 1
                        st.success(f"✅ คัดลอก {model_file} จาก {dir_name}")
                    else:
                        skipped_count += 1
                        st.info(f"⏭️ ข้าม {model_file} (มีอยู่แล้ว)")
        
        if moved_count > 0:
            st.success(f"🎉 จัดระเบียบโมเดลเสร็จสิ้น! คัดลอกไฟล์ {moved_count} ไฟล์")
            if skipped_count > 0:
                st.info(f"📋 ข้ามไฟล์ที่มีอยู่แล้ว {skipped_count} ไฟล์")
            try:
                st.rerun()
            except:
                st.info("กรุณารีเฟรชหน้าเพื่อดูการเปลี่ยนแปลง")
        else:
            st.info("📂 โมเดลทั้งหมดอยู่ใน root models directory แล้ว")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการจัดระเบียบโมเดล: {str(e)}")

def find_all_model_directories():
    """ค้นหา directory ทั้งหมดที่มีไฟล์ .zip (โมเดลที่เทรนแล้ว)"""
    model_dirs = []
    root_path = Path(".")
    
    # ค้นหาใน directory หลักก่อน
    if os.path.exists(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
        if model_files:
            model_dirs.append(('Root Models (Production)', MODEL_DIR))
    
    # ค้นหาใน subdirectories ทั้งหมด
    for path in root_path.rglob("*.zip"):
        dir_path = str(path.parent)
        # ข้าม directory ที่เพิ่มไปแล้ว
        if any(existing_path == dir_path for _, existing_path in model_dirs):
            continue
            
        # ตรวจสอบว่ามีไฟล์ .zip อย่างน้อย 1 ไฟล์
        zip_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
        if zip_files:
            # สร้างชื่อที่อ่านง่าย พร้อมระบุประเภท
            relative_path = os.path.relpath(dir_path, ".")
            
            # กำหนดประเภทตาม path
            if 'notebooks/models' in relative_path:
                dir_name = f"{relative_path.replace(os.sep, '/')} (Training Output)"
            elif 'agents' in relative_path:
                dir_name = f"{relative_path.replace(os.sep, '/')} (Configs - Skip)"
                continue  # ข้าม agents directory เพราะเป็น config files
            else:
                dir_name = relative_path.replace(os.sep, '/') if relative_path != '.' else 'Root'
            
            model_dirs.append((dir_name, dir_path))
    
    return model_dirs

def find_all_data_files():
    """ค้นหาไฟล์ข้อมูล .csv ทั้งหมดในโปรเจกต์"""
    data_files = []
    root_path = Path(".")
    
    # ค้นหาไฟล์ .csv ทั้งหมด
    for path in root_path.rglob("*.csv"):
        # ข้าม hidden directories และ temporary files
        if any(part.startswith('.') for part in path.parts):
            continue
            
        relative_path = str(path.relative_to("."))
        file_info = {
            'name': path.name,
            'path': str(path.parent),
            'full_path': str(path),
            'relative_path': relative_path,
            'size': path.stat().st_size if path.exists() else 0
        }
        data_files.append(file_info)
    
    # เรียงตาม size (ใหญ่ไปเล็ก) และชื่อไฟล์
    data_files.sort(key=lambda x: (-x['size'], x['name']))
    
    return data_files

def load_model_list():
    """Load list of available models from all discovered directories"""
    model_dirs = find_all_model_directories()
    all_models = {}
    
    for dir_name, dir_path in model_dirs:
        if os.path.exists(dir_path):
            # ค้นหาไฟล์ .zip ทั้งหมดที่เป็นโมเดล
            model_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
            
            for model_file in model_files:
                model_name = model_file.replace('.zip', '')
                # เก็บข้อมูลโมเดลพร้อมที่อยู่
                all_models[f"{model_name} ({dir_name})"] = {
                    'name': model_name,
                    'file': model_file,
                    'path': dir_path,
                    'full_path': os.path.join(dir_path, model_file)
                }
    
    return all_models

def load_model(model_info_or_path):
    """Load a trained model"""
    try:
        # ถ้าเป็น dict (ข้อมูลโมเดลจาก load_model_list)
        if isinstance(model_info_or_path, dict):
            model_path = model_info_or_path['full_path']
            model_name = model_info_or_path['name']
        # ถ้าเป็น string (path โดยตรง)
        elif isinstance(model_info_or_path, str):
            if os.path.exists(model_info_or_path):
                model_path = model_info_or_path
                model_name = os.path.basename(model_info_or_path)
            else:
                # เพิ่ม .zip ถ้ายังไม่มี
                if not model_info_or_path.endswith('.zip'):
                    model_info_or_path = f"{model_info_or_path}.zip"
                    
                model_path = os.path.join(MODEL_DIR, model_info_or_path)
                model_name = model_info_or_path
        else:
            st.error("รูปแบบข้อมูลโมเดลไม่ถูกต้อง")
            return None
            
        if not os.path.exists(model_path):
            st.error(f"ไม่พบไฟล์โมเดลที่ {model_path}")
            return None
            
        # โหลดโมเดลด้วย device ที่เหมาะสม
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PPO.load(model_path, device=device)
        
        st.success(f"โหลดโมเดล {model_name} สำเร็จ!")
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None

def create_evaluation_env(df, initial_amount=INITIAL_AMOUNT):
    """สร้าง environment สำหรับการประเมิน"""
    try:
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if df is None or len(df) == 0:
            st.error("ไม่มีข้อมูลสำหรับการสร้าง environment")
            return None
            
        # ตรวจสอบข้อมูลขั้นต่ำสำหรับการประเมิน
        if len(df) < 10:
            st.warning(f"⚠️ ข้อมูลมีจำนวนน้อย ({len(df)} แถว) อาจส่งผลต่อความแม่นยำของการประเมิน")
            st.info("แนะนำให้ใช้ข้อมูลอย่างน้อย 30-50 แถวเพื่อการประเมินที่มีประสิทธิภาพ")
            
        # ตรวจสอบและแปลงคอลัมน์วันที่
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        else:
            st.error("ไม่พบคอลัมน์ date หรือ timestamp ในข้อมูล")
            return None
            
        # ตรวจสอบและแปลงคอลัมน์ symbol
        if 'symbol' in df.columns:
            df['tic'] = df['symbol']
        elif 'tic' not in df.columns:
            st.error("ไม่พบคอลัมน์ symbol หรือ tic ในข้อมูล")
            return None
            
        # ตรวจสอบคอลัมน์ราคาและ volume
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col not in df.columns:
                st.error(f"ไม่พบคอลัมน์ {col} ในข้อมูล")
                return None
            # แปลงเป็น float
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # ตรวจสอบค่า NaN
        if df[price_cols].isna().any().any():
            st.warning("พบค่า NaN ในข้อมูลราคา จะทำการแทนที่ด้วยค่าเฉลี่ย")
            df[price_cols] = df[price_cols].fillna(df[price_cols].mean())
            
        # กำหนด indicators ที่ใช้
        indicators = [
            'sma_20', 'ema_20', 'rsi_14', 
            'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
            'volume_sma_20', 'volume_ratio'
        ]
        
        # ตรวจสอบว่ามี indicators ครบหรือไม่
        missing_indicators = [ind for ind in indicators if ind not in df.columns]
        if missing_indicators:
            st.warning(f"กำลังคำนวณ indicators ที่ขาดหาย: {', '.join(missing_indicators)}")
            df = add_technical_indicators(df)
            
        # เรียงข้อมูลตามวันที่และ tic
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # สร้าง environment
        env_kwargs = {
            "hmax": HMAX,
            "initial_amount": initial_amount,
            "num_stock_shares": [0] * len(df['tic'].unique()),
            "buy_cost_pct": [TRANSACTION_COST_PCT] * len(df['tic'].unique()),
            "sell_cost_pct": [TRANSACTION_COST_PCT] * len(df['tic'].unique()),
            "state_space": 1 + 2 * len(df['tic'].unique()) + len(df['tic'].unique()) * len(indicators),
            "stock_dim": len(df['tic'].unique()),
            "tech_indicator_list": indicators,
            "action_space": len(df['tic'].unique()),
            "reward_scaling": 1e-3,
            "print_verbosity": 1
        }
        
        # ตรวจสอบข้อมูลก่อนสร้าง environment (แสดงเฉพาะเมื่อจำนวนน้อย)
        if len(df) <= 25:
            st.info(f"""
            📊 Environment สำหรับการประเมิน:
            - จำนวนข้อมูล: {len(df):,} แถว
            - สกุลเงิน: {', '.join(df['tic'].unique())}
            - ช่วงเวลา: {df['date'].min()} ถึง {df['date'].max()}
            - จำนวน indicators: {len(indicators)}
            """)
        
        return StockTradingEnv(df=df, **env_kwargs)
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้าง environment: {str(e)}")
        st.error("กรุณาตรวจสอบข้อมูลและลองใหม่อีกครั้ง")
        return None

def get_model_versions(model_info):
    """ดึงรายการเวอร์ชั่นทั้งหมดของโมเดล"""
    try:
        if isinstance(model_info, dict):
            model_name = model_info['name']
            search_dir = model_info['path']
        else:
            model_name = model_info
            search_dir = MODEL_DIR
            
        # ลบ .zip ถ้ามี
        if model_name.endswith('.zip'):
            model_name = model_name.replace('.zip', '')
            
        # ค้นหาไฟล์ที่มีชื่อขึ้นต้นเหมือนกัน โดยค้นหาทั้ง pattern _v และไฟล์เดี่ยว
        existing_files = []
        if os.path.exists(search_dir):
            all_files = [f for f in os.listdir(search_dir) if f.endswith('.zip')]
            
            for file in all_files:
                file_base = file.replace('.zip', '')
                if file_base == model_name or file_base.startswith(f"{model_name}_v"):
                    existing_files.append((file, search_dir))
        
        # ค้นหาในโฟลเดอร์อื่นๆ ด้วย - ใช้การค้นหาแบบอัตโนมัติ
        discovered_dirs = find_all_model_directories()
        
        for dir_name, dir_path in discovered_dirs:
            if dir_path != search_dir and os.path.exists(dir_path):
                all_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
                
                for file in all_files:
                    file_base = file.replace('.zip', '')
                    if file_base == model_name or file_base.startswith(f"{model_name}_v"):
                        existing_files.append((file, dir_path))
        
        if not existing_files:
            return []
        
        # ดึงเลขเวอร์ชั่นและเรียงลำดับ
        versions = []
        for file, file_dir in existing_files:
            try:
                file_base = file.replace('.zip', '')
                if '_v' in file_base:
                    version = int(file_base.split('_v')[1])
                else:
                    version = 1  # ไฟล์เดี่ยวถือเป็น version 1
                    
                versions.append({
                    'version': version,
                    'file': file,
                    'path': file_dir,
                    'full_path': os.path.join(file_dir, file),
                    'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(file_dir, file)))
                })
            except Exception as e:
                # สำหรับไฟล์ที่ไม่มี version ให้ใช้ version 1
                versions.append({
                    'version': 1,
                    'file': file,
                    'path': file_dir,
                    'full_path': os.path.join(file_dir, file),
                    'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(file_dir, file)))
                })
                continue
        
        return sorted(versions, key=lambda x: x['version'])
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการค้นหาเวอร์ชั่นโมเดล: {str(e)}")
        return []

def plot_version_comparison(version_results):
    """สร้างกราฟเปรียบเทียบประสิทธิภาพระหว่างเวอร์ชั่น"""
    fig = go.Figure()
    
    # สีสำหรับแต่ละเวอร์ชั่น
    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC', '#20B2AA']
    
    # เพิ่มเส้นสำหรับแต่ละเวอร์ชั่น
    for i, (version, result) in enumerate(version_results.items()):
        fig.add_trace(go.Scatter(
            x=result['dates'],
            y=result['account_values'],
            name=f"Version {version}",
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # เพิ่มเส้น benchmark
    fig.add_trace(go.Scatter(
        x=version_results[list(version_results.keys())[0]]['dates'],
        y=version_results[list(version_results.keys())[0]]['benchmark_values'],
        name="Buy & Hold",
        line=dict(color="#808080", width=2, dash="dash")
    ))
    
    # เพิ่มเส้นเงินเริ่มต้น
    fig.add_trace(go.Scatter(
        x=version_results[list(version_results.keys())[0]]['dates'],
        y=[INITIAL_AMOUNT] * len(version_results[list(version_results.keys())[0]]['dates']),
        name="Initial Investment",
        line=dict(color="#000000", width=1, dash="dot")
    ))
    
    # อัพเดท layout
    fig.update_layout(
        title="Version Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def calculate_metrics(account_values, benchmark_values, initial_amount):
    """Calculate performance metrics"""
    try:
        # Calculate returns
        total_return = (account_values[-1] - initial_amount) / initial_amount * 100
        benchmark_return = (benchmark_values[-1] - initial_amount) / initial_amount * 100
        
        # Calculate daily returns
        daily_returns = pd.Series(account_values).pct_change().dropna()
        benchmark_daily_returns = pd.Series(benchmark_values).pct_change().dropna()
        
        # Calculate metrics
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)  # Annualized
        max_drawdown = ((pd.Series(account_values).cummax() - account_values) / 
                       pd.Series(account_values).cummax()).max() * 100
        
        return {
            "Total Return (%)": f"{total_return:.2f}%",
            "vs. Buy & Hold": f"{total_return - benchmark_return:+.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown (%)": f"{max_drawdown:.2f}%",
            "Final Portfolio Value": f"${account_values[-1]:,.2f}"
        }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ metrics: {str(e)}")
        return None

def evaluate_agent_ui():
    """UI for evaluating agent performance"""
    st.header("📈 Evaluate Agent Performance")
    
    # แสดงข้อมูลสรุปโมเดลทั้งหมด
    with st.expander("📊 Models Overview", expanded=False):
        models_dict = load_model_list()
        
        if not models_dict:
            st.info("ไม่พบโมเดลในระบบ")
        else:
            # จัดกลุ่มตามโฟลเดอร์
            model_by_location = {}
            total_size = 0
            
            for display_name, model_info in models_dict.items():
                location = model_info['path']
                if location not in model_by_location:
                    model_by_location[location] = []
                model_by_location[location].append(model_info)
                total_size += os.path.getsize(model_info['full_path'])
            
            # แสดงสถิติรวม
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 Total Models", len(models_dict))
            with col2:
                st.metric("📂 Directories", len(model_by_location))
            with col3:
                st.metric("💾 Total Size", f"{total_size/(1024*1024):.1f} MB")
            
            st.markdown("---")
            
                         # แสดงรายละเอียดแต่ละโฟลเดอร์
            for location, models in model_by_location.items():
                # กำหนดประเภทโฟลเดอร์
                if location == MODEL_DIR:
                    location_display = "📁 Root Models (Production Ready)"
                    folder_icon = "🚀"
                elif 'notebooks/models' in location:
                    location_display = f"📁 {location.replace(os.sep, '/')} (Training Output)"
                    folder_icon = "🧪"
                else:
                    location_display = f"📁 {location.replace(os.sep, '/')}"
                    folder_icon = "📂"
                
                folder_size = sum(os.path.getsize(m['full_path']) for m in models) / (1024*1024)
                
                st.write(f"**{location_display}:** {len(models)} models ({folder_size:.1f} MB)")
                for model in models:
                    file_size = os.path.getsize(model['full_path']) / (1024*1024)  # MB
                    modified_time = datetime.fromtimestamp(os.path.getmtime(model['full_path']))
                    
                    # กำหนดไอคอนตามขนาดไฟล์
                    if file_size > 1.0:
                        size_icon = "🔴"  # ใหญ่
                    elif file_size > 0.5:
                        size_icon = "🟡"  # กลาง  
                    else:
                        size_icon = "🟢"  # เล็ก
                    
                    st.write(f"  {size_icon} **{model['name']}** ({file_size:.1f} MB) - {modified_time.strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        
        # เพิ่มคำแนะนำการใช้งาน
        st.info("""
        💡 **คำแนะนำการเลือกโมเดล:**
        - 🚀 **Production Models**: โมเดลที่พร้อมใช้งานจริง (แนะนำ)
        - 🧪 **Training Output**: โมเดลจากการทดลอง/เทรน
        - 🔴 **ไฟล์ใหญ่** (>1MB): โมเดลซับซ้อน, ประสิทธิภาพสูง
        - 🟡 **ไฟล์กลาง** (0.5-1MB): โมเดลปานกลาง
        - 🟢 **ไฟล์เล็ก** (<0.5MB): โมเดลเบา, เร็ว
        """)
    
    # Load available models
    models_dict = load_model_list()
    if not models_dict:
        st.warning("⚠️ ไม่พบโมเดลที่เทรนแล้ว (.zip files)")
        st.info("""
        📝 **วิธีแก้ไข:**
        1. ไปที่หน้า "Train Agent" เพื่อเทรนโมเดลใหม่
        2. หรือตรวจสอบว่ามีไฟล์ .zip ในโฟลเดอร์ models/ หรือ notebooks/models/
        3. หากมีไฟล์ .pkl เท่านั้น แสดงว่าเป็นไฟล์ config ไม่ใช่โมเดล
        """)
        return
    
    # Model selection - แสดงโมเดลทั้งหมดจากหลายโฟลเดอร์
    model_display_names = list(models_dict.keys())
    selected_model_display = st.selectbox(
        "Select Model to Evaluate",
        model_display_names,
        help="Choose a trained model to evaluate its performance"
    )
    
    # ได้ข้อมูลโมเดลที่เลือก
    selected_model_info = models_dict[selected_model_display]
    
    # แสดงข้อมูลโมเดลที่เลือก
    with st.expander("📋 Model Information", expanded=False):
        st.write(f"**Model Name:** {selected_model_info['name']}")
        st.write(f"**File:** {selected_model_info['file']}")
        st.write(f"**Location:** {selected_model_info['path']}")
        st.write(f"**Full Path:** {selected_model_info['full_path']}")
        
        # เพิ่มปุ่มสำหรับจัดการโมเดล
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Copy to Root Models", key=f"copy_{selected_model_info['name']}"):
                try:
                    source_path = selected_model_info['full_path']
                    dest_path = os.path.join(MODEL_DIR, selected_model_info['file'])
                    
                    if not os.path.exists(dest_path):
                        shutil.copy2(source_path, dest_path)
                        st.success(f"คัดลอกโมเดลไปยัง {dest_path} สำเร็จ!")
                        try:
                            st.rerun()
                        except:
                            st.info("กรุณารีเฟรชหน้าเพื่อดูการเปลี่ยนแปลง")
                    else:
                        st.warning("โมเดลนี้มีอยู่ใน Root Models แล้ว")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการคัดลอก: {str(e)}")
        
        with col2:
            if st.button("🗂️ Organize All Models", key="organize_all"):
                organize_all_models()
    
    # ดึงรายการเวอร์ชั่นของโมเดล
    versions = get_model_versions(selected_model_info)
    if not versions:
        st.warning("⚠️ ไม่พบเวอร์ชั่นของโมเดลนี้")
        return
    
    # เลือกเวอร์ชั่นที่ต้องการเปรียบเทียบ
    st.subheader("📊 Version Comparison")
    selected_versions = st.multiselect(
        "เลือกเวอร์ชั่นที่ต้องการเปรียบเทียบ",
        options=[v['version'] for v in versions],
        default=[v['version'] for v in versions[-2:]] if len(versions) >= 2 else [v['version'] for v in versions],
        help="เลือกเวอร์ชั่นที่ต้องการเปรียบเทียบประสิทธิภาพ"
    )
    
    if not selected_versions:
        st.warning("⚠️ กรุณาเลือกเวอร์ชั่นที่ต้องการเปรียบเทียบ")
        return
    
    # Data file selection with browse capability
    st.subheader("📂 Data File Selection")
    
    # ค้นหาไฟล์ข้อมูลทั้งหมด
    all_data_files = find_all_data_files()
    
    if not all_data_files:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูล .csv ในโปรเจกต์ กรุณาโหลดข้อมูลก่อน")
        return
    
    # แสดงข้อมูลสรุปไฟล์
    with st.expander("📋 Available Data Files", expanded=False):
        st.write(f"**Total Files Found:** {len(all_data_files)}")
        for file_info in all_data_files:
            size_mb = file_info['size'] / (1024*1024)
            st.write(f"- **{file_info['name']}** ({size_mb:.2f} MB) - `{file_info['relative_path']}`")
    
    # สร้าง display options
    data_file_options = []
    for file_info in all_data_files:
        size_mb = file_info['size'] / (1024*1024)
        display_name = f"{file_info['name']} ({size_mb:.2f} MB) - {file_info['relative_path']}"
        data_file_options.append(display_name)
    
    selected_data_display = st.selectbox(
        "เลือกไฟล์ข้อมูลสำหรับการประเมิน",
        data_file_options,
        help="เลือกไฟล์ข้อมูลที่ต้องการใช้ในการประเมิน"
    )
    
    # หาไฟล์ที่เลือกจริง
    selected_data_index = data_file_options.index(selected_data_display)
    selected_data_info = all_data_files[selected_data_index]
    selected_data_file = selected_data_info['full_path']
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=30),
            help="Start date for evaluation period"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            help="End date for evaluation period"
        )
    
    # Trading parameters
    with st.expander("🔧 Trading Parameters"):
        initial_amount = st.number_input(
            "Initial Investment ($)",
            min_value=1000,
            value=INITIAL_AMOUNT,
            step=1000,
            help="Starting portfolio value"
        )
    
    if st.button("📊 Run Evaluation"):
        with st.spinner("Running evaluation..."):
            try:
                # โหลดข้อมูล
                if not os.path.exists(selected_data_file):
                    st.error(f"ไม่พบไฟล์ข้อมูลที่ {selected_data_file}")
                    return
                    
                st.info(f"🔄 กำลังโหลดข้อมูลจาก: `{selected_data_info['relative_path']}`")
                df = pd.read_csv(selected_data_file)
                
                # เตรียมข้อมูล
                processed_df = prepare_data_for_training(df)
                
                # กรองข้อมูลตามช่วงวันที่
                processed_df['date'] = pd.to_datetime(processed_df['date'])
                mask = (processed_df['date'].dt.date >= start_date) & (processed_df['date'].dt.date <= end_date)
                eval_df = processed_df[mask].copy()
                
                if len(eval_df) == 0:
                    st.error("ไม่พบข้อมูลในช่วงวันที่ที่เลือก")
                    return
                
                # แสดงข้อมูลสถิติ
                st.info(f"""
                📊 ข้อมูลสำหรับการประเมิน:
                - จำนวนข้อมูล: {len(eval_df):,} แถว
                - สกุลเงิน: {', '.join(eval_df['tic'].unique()) if 'tic' in eval_df.columns else 'ไม่ระบุ'}
                - ช่วงเวลา: {eval_df['date'].min().strftime('%Y-%m-%d')} ถึง {eval_df['date'].max().strftime('%Y-%m-%d')}
                - จำนวน indicators: {len([col for col in eval_df.columns if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']])}
                """)
                
                # ตรวจสอบข้อมูลขั้นต่ำ
                if len(eval_df) < 5:
                    st.error(f"ข้อมูลมีจำนวนน้อยเกินไป ({len(eval_df)} แถว) ไม่สามารถทำการประเมินได้")
                    st.info("กรุณาเลือกช่วงวันที่ที่กว้างขึ้น หรือใช้ข้อมูลที่มีจำนวนมากกว่า")
                    return
                
                # สร้าง environment
                env = create_evaluation_env(eval_df, initial_amount)
                if env is None:
                    return
                
                # เก็บผลลัพธ์ของแต่ละเวอร์ชั่น
                version_results = {}
                
                # ทดสอบแต่ละเวอร์ชั่น
                for version in selected_versions:
                    version_info = next(v for v in versions if v['version'] == version)
                    
                    # โหลดโมเดล - แปลง version_info เป็นรูปแบบที่ load_model เข้าใจ
                    model_info_for_loading = {
                        'name': version_info['file'].replace('.zip', ''),
                        'file': version_info['file'],
                        'path': version_info['path'],
                        'full_path': version_info['full_path']
                    }
                    model = load_model(model_info_for_loading)
                    if model is None:
                        continue
                    
                    # ทดสอบโมเดล
                    try:
                        st.info(f"🔄 กำลังประเมิน Version {version}...")
                        
                        # สร้าง environment สำหรับ version นี้
                        version_env = create_evaluation_env(eval_df, initial_amount)
                        if version_env is None:
                            st.error(f"ไม่สามารถสร้าง environment สำหรับ Version {version}")
                            continue
                        
                        df_account_value, df_actions = DRLAgent.DRL_prediction(
                            model=model,
                            environment=version_env
                        )
                        
                        # ตรวจสอบผลลัพธ์
                        if df_account_value is None or len(df_account_value) == 0:
                            st.error(f"ไม่ได้รับผลลัพธ์จากการประเมิน Version {version}")
                            continue
                        
                        # คำนวณ benchmark (Buy & Hold)
                        benchmark_values = []
                        unique_dates = sorted(eval_df['date'].unique())
                        initial_price = eval_df['close'].iloc[0]
                        
                        for date in unique_dates:
                            date_data = eval_df[eval_df['date'] == date]
                            if len(date_data) > 0:
                                current_price = date_data['close'].iloc[0]
                                benchmark_value = initial_amount * (current_price / initial_price)
                                benchmark_values.append(benchmark_value)
                            else:
                                benchmark_values.append(initial_amount)
                        
                        # เก็บผลลัพธ์
                        version_results[version] = {
                            'dates': unique_dates[:len(df_account_value)],
                            'account_values': df_account_value['account_value'].values,
                            'benchmark_values': benchmark_values[:len(df_account_value)],
                            'metrics': calculate_metrics(
                                df_account_value['account_value'].values,
                                benchmark_values[:len(df_account_value)],
                                initial_amount
                            )
                        }
                        
                        st.success(f"✅ ประเมิน Version {version} สำเร็จ!")
                        
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการประเมินเวอร์ชั่น {version}: {str(e)}")
                        import traceback
                        st.text(f"รายละเอียดข้อผิดพลาด: {traceback.format_exc()}")
                        continue
                
                if not version_results:
                    st.error("ไม่สามารถประเมินผลได้สำหรับเวอร์ชั่นที่เลือก")
                    return
                
                # แสดงกราฟเปรียบเทียบ
                st.plotly_chart(plot_version_comparison(version_results))
                
                # แสดงตารางเปรียบเทียบ metrics
                st.subheader("📊 Performance Comparison")
                metrics_df = pd.DataFrame({
                    'Version': [],
                    'Total Return (%)': [],
                    'vs. Buy & Hold': [],
                    'Sharpe Ratio': [],
                    'Max Drawdown (%)': [],
                    'Final Portfolio Value': []
                })
                
                for version, result in version_results.items():
                    metrics = result['metrics']
                    metrics_df = pd.concat([metrics_df, pd.DataFrame({
                        'Version': [f"v{version}"],
                        'Total Return (%)': [metrics['Total Return (%)']],
                        'vs. Buy & Hold': [metrics['vs. Buy & Hold']],
                        'Sharpe Ratio': [metrics['Sharpe Ratio']],
                        'Max Drawdown (%)': [metrics['Max Drawdown (%)']],
                        'Final Portfolio Value': [metrics['Final Portfolio Value']]
                    })])
                
                st.dataframe(metrics_df, hide_index=True)
                
                # Download results button
                results_df = pd.DataFrame()
                for version, result in version_results.items():
                    version_df = pd.DataFrame({
                        'Date': result['dates'],
                        f'Portfolio_Value_v{version}': result['account_values'],
                        'Benchmark_Value': result['benchmark_values']
                    })
                    if results_df.empty:
                        results_df = version_df
                    else:
                        results_df = results_df.merge(version_df, on='Date', how='outer')
                
                st.download_button(
                    "📥 Download Comparison Report",
                    results_df.to_csv(index=False),
                    "version_comparison_report.csv",
                    help="Download detailed comparison results as CSV"
                )
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประเมิน: {str(e)}")
                st.error("กรุณาตรวจสอบข้อมูลและลองใหม่อีกครั้ง") 