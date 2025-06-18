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

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR, INITIAL_AMOUNT, CRYPTO_SYMBOLS, DATA_DIR, TRANSACTION_COST_PCT, HMAX
from ui.pipeline.train import add_technical_indicators, prepare_data_for_training

def load_model_list():
    """Load list of available models"""
    if not os.path.exists(MODEL_DIR):
        return []
    return [f.replace("minimal_crypto_", "") for f in os.listdir(MODEL_DIR) 
            if f.startswith("minimal_crypto_")]

def load_model(model_name):
    """Load a trained model"""
    try:
        model_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_name}")
        if not os.path.exists(model_path):
            st.error(f"ไม่พบไฟล์โมเดลที่ {model_path}")
            return None
            
        # ตรวจสอบว่าเป็นไฟล์โมเดลหรือไม่
        if not model_path.endswith('.zip'):
            st.error(f"ไฟล์ {model_path} ไม่ใช่ไฟล์โมเดลที่ถูกต้อง")
            return None
            
        # โหลดโมเดลด้วย device ที่เหมาะสม
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PPO.load(model_path, device=device)
        
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
        
        # ตรวจสอบข้อมูลก่อนสร้าง environment
        st.info(f"""
        📊 ข้อมูลสำหรับการประเมิน:
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

def get_model_versions(model_name):
    """ดึงรายการเวอร์ชั่นทั้งหมดของโมเดล"""
    try:
        # ค้นหาไฟล์ที่มีชื่อขึ้นต้นเหมือนกัน
        existing_files = [f for f in os.listdir(MODEL_DIR) 
                         if f.startswith(f"{model_name}_v") and f.endswith('.zip')]
        
        if not existing_files:
            return []
        
        # ดึงเลขเวอร์ชั่นและเรียงลำดับ
        versions = []
        for file in existing_files:
            try:
                version = int(file.split('_v')[1].split('.')[0])
                versions.append({
                    'version': version,
                    'file': file,
                    'path': os.path.join(MODEL_DIR, file),
                    'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(MODEL_DIR, file)))
                })
            except:
                continue
        
        return sorted(versions, key=lambda x: x['version'])
    except:
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
    
    # Load available models
    models = load_model_list()
    if not models:
        st.info("No trained models found. Please train a model first!")
        return
    
    # Model selection
    model_name = st.selectbox(
        "Select Model to Evaluate",
        models,
        help="Choose a trained model to evaluate its performance"
    )
    
    # ดึงรายการเวอร์ชั่นของโมเดล
    versions = get_model_versions(f"minimal_crypto_{model_name}")
    if not versions:
        st.warning("⚠️ ไม่พบเวอร์ชั่นของโมเดลนี้")
        return
    
    # เลือกเวอร์ชั่นที่ต้องการเปรียบเทียบ
    st.subheader("📊 Version Comparison")
    selected_versions = st.multiselect(
        "เลือกเวอร์ชั่นที่ต้องการเปรียบเทียบ",
        options=[v['version'] for v in versions],
        default=[v['version'] for v in versions[-2:]],  # เลือก 2 เวอร์ชั่นล่าสุดเป็นค่าเริ่มต้น
        help="เลือกเวอร์ชั่นที่ต้องการเปรียบเทียบประสิทธิภาพ"
    )
    
    if not selected_versions:
        st.warning("⚠️ กรุณาเลือกเวอร์ชั่นที่ต้องการเปรียบเทียบ")
        return
    
    # Data file selection
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    if not data_files:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูลในโฟลเดอร์ data กรุณาโหลดข้อมูลก่อน")
        return

    selected_data_file = st.selectbox(
        "เลือกไฟล์ข้อมูลสำหรับการประเมิน",
        data_files,
        help="เลือกไฟล์ข้อมูลที่ต้องการใช้ในการประเมิน"
    )
    
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
                file_path = os.path.join(DATA_DIR, selected_data_file)
                if not os.path.exists(file_path):
                    st.error(f"ไม่พบไฟล์ข้อมูลที่ {file_path}")
                    return
                    
                df = pd.read_csv(file_path)
                
                # เตรียมข้อมูล
                processed_df = prepare_data_for_training(df)
                
                # กรองข้อมูลตามช่วงวันที่
                processed_df['date'] = pd.to_datetime(processed_df['date'])
                mask = (processed_df['date'].dt.date >= start_date) & (processed_df['date'].dt.date <= end_date)
                eval_df = processed_df[mask].copy()
                
                if len(eval_df) == 0:
                    st.error("ไม่พบข้อมูลในช่วงวันที่ที่เลือก")
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
                    
                    # โหลดโมเดล
                    model = load_model(version_info['file'].replace("minimal_crypto_", "").replace(".zip", ""))
                    if model is None:
                        continue
                    
                    # ทดสอบโมเดล
                    try:
                        df_account_value, df_actions = DRLAgent.DRL_prediction(
                            model=model,
                            environment=env
                        )
                        
                        # คำนวณ benchmark (Buy & Hold)
                        benchmark_values = []
                        for date in eval_df['date'].unique():
                            date_data = eval_df[eval_df['date'] == date]
                            benchmark_value = initial_amount * (date_data['close'].iloc[0] / eval_df['close'].iloc[0])
                            benchmark_values.append(benchmark_value)
                        
                        # เก็บผลลัพธ์
                        version_results[version] = {
                            'dates': eval_df['date'].unique(),
                            'account_values': df_account_value['account_value'].values,
                            'benchmark_values': benchmark_values,
                            'metrics': calculate_metrics(
                                df_account_value['account_value'].values,
                                benchmark_values,
                                initial_amount
                            )
                        }
                        
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการประเมินเวอร์ชั่น {version}: {str(e)}")
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