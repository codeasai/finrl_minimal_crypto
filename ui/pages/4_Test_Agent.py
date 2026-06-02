import streamlit as st
import sys
import os
import json
import time
import traceback
from pathlib import Path

st.set_page_config(
    page_title="Test Agent",
    page_icon="🧪",
    layout="wide"
)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import INITIAL_AMOUNT, HMAX, TRANSACTION_COST_PCT

MODEL_DIR = os.path.join(root_path, "models")
DATA_DIR  = os.path.join(root_path, "data")

REQUIRED_INDICATORS = [
    'sma_20', 'ema_20', 'rsi_14',
    'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
    'volume_sma_20', 'volume_ratio'
]


def load_sidecar(model_zip_path):
    sidecar = model_zip_path.replace('.zip', '.json')
    if os.path.exists(sidecar):
        try:
            with open(sidecar, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def prepare_test_df(df):
    """เตรียม DataFrame สำหรับ test environment (เหมือนกับ Train Agent)"""
    df = df.copy()
    if 'symbol' in df.columns and 'tic' not in df.columns:
        df = df.rename(columns={'symbol': 'tic'})
    if 'tic' not in df.columns:
        return None, [], "ไม่พบคอลัมน์ 'tic' หรือ 'symbol'"
    if 'bb_std' not in df.columns and 'bb_upper' in df.columns and 'bb_middle' in df.columns:
        df['bb_std'] = (df['bb_upper'] - df['bb_middle']) / 2
    missing = [c for c in REQUIRED_INDICATORS if c not in df.columns]
    if missing:
        return None, [], f"ขาด indicators: {', '.join(missing)} — กรุณา Re-run Data Prepare"
    keep = ['timestamp', 'tic', 'open', 'high', 'low', 'close', 'volume'] + REQUIRED_INDICATORS
    df = df[[c for c in keep if c in df.columns]]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    symbols = sorted(df['tic'].unique().tolist())
    return df, symbols, None


def resample_to_daily(df, indicators):
    """
    Resample intraday OHLCV+indicators เป็น daily (1 row ต่อ symbol ต่อวัน)
    แล้วตั้ง day-based index ที่ FinRL StockTradingEnv ต้องการ
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')

    # ถ้าข้อมูลเป็น daily อยู่แล้ว (1 row/symbol/day) ไม่ต้อง resample
    rows_per_day = df.groupby(['date', 'tic']).size().max()
    if rows_per_day == 1:
        daily_df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    else:
        # resample intraday -> daily
        agg = {'open': 'first', 'high': 'max', 'low': 'min',
               'close': 'last', 'volume': 'sum'}
        for ind in indicators:
            if ind in df.columns:
                agg[ind] = 'last'

        parts = []
        for tic in df['tic'].unique():
            sub = df[df['tic'] == tic].set_index('timestamp').sort_index()
            d = sub.resample('1D').agg(agg).dropna(subset=['close'])
            d['tic'] = tic
            d['date'] = d.index.strftime('%Y-%m-%d')
            parts.append(d.reset_index(drop=True))

        daily_df = pd.concat(parts).sort_values(['date', 'tic']).reset_index(drop=True)

    # แปลงตัวเลขให้ครบ
    for col in ['open', 'high', 'low', 'close', 'volume'] + indicators:
        if col in daily_df.columns:
            daily_df[col] = pd.Series(daily_df[col].values).astype('float64').fillna(0.0)
    daily_df = daily_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ตั้ง day-based index: แต่ละ day number ครอบคลุม n_symbols rows
    # FinRL ใช้ df.loc[day] ซึ่งต้องได้ DataFrame ไม่ใช่ Series
    dates = sorted(daily_df['date'].unique())
    day_index = []
    for i, date in enumerate(dates):
        cnt = (daily_df['date'] == date).sum()
        day_index.extend([i] * cnt)
    daily_df.index = day_index

    return daily_df


def detect_model_spec(model_path):
    """
    อ่าน n_symbols และ obs_dim จาก model .zip โดยตรง
    Returns (n_symbols, obs_dim)
    """
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    obs_dim = model.observation_space.shape[0]
    n_symbols = model.action_space.shape[0]
    return n_symbols, obs_dim, model


def build_test_env(df, model_path):
    """
    สร้าง test environment ให้ตรง spec ของ model
    - detect n_symbols จาก model ก่อน
    - กรอง symbols ในข้อมูลให้เหลือ n_symbols ตัวแรก (sorted)
    - resample เป็น daily แล้วตั้ง day-based index
    """
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    indicators = REQUIRED_INDICATORS
    n_symbols, obs_dim, model = detect_model_spec(model_path)

    # กรอง symbols ให้ตรงกับจำนวนที่ model เทรนมา
    all_symbols = sorted(df['tic'].unique().tolist())
    if len(all_symbols) > n_symbols:
        symbols = all_symbols[:n_symbols]
        df = df[df['tic'].isin(symbols)].copy()
    else:
        symbols = all_symbols
    n = len(symbols)

    # ตรวจว่า obs_dim ตรงกัน
    expected_obs = 1 + 2 * n + n * len(indicators)
    if expected_obs != obs_dim:
        raise ValueError(
            f"Model ต้องการ obs_dim={obs_dim} แต่ data+indicators ให้ {expected_obs} "
            f"(n_symbols={n}, n_indicators={len(indicators)})\n"
            f"Symbols ในข้อมูล: {all_symbols}\n"
            f"กรุณาใช้ไฟล์ข้อมูลที่เทรนกับ model นี้"
        )

    # resample เป็น daily
    daily_df = resample_to_daily(df, indicators)
    dates = sorted(daily_df['date'].unique())
    split_date = dates[int(len(dates) * 0.8)]
    test_df = daily_df[daily_df['date'] >= split_date].copy()

    # day-based index สำหรับ test set
    test_dates = sorted(test_df['date'].unique())
    day_index = []
    for i, date in enumerate(test_dates):
        cnt = (test_df['date'] == date).sum()
        day_index.extend([i] * cnt)
    test_df.index = day_index

    env_kwargs = {
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * n,
        "buy_cost_pct":  [TRANSACTION_COST_PCT] * n,
        "sell_cost_pct": [TRANSACTION_COST_PCT] * n,
        "state_space": obs_dim,
        "stock_dim": n,
        "tech_indicator_list": indicators,
        "action_space": n,
        "reward_scaling": 1e-3,
        "print_verbosity": 1,
    }
    env = StockTradingEnv(df=test_df, **env_kwargs)
    return env, test_df, model, symbols


def run_backtest(model, test_env):
    """รัน DRL_prediction ด้วย model object ที่โหลดแล้ว"""
    from finrl.agents.stablebaselines3.models import DRLAgent
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model,
        environment=test_env,
    )
    return df_account_value, df_actions


def save_backtest_results(model_name, df_account_value, df_actions, test_df):
    """บันทึกผล backtest เป็น CSV ใน models/"""
    results_dir = os.path.join(MODEL_DIR, "backtest_results")
    os.makedirs(results_dir, exist_ok=True)
    av_path = os.path.join(results_dir, f"{model_name}_account_value.csv")
    ac_path = os.path.join(results_dir, f"{model_name}_actions.csv")
    df_account_value.to_csv(av_path, index=False)
    df_actions.to_csv(ac_path, index=False)
    return av_path, ac_path


# ─── UI ──────────────────────────────────────────────────────────────────────

def test_agent_ui():
    st.header("🧪 Test Agent — Backtest")

    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown("""
**ขั้นตอนที่ 4 จาก 6** — ทดสอบ agent บน test set ที่ไม่เคยเห็นระหว่างเทรน

1. เลือก **Model** ที่ต้องการทดสอบ
2. เลือก **ไฟล์ข้อมูล** — ควรเป็นไฟล์เดียวกับที่ใช้เทรน (ระบบแนะนำอัตโนมัติ)
3. กด **Run Backtest** — รอจน status เปลี่ยนเป็น ✅
4. ดูผลสรุปแล้วไปต่อที่หน้า **📈 Evaluate Performance**

> Test set = 20% หลังของข้อมูล (agent ไม่เคยเห็นช่วงนี้ระหว่างเทรน)
""")

    # ── เลือก model ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_DIR):
        st.warning("⚠️ ไม่พบโฟลเดอร์ models — กรุณาเทรน agent ก่อน")
        return

    model_files = [f for f in os.listdir(MODEL_DIR)
                   if f.startswith("minimal_crypto_") and f.endswith(".zip")]
    if not model_files:
        st.warning("⚠️ ไม่พบ model — ไปหน้า Train Agent เพื่อสร้าง agent ก่อน")
        return

    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
    selected_model_file = st.selectbox("เลือก Model", model_files,
                                       help="เรียงตามวันที่แก้ไขล่าสุด")
    model_path = os.path.join(MODEL_DIR, selected_model_file)
    model_name = selected_model_file.replace("minimal_crypto_", "").replace(".zip", "")

    # แสดง sidecar metadata
    meta = load_sidecar(model_path)
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exchange", meta.get("exchange", "N/A").upper())
        c2.metric("Grade", meta.get("grade", "N/A"))
        c3.metric("Steps trained", f"{meta.get('total_steps', 0):,}")
        c4.metric("Symbols", ", ".join(meta.get("symbols", [])))

    st.markdown("---")

    # ── เลือกไฟล์ข้อมูล ──────────────────────────────────────────────────────
    # แยก prepared files (data_prepare/) ออกจาก raw files
    prepared_files = []
    raw_files = []
    for dirpath, _, files in os.walk(DATA_DIR):
        for f in files:
            if not f.endswith(".csv"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), start=str(root_path))
            if "data_prepare" in dirpath:
                prepared_files.append(rel)
            else:
                raw_files.append(rel)
    prepared_files.sort()
    raw_files.sort()
    # ให้ prepared files ขึ้นก่อนเสมอ
    csv_candidates = prepared_files + raw_files

    if not csv_candidates:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูล — ไปหน้า Data Loader ก่อน")
        return

    # default: ไฟล์ที่ตรงกับ sidecar ก่อน → ถ้าไม่มีให้เลือก prepared file แรก
    default_idx = 0
    if meta.get("data_file"):
        for i, p in enumerate(csv_candidates):
            if meta["data_file"] in p:
                default_idx = i
                break

    selected_data = st.selectbox("เลือกไฟล์ข้อมูล (ควรเป็นไฟล์เดียวกับที่ใช้เทรน)",
                                 csv_candidates, index=default_idx)

    if "data_prepare" not in selected_data:
        st.warning("⚠️ ไฟล์ที่เลือกเป็น raw data — ควรใช้ไฟล์จาก `data/data_prepare/` "
                   "ที่มี technical indicators ครบ (ผ่านขั้นตอน Data Prepare แล้ว)")
    else:
        st.caption("✅ ไฟล์จาก data/data_prepare/ — มี indicators ครบพร้อม backtest")

    st.markdown("---")

    # ── ปุ่ม Run Backtest ────────────────────────────────────────────────────
    if st.button("▶️ Run Backtest", type="primary"):
        data_path = os.path.join(root_path, selected_data)
        try:
            df_raw = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"❌ อ่านไฟล์ข้อมูลไม่ได้: {e}")
            return

        df_ready, symbols, err = prepare_test_df(df_raw)
        if err:
            st.error(f"❌ {err}")
            return

        t0 = time.time()
        try:
            with st.status("กำลัง Backtest...", expanded=True) as status:
                st.write(f"1/3 โหลด model และตรวจสอบ spec...")
                try:
                    test_env, test_df, loaded_model, used_symbols = build_test_env(
                        df_ready, model_path)
                    n_steps = len(test_df['date'].unique())
                    st.write(f"   Model spec: {len(used_symbols)} symbols {used_symbols}")
                    st.write(f"   Test set: {n_steps} วัน "
                             f"({test_df['date'].min()} → {test_df['date'].max()})")
                except Exception as e:
                    status.update(label="สร้าง environment ล้มเหลว", state="error")
                    raise

                st.write(f"2/3 รัน DRL_prediction...")
                try:
                    df_account_value, df_actions = run_backtest(loaded_model, test_env)
                except Exception as e:
                    status.update(label="Backtest ล้มเหลว", state="error")
                    raise

                st.write("3/3 บันทึกผลลัพธ์...")
                av_path, ac_path = save_backtest_results(
                    model_name, df_account_value, df_actions, test_df)

                elapsed = time.time() - t0
                status.update(label=f"Backtest เสร็จสิ้น! ({elapsed:.1f}s)", state="complete")

            # เก็บผลไว้ใน session_state สำหรับหน้า Evaluate
            st.session_state["backtest_account_value"] = df_account_value
            st.session_state["backtest_actions"]       = df_actions
            st.session_state["backtest_test_df"]       = test_df
            st.session_state["backtest_model_name"]    = model_name
            st.session_state["backtest_symbols"]       = used_symbols

            # ── quick summary ────────────────────────────────────────────────
            final_val   = df_account_value['account_value'].iloc[-1]
            total_ret   = (final_val - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100

            st.success(f"Backtest สำเร็จ! ผลลัพธ์ถูกบันทึกไว้ใน session สำหรับหน้า Evaluate")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("เงินทุนเริ่มต้น",  f"${INITIAL_AMOUNT:,.0f}")
            m2.metric("มูลค่าสุดท้าย",    f"${final_val:,.0f}")
            color = "normal" if total_ret >= 0 else "inverse"
            m3.metric("Total Return",     f"{total_ret:.2f}%",
                      delta=f"{total_ret:.2f}%")
            m4.metric("Test Period",      f"{n_steps} วัน")

            # portfolio curve
            st.subheader("📈 Portfolio Value")
            chart_df = pd.DataFrame({
                "account_value": df_account_value["account_value"].values
            })
            st.line_chart(chart_df)

            st.info("ไปหน้า **Evaluate Performance** เพื่อดู metrics และกราฟเปรียบเทียบ Buy & Hold")

            with st.expander("📁 ไฟล์ที่บันทึก"):
                st.write(f"- {av_path}")
                st.write(f"- {ac_path}")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
            with st.expander("รายละเอียด error"):
                st.code(traceback.format_exc())

    # ── แสดงผลจาก session ถ้ามี (reload-safe) ───────────────────────────────
    elif "backtest_account_value" in st.session_state:
        st.info(f"📋 ผล Backtest ล่าสุด: **{st.session_state['backtest_model_name']}** "
                f"— ไปหน้า Evaluate Performance เพื่อดูรายละเอียด")


test_agent_ui()
