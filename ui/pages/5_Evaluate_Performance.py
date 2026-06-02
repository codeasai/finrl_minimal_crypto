import streamlit as st
import sys
import os
import json
from pathlib import Path

st.set_page_config(
    page_title="Evaluate Performance",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import INITIAL_AMOUNT

MODEL_DIR = os.path.join(root_path, "models")


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(account_values: np.ndarray, initial_amount: float) -> dict:
    """คำนวณ performance metrics จาก array ของ portfolio value"""
    av = np.array(account_values, dtype=float)
    returns = np.diff(av) / av[:-1]  # daily returns

    total_return  = (av[-1] - initial_amount) / initial_amount * 100
    final_value   = av[-1]

    # Sharpe Ratio (annualised, assuming daily, 365 trading days crypto)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    # Sortino Ratio (downside deviation only)
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(365)
    else:
        sortino = 0.0

    # Max Drawdown
    cummax = np.maximum.accumulate(av)
    drawdowns = (av - cummax) / cummax
    max_drawdown = drawdowns.min() * 100  # negative %

    # Calmar Ratio
    calmar = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Win Rate (% of days with positive return)
    win_rate = (returns > 0).mean() * 100 if len(returns) > 0 else 0.0

    # Volatility (annualised)
    volatility = returns.std() * np.sqrt(365) * 100

    return {
        "total_return":  total_return,
        "final_value":   final_value,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "max_drawdown":  max_drawdown,
        "win_rate":      win_rate,
        "volatility":    volatility,
    }


def compute_bnh_returns(test_df: pd.DataFrame, symbols: list,
                        initial_amount: float) -> dict:
    """
    คำนวณ Buy & Hold return สำหรับแต่ละ symbol
    (เทียบ close ราคาแรกและราคาสุดท้ายของ test set)
    """
    results = {}
    for sym in symbols:
        sdf = test_df[test_df['tic'] == sym].sort_values('date')
        if len(sdf) < 2:
            continue
        p0 = sdf['close'].iloc[0]
        p1 = sdf['close'].iloc[-1]
        if p0 != 0:
            ret = (p1 - p0) / p0 * 100
        else:
            ret = 0.0
        results[sym] = ret
    return results


def build_bnh_curve(test_df: pd.DataFrame, symbols: list,
                    initial_amount: float) -> np.ndarray:
    """สร้าง equal-weight B&H portfolio curve เพื่อเปรียบเทียบกับ agent"""
    dates = sorted(test_df['date'].unique())
    n = len(symbols)
    if n == 0:
        return np.full(len(dates), initial_amount)

    alloc = initial_amount / n  # equal weight
    curves = []
    for sym in symbols:
        sdf = test_df[test_df['tic'] == sym].sort_values('date').set_index('date')
        prices = sdf['close'].reindex(dates).ffill().bfill().values
        if prices[0] != 0:
            curves.append(alloc * prices / prices[0])
        else:
            curves.append(np.full(len(dates), alloc))
    return np.sum(curves, axis=0)


# ─── UI ──────────────────────────────────────────────────────────────────────

def evaluate_agent_ui():
    st.header("📈 Evaluate Performance")

    with st.expander("💡 วิธีใช้งาน", expanded=False):
        st.markdown("""
**ขั้นตอนที่ 5 จาก 6** — วิเคราะห์ผลการเทรดและเปรียบเทียบกับ Buy & Hold

- ถ้าเพิ่งรัน backtest จากขั้นที่ 4 → ผลโหลดอัตโนมัติ
- ถ้า reload หน้า → เลือกไฟล์จาก dropdown ด้านล่าง

**Metrics สำคัญ:**
- **Sharpe > 1** = ผลดี, **> 2** = ผลดีมาก
- **Max Drawdown** ยิ่งน้อยยิ่งดี (ใกล้ 0%)
- ถ้า agent แพ้ Buy & Hold → ลองเพิ่ม Grade หรือข้อมูลให้มากขึ้น

> ดูคำอธิบาย metrics ทั้งหมดที่หน้า **📖 Help**
""")

    # ── แหล่งข้อมูล: session_state หรือโหลดจากไฟล์ CSV ─────────────────────
    source = "session"
    df_account_value = None
    df_actions       = None
    test_df          = None
    model_name       = None
    symbols          = None

    if "backtest_account_value" in st.session_state:
        df_account_value = st.session_state["backtest_account_value"]
        df_actions       = st.session_state["backtest_actions"]
        test_df          = st.session_state["backtest_test_df"]
        model_name       = st.session_state["backtest_model_name"]
        symbols          = st.session_state["backtest_symbols"]
        st.success(f"✅ ใช้ผล Backtest จาก session: **{model_name}**")
    else:
        st.info("ยังไม่มีผล Backtest ใน session — โหลดจากไฟล์ที่บันทึกไว้แทน")
        source = "file"

    # ── โหลดจากไฟล์ถ้าไม่มีใน session ──────────────────────────────────────
    results_dir = os.path.join(MODEL_DIR, "backtest_results")
    av_files = []
    if os.path.exists(results_dir):
        av_files = [f for f in os.listdir(results_dir) if f.endswith("_account_value.csv")]
        av_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)

    if source == "file":
        if not av_files:
            st.warning("ยังไม่มีผล Backtest — ไปหน้า **Test Agent** แล้วกด Run Backtest ก่อน")
            return
        selected_av = st.selectbox("เลือกไฟล์ผล Backtest", av_files)
        av_path = os.path.join(results_dir, selected_av)
        ac_path = av_path.replace("_account_value.csv", "_actions.csv")
        try:
            df_account_value = pd.read_csv(av_path)
            if os.path.exists(ac_path):
                df_actions = pd.read_csv(ac_path)
            model_name = selected_av.replace("_account_value.csv", "")
        except Exception as e:
            st.error(f"❌ โหลดไฟล์ไม่ได้: {e}")
            return

        # โหลด sidecar เพื่อรู้ symbols
        zip_path = os.path.join(MODEL_DIR, f"minimal_crypto_{model_name}.zip")
        meta = {}
        sidecar_path = zip_path.replace('.zip', '.json')
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                pass
        symbols = meta.get("symbols", [])

    if df_account_value is None:
        return

    account_values = df_account_value['account_value'].values
    metrics = compute_metrics(account_values, INITIAL_AMOUNT)

    st.markdown("---")

    # ── Metrics row ──────────────────────────────────────────────────────────
    st.subheader("📊 Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return",
              f"{metrics['total_return']:.2f}%",
              delta=f"{metrics['total_return']:.2f}%")
    c2.metric("Final Value",   f"${metrics['final_value']:,.0f}")
    c3.metric("Sharpe Ratio",  f"{metrics['sharpe']:.3f}")
    c4.metric("Max Drawdown",  f"{metrics['max_drawdown']:.2f}%",
              delta=f"{metrics['max_drawdown']:.2f}%", delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sortino Ratio",  f"{metrics['sortino']:.3f}")
    c6.metric("Calmar Ratio",   f"{metrics['calmar']:.3f}")
    c7.metric("Win Rate",       f"{metrics['win_rate']:.1f}%")
    c8.metric("Volatility",     f"{metrics['volatility']:.1f}%")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────────
    st.subheader("📈 Portfolio vs Buy & Hold")

    # สร้าง Buy & Hold curve ถ้ามี test_df
    if test_df is not None and symbols:
        bnh_curve = build_bnh_curve(test_df, symbols, INITIAL_AMOUNT)
        n = min(len(account_values), len(bnh_curve))

        chart_data = pd.DataFrame({
            "Agent Portfolio": account_values[:n],
            "Buy & Hold (Equal Weight)": bnh_curve[:n],
        })
        st.line_chart(chart_data)

        # B&H per-symbol returns
        bnh_returns = compute_bnh_returns(test_df, symbols, INITIAL_AMOUNT)
        if bnh_returns:
            st.subheader("📊 Buy & Hold Return per Symbol")
            bnh_df = pd.DataFrame(
                [(sym, f"{ret:.2f}%") for sym, ret in bnh_returns.items()],
                columns=["Symbol", "B&H Return"]
            )
            # เพิ่ม Alpha
            bnh_df["Alpha vs Agent"] = bnh_df["B&H Return"].apply(
                lambda x: f"{metrics['total_return'] - float(x.replace('%','')):.2f}%"
            )
            st.dataframe(bnh_df, use_container_width=True)
    else:
        # ไม่มี test_df — แสดงแค่ portfolio
        chart_data = pd.DataFrame({"Agent Portfolio": account_values})
        st.line_chart(chart_data)
        st.caption("โหลดผล Backtest จาก session เพื่อดูกราฟเปรียบเทียบ Buy & Hold")

    # ── Drawdown chart ───────────────────────────────────────────────────────
    st.subheader("📉 Drawdown")
    av = np.array(account_values, dtype=float)
    cummax = np.maximum.accumulate(av)
    drawdown_series = (av - cummax) / cummax * 100
    dd_df = pd.DataFrame({"Drawdown (%)": drawdown_series})
    st.area_chart(dd_df)

    # ── Actions ──────────────────────────────────────────────────────────────
    if df_actions is not None:
        st.markdown("---")
        with st.expander("📋 Trading Actions"):
            st.dataframe(df_actions.head(50), use_container_width=True)
            st.caption(f"แสดง 50 แถวแรก จาก {len(df_actions):,} แถวทั้งหมด")

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Export ผลลัพธ์")
    col1, col2 = st.columns(2)
    with col1:
        csv_av = df_account_value.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ ดาวน์โหลด Account Value CSV",
            data=csv_av,
            file_name=f"{model_name}_account_value.csv",
            mime="text/csv"
        )
    with col2:
        if df_actions is not None:
            csv_ac = df_actions.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ ดาวน์โหลด Actions CSV",
                data=csv_ac,
                file_name=f"{model_name}_actions.csv",
                mime="text/csv"
            )

    # ── Summary card ─────────────────────────────────────────────────────────
    st.markdown("---")
    verdict = ("Agent ทำผลได้ **ดีกว่า** Buy & Hold"
               if metrics['total_return'] > 0 else
               "Agent ทำผลได้ **ต่ำกว่า** Buy & Hold — ลองเพิ่ม steps หรือปรับ Grade")
    if metrics['sharpe'] > 1:
        verdict += " | Sharpe > 1 ✅ (ดี)"
    elif metrics['sharpe'] > 0:
        verdict += " | Sharpe > 0 ⚠️ (พอใช้)"
    else:
        verdict += " | Sharpe < 0 ❌ (ควรปรับปรุง)"

    st.info(f"**สรุป:** {verdict}")


evaluate_agent_ui()
