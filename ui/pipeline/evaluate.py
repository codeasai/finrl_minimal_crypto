import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR, INITIAL_AMOUNT, CRYPTO_SYMBOLS

def load_model_list():
    """Load list of available models"""
    if not os.path.exists(MODEL_DIR):
        return []
    return [f.replace("minimal_crypto_", "") for f in os.listdir(MODEL_DIR) 
            if f.startswith("minimal_crypto_")]

def plot_performance(account_values, benchmark_values, dates):
    """Plot performance comparison using plotly"""
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=dates,
        y=account_values,
        name="Agent Portfolio",
        line=dict(color="#2E8B57", width=2)
    ))
    
    # Add benchmark line
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_values,
        name="Buy & Hold",
        line=dict(color="#4169E1", width=2, dash="dash")
    ))
    
    # Add initial investment line
    fig.add_trace(go.Scatter(
        x=dates,
        y=[INITIAL_AMOUNT] * len(dates),
        name="Initial Investment",
        line=dict(color="#DC143C", width=1, dash="dot")
    ))
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance Comparison",
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
        
        trade_interval = st.selectbox(
            "Trading Interval",
            ["Daily", "Weekly", "Monthly"],
            help="How often the agent makes trading decisions"
        )
    
    if st.button("📊 Run Evaluation"):
        with st.spinner("Running evaluation..."):
            # TODO: Implement actual evaluation
            # For now, we'll use dummy data
            dates = pd.date_range(start_date, end_date)
            account_values = [initial_amount * (1 + i * 0.001) for i in range(len(dates))]
            benchmark_values = [initial_amount * (1 + i * 0.0008) for i in range(len(dates))]
            
            # Display performance plot
            st.plotly_chart(plot_performance(account_values, benchmark_values, dates))
            
            # Display metrics
            metrics = calculate_metrics(account_values, benchmark_values, initial_amount)
            
            # Create three columns for metrics
            cols = st.columns(3)
            for i, (metric, value) in enumerate(metrics.items()):
                with cols[i % 3]:
                    st.metric(metric, value)
            
            # Trading activity summary
            st.subheader("📊 Trading Activity")
            st.write("Top 5 Trades:")
            
            # Dummy trade data
            trade_data = {
                "Date": ["2024-03-01", "2024-03-05", "2024-03-10"],
                "Action": ["Buy", "Sell", "Buy"],
                "Symbol": ["BTC-USD", "BTC-USD", "BTC-USD"],
                "Amount": ["$10,000", "$15,000", "$12,000"],
                "Price": ["$65,000", "$68,000", "$64,000"],
                "Return": ["+5.2%", "+4.1%", "-1.2%"]
            }
            st.dataframe(trade_data, hide_index=True)
            
            # Download results button
            st.download_button(
                "📥 Download Evaluation Report",
                "Evaluation report data...",  # TODO: Implement actual report generation
                "evaluation_report.csv",
                help="Download detailed evaluation results as CSV"
            ) 