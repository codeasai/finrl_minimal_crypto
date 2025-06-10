import streamlit as st

def test_agent_ui():
    st.header("📊 Backtest RL Agent")
    agent_name = st.text_input("Agent Name to Test")
    if st.button("🔍 Run Backtest"):
        st.write(f"Running backtest for {agent_name}... (stub)")
        st.success("Backtest complete (placeholder)")
