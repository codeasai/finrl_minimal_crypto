import streamlit as st
from pipeline.agent_manager import list_agents, create_agent, delete_agent, manage_agents_ui
from pipeline.train import train_agent_ui
from pipeline.test import test_agent_ui
from pipeline.evaluate import evaluate_agent_ui

st.set_page_config(layout="wide", page_title="Crypto RL Agent Dashboard", page_icon="🧠")

st.sidebar.title("🧠 Crypto RL UI")
section = st.sidebar.radio("Select Section", ["Create Agent", "Train", "Test", "Evaluate", "Manage"])

if section == "Create Agent":
    create_agent()
elif section == "Train":
    train_agent_ui()
elif section == "Test":
    test_agent_ui()
elif section == "Evaluate":
    evaluate_agent_ui()
elif section == "Manage":
    manage_agents_ui()
