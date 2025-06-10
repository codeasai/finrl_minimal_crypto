import os
import datetime
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR  # Now we can import from root

def get_agent_info(agent_name):
    """Get detailed information about an agent"""
    path = os.path.join(MODEL_DIR, f"{agent_name}")
    if os.path.exists(path):
        created_time = datetime.datetime.fromtimestamp(os.path.getctime(path))
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        size = os.path.getsize(path)
        return {
            "name": agent_name.replace("minimal_crypto_", ""),  # Remove prefix for display
            "type": "PPO" if "simple" not in agent_name else "PPO (Simple)",
            "created": created_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
            "size": f"{size/1024:.1f} KB"
        }
    return None

def list_agents():
    """List all available agents with their information"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    # List only the minimal_crypto models
    agents = [f for f in os.listdir(MODEL_DIR) if f.startswith("minimal_crypto_")]
    return [get_agent_info(agent) for agent in agents]

def create_agent():
    """Create a new agent with UI"""
    st.write("⚠️ Agents are created through training.")
    st.write("Please go to the Train section to create a new agent.")

def delete_agent(agent_name):
    """Delete an agent with confirmation"""
    # Add prefix back for deletion
    full_name = f"minimal_crypto_{agent_name}"
    path = os.path.join(MODEL_DIR, full_name)
    if os.path.exists(path):
        os.remove(path)
        # Also remove any associated files (like performance plots)
        plot_path = os.path.join(MODEL_DIR, f"{full_name}_performance.png")
        if os.path.exists(plot_path):
            os.remove(plot_path)
        return True
    return False

def manage_agents_ui():
    """UI for managing agents"""
    st.header("🤖 Manage Agents")
    
    # List all agents in a table
    agents = list_agents()
    if not agents:
        st.info("No trained agents found. Go to the Train section to create one!")
        return
        
    # Create a table with agent information
    agent_df = {
        "Name": [],
        "Type": [],
        "Created": [],
        "Last Modified": [],
        "Size": [],
        "Actions": []
    }
    
    for agent in agents:
        if agent:  # Check if agent info exists
            agent_df["Name"].append(agent["name"])
            agent_df["Type"].append(agent["type"])
            agent_df["Created"].append(agent["created"])
            agent_df["Last Modified"].append(agent["last_modified"])
            agent_df["Size"].append(agent["size"])
            agent_df["Actions"].append(agent["name"])
    
    if agent_df["Name"]:  # Check if we have any valid agents
        st.dataframe(agent_df, hide_index=True)
        
        # Delete agent section
        st.subheader("Delete Agent")
        agent_to_delete = st.selectbox("Select agent to delete", agent_df["Name"])
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Delete"):
                if st.session_state.get("confirm_delete") == agent_to_delete:
                    if delete_agent(agent_to_delete):
                        st.success(f"Deleted agent {agent_to_delete}")
                        st.session_state.confirm_delete = None
                        st.rerun()
                    else:
                        st.error(f"Failed to delete agent {agent_to_delete}")
                else:
                    st.session_state.confirm_delete = agent_to_delete
                    st.warning(f"Click delete again to confirm deleting {agent_to_delete}")
        with col2:
            if st.session_state.get("confirm_delete"):
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete = None
                    st.rerun()
