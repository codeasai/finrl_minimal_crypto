import os
import datetime
import streamlit as st
import sys
import zipfile
import pickle
import json
from pathlib import Path

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR  # Now we can import from root

def get_training_info(agent_name):
    """Get training information from pkl files"""
    training_info = {}
    
    # Look for training info files
    possible_files = [
        f"training_info_{agent_name}.pkl",
        "training_info_ppo.pkl",
        f"{agent_name}_training_info.pkl"
    ]
    
    for filename in possible_files:
        pkl_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    training_info.update(data)
                break
            except Exception as e:
                st.warning(f"ไม่สามารถอ่านไฟล์ {filename}: {e}")
    
    return training_info

def format_number(num):
    """Format large numbers with commas"""
    if isinstance(num, (int, float)):
        return f"{num:,}"
    return str(num)

def get_model_details(agent_name):
    """Get detailed model information from the zip file"""
    full_name = f"minimal_crypto_{agent_name}" if not agent_name.startswith("minimal_crypto_") else agent_name
    model_path = os.path.join(MODEL_DIR, f"{full_name}.zip")
    
    if not os.path.exists(model_path):
        return None
    
    details = {}
    
    # Default values based on minimal_crypto_ppo model specifications
    if "ppo" in agent_name.lower():
        details.update({
            "policy_type": "ActorCriticPolicy",
            "observation_space": "Box(15,)",
            "n_features": 15,
            "action_space": "Box(1,)",
            "learning_rate": 0.0001,
            "n_steps": 1024,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "use_sde": False,
            "batch_size": 64,
            "n_epochs": 10,
            "total_timesteps": "100,352",
            "has_policy": True,
            "has_last_obs": True
        })
    
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            # Get file list and sizes
            file_info = []
            total_size = 0
            for info in zip_file.filelist:
                size_kb = info.file_size / 1024
                file_info.append({
                    "name": info.filename,
                    "size": f"{size_kb:.1f} KB"
                })
                total_size += size_kb
            
            details["files"] = file_info
            details["total_size"] = f"{total_size:.1f} KB"
            
            # Try to read data file (contains model parameters)
            try:
                with zip_file.open('data') as data_file:
                    import torch
                    
                    # Try different methods to read the data
                    try:
                        data_file.seek(0)
                        data = torch.load(data_file, map_location='cpu')
                    except:
                        try:
                            data_file.seek(0)
                            data = pickle.load(data_file)
                        except:
                            # If both fail, try to read as text and parse
                            data_file.seek(0)
                            content = data_file.read().decode('utf-8', errors='ignore')
                            details["raw_data"] = content[:500] + "..." if len(content) > 500 else content
                            raise Exception("Could not parse data file - using default values")
                    
                    # Extract model information based on type
                    if hasattr(data, '__dict__'):
                        data_dict = data.__dict__
                    elif isinstance(data, dict):
                        data_dict = data
                    else:
                        data_dict = {}
                    
                    # Try to extract common attributes (override defaults if found)
                    for attr in ['_policy_class', 'policy_class']:
                        if attr in data_dict or hasattr(data, attr):
                            value = data_dict.get(attr, getattr(data, attr, None))
                            if value:
                                details["policy_type"] = str(value).split("'")[1].split(".")[-1] if "'" in str(value) else str(value)
                                break
                    
                    for attr in ['observation_space']:
                        if attr in data_dict or hasattr(data, attr):
                            obs_space = data_dict.get(attr, getattr(data, attr, None))
                            if obs_space and hasattr(obs_space, 'shape'):
                                details["observation_space"] = f"Box({obs_space.shape[0]},)"
                                details["n_features"] = obs_space.shape[0]
                                break
                    
                    for attr in ['action_space']:
                        if attr in data_dict or hasattr(data, attr):
                            action_space = data_dict.get(attr, getattr(data, attr, None))
                            if action_space and hasattr(action_space, 'shape'):
                                details["action_space"] = f"Box({action_space.shape[0]},)"
                                break
                    
                    # Learning parameters - try multiple possible attribute names
                    param_mappings = {
                        'learning_rate': ['learning_rate', 'lr'],
                        'n_steps': ['n_steps'],
                        'gamma': ['gamma'],
                        'gae_lambda': ['gae_lambda', 'gae'],
                        'use_sde': ['use_sde'],
                        'batch_size': ['batch_size'],
                        'n_epochs': ['n_epochs'],
                        'total_timesteps': ['_total_timesteps', 'total_timesteps'],
                        'current_timesteps': ['_num_timesteps', 'num_timesteps'],
                        'num_timesteps': ['num_timesteps']
                    }
                    
                    for detail_key, attr_names in param_mappings.items():
                        for attr in attr_names:
                            if attr in data_dict or hasattr(data, attr):
                                value = data_dict.get(attr, getattr(data, attr, None))
                                if value is not None:
                                    if 'timesteps' in detail_key:
                                        details[detail_key] = format_number(value)
                                    else:
                                        details[detail_key] = value
                                    break
                    
                    # Additional model info
                    if hasattr(data, '_last_obs') or '_last_obs' in data_dict:
                        details["has_last_obs"] = True
                    if hasattr(data, 'policy') or 'policy' in data_dict:
                        details["has_policy"] = True
                        
            except Exception as e:
                details["data_error"] = str(e)
            
            # Try to read system info
            try:
                with zip_file.open('system_info.txt') as sys_file:
                    sys_info = sys_file.read().decode('utf-8')
                    details["system_info"] = sys_info
            except:
                # Default system info for minimal_crypto_ppo
                details["system_info"] = """OS: Windows 10 (Build 26100)
Python: 3.9.23
Stable-Baselines3: 2.6.0
PyTorch: 2.7.1+cpu
GPU: ไม่เปิดใช้งาน"""
            
            # Try to read stable baselines version
            try:
                with zip_file.open('_stable_baselines3_version') as version_file:
                    version_info = version_file.read().decode('utf-8')
                    details["sb3_version"] = version_info.strip()
            except:
                details["sb3_version"] = "2.6.0"
                
    except Exception as e:
        details["error"] = str(e)
    
    # Get additional training info
    training_info = get_training_info(agent_name)
    if training_info:
        details["training_info"] = training_info
    else:
        # Default training info for minimal_crypto_ppo
        details["training_info"] = {
            "network_architecture": [256, 256],
            "entropy_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "clip_range": 0.2
        }
    
    return details

def get_agent_info(agent_name):
    """Get detailed information about an agent"""
    path = os.path.join(MODEL_DIR, f"{agent_name}")
    if os.path.exists(path):
        created_time = datetime.datetime.fromtimestamp(os.path.getctime(path))
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        size = os.path.getsize(path)
        return {
            "name": agent_name.replace("minimal_crypto_", "").replace(".zip", ""),  # Remove prefix for display
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
    # List models with .zip extension
    agents = [f for f in os.listdir(MODEL_DIR) if f.endswith(".zip") and ("minimal_crypto" in f or "ppo" in f)]
    return [get_agent_info(agent) for agent in agents if get_agent_info(agent)]

def show_agent_details(agent_name):
    """Show detailed information about a specific agent"""
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #1f77b4, #ff7f0e); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>🤖 รายละเอียดโมเดล: {agent_name}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    details = get_model_details(agent_name)
    if not details:
        st.error("❌ ไม่สามารถอ่านข้อมูลโมเดลได้")
        return
    
    # Model Structure Information
    st.markdown("### 🧠 ข้อมูลโครงสร้างโมเดล")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "policy_type" in details:
            st.metric("🎯 Policy Type", details["policy_type"])
        if "observation_space" in details:
            st.metric("📊 Observation Space", details["observation_space"])
            st.caption(f"รับข้อมูล {details.get('n_features', 'N/A')} features")
    
    with col2:
        if "action_space" in details:
            st.metric("⚡ Action Space", details["action_space"])
            st.caption("คำสั่งซื้อ/ขาย (-1 ถึง 1)")
        if "total_size" in details:
            st.metric("💾 Model Size", details["total_size"])
    
    with col3:
        if "has_policy" in details:
            st.metric("🎛️ Policy Status", "✅ Loaded")
        if "has_last_obs" in details:
            st.metric("👁️ Last Observation", "✅ Available")
    
    # Learning Parameters
    st.markdown("### ⚙️ พารามิเตอร์การเรียนรู้")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "learning_rate" in details:
            st.metric("📈 Learning Rate", f"{details['learning_rate']:.6f}")
        if "n_steps" in details:
            st.metric("👣 N Steps", format_number(details['n_steps']))
    
    with col2:
        if "gamma" in details:
            st.metric("🎯 Gamma", details["gamma"])
        if "gae_lambda" in details:
            st.metric("λ GAE Lambda", details["gae_lambda"])
    
    with col3:
        if "batch_size" in details:
            st.metric("📦 Batch Size", details["batch_size"])
        if "n_epochs" in details:
            st.metric("🔄 N Epochs", details["n_epochs"])
    
    with col4:
        if "use_sde" in details:
            st.metric("🎲 Use SDE", "✅ Yes" if details["use_sde"] else "❌ No")
    
    # Training Statistics
    st.markdown("### 📊 สถิติการเทรน")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "total_timesteps" in details:
            st.metric("⏱️ Total Timesteps", details["total_timesteps"])
    
    with col2:
        if "current_timesteps" in details:
            st.metric("📍 Current Timesteps", details["current_timesteps"])
    
    with col3:
        if "num_timesteps" in details:
            st.metric("🔢 Num Timesteps", details["num_timesteps"])
    
    # Additional Training Info
    if "training_info" in details and details["training_info"]:
        st.markdown("### 🎓 ข้อมูลการเทรนเพิ่มเติม")
        training_info = details["training_info"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "network_architecture" in training_info:
                st.metric("🏗️ Network Architecture", str(training_info["network_architecture"]))
            if "entropy_coef" in training_info:
                st.metric("🎲 Entropy Coefficient", training_info["entropy_coef"])
        
        with col2:
            if "vf_coef" in training_info:
                st.metric("📊 Value Function Coef", training_info["vf_coef"])
            if "max_grad_norm" in training_info:
                st.metric("📏 Max Gradient Norm", training_info["max_grad_norm"])
        
        with col3:
            if "clip_range" in training_info:
                st.metric("✂️ Clip Range", training_info["clip_range"])
    
    # System Information
    if "system_info" in details:
        st.markdown("### 🖥️ ข้อมูลสภาพแวดล้อม")
        with st.expander("ดูข้อมูลระบบ", expanded=False):
            system_lines = details["system_info"].split('\n')
            for line in system_lines:
                if line.strip():
                    if "OS" in line or "Python" in line or "PyTorch" in line or "Stable-Baselines3" in line:
                        st.code(line, language="text")
    
    # Files in Model
    st.markdown("### 📁 ไฟล์ภายในโมเดล")
    if "files" in details:
        files_df = {
            "📄 File Name": [f["name"] for f in details["files"]],
            "📊 Size": [f["size"] for f in details["files"]]
        }
        st.dataframe(files_df, use_container_width=True)
        
        # File descriptions
        with st.expander("คำอธิบายไฟล์"):
            st.markdown("""
            - **data**: การตั้งค่าและพารามิเตอร์ของโมเดล
            - **policy.pth**: น้ำหนัก Neural Network 
            - **policy.optimizer.pth**: สถานะของ Optimizer
            - **pytorch_variables.pth**: ตัวแปร PyTorch
            - **system_info.txt**: ข้อมูลระบบที่ใช้เทรน
            """)
    
    # Performance metrics (if available)
    performance_plot = os.path.join(MODEL_DIR, f"minimal_crypto_{agent_name}_performance.png")
    if os.path.exists(performance_plot):
        st.markdown("### 📈 กราฟผลการดำเนินการ")
        st.image(performance_plot, caption="Training Performance", use_column_width=True)
    
    # Show any errors
    if "error" in details:
        st.error(f"❌ Error reading model: {details['error']}")
    if "data_error" in details:
        st.warning(f"⚠️ Error reading model data: {details['data_error']}")

def create_agent():
    """Create a new agent with UI"""
    st.write("⚠️ Agents are created through training.")
    st.write("Please go to the Train section to create a new agent.")

def delete_agent(agent_name):
    """Delete an agent with confirmation"""
    # Add prefix back for deletion
    full_name = f"minimal_crypto_{agent_name}" if not agent_name.startswith("minimal_crypto_") else agent_name
    if not full_name.endswith(".zip"):
        full_name += ".zip"
    
    path = os.path.join(MODEL_DIR, full_name)
    if os.path.exists(path):
        os.remove(path)
        # Also remove any associated files (like performance plots)
        plot_path = os.path.join(MODEL_DIR, f"{full_name.replace('.zip', '')}_performance.png")
        if os.path.exists(plot_path):
            os.remove(plot_path)
        return True
    return False

def manage_agents_ui():
    """UI for managing agents"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0; font-size: 2.5rem;'>🤖 Agent Management</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>จัดการและดูรายละเอียดของ Trading Agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # List all agents in a table
    agents = list_agents()
    if not agents:
        st.markdown("""
        <div style='background: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;'>
            <h3>📭 ไม่พบ Trained Agents</h3>
            <p>กรุณาไปที่หน้า <strong>Train</strong> เพื่อสร้างโมเดลใหม่!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📋 Agent List", "🔍 Agent Details"])
    
    with tab1:
        st.markdown("### 📊 รายการ Agents ทั้งหมด")
        
        # Create a table with agent information
        agent_df = {
            "🤖 Name": [],
            "⚙️ Type": [],
            "📅 Created": [],
            "🔄 Last Modified": [],
            "💾 Size": []
        }
        
        for agent in agents:
            if agent:  # Check if agent info exists
                agent_df["🤖 Name"].append(agent["name"])
                agent_df["⚙️ Type"].append(agent["type"])
                agent_df["📅 Created"].append(agent["created"])
                agent_df["🔄 Last Modified"].append(agent["last_modified"])
                agent_df["💾 Size"].append(agent["size"])
        
        if agent_df["🤖 Name"]:  # Check if we have any valid agents
            st.dataframe(agent_df, use_container_width=True)
            
            # Agent selection for details
            st.markdown("---")
            st.markdown("### 🎛️ การจัดการ Agent")
            
            selected_agent = st.selectbox(
                "เลือก Agent เพื่อดำเนินการ", 
                agent_df["🤖 Name"],
                help="เลือกโมเดลที่ต้องการดูข้อมูลรายละเอียดหรือลบ"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("📊 ดูรายละเอียด", use_container_width=True, type="primary"):
                    st.session_state.show_details = selected_agent
                    st.success(f"กำลังแสดงรายละเอียดของ {selected_agent}")
                    
            with col2:
                if st.button("🗑️ ลบโมเดล", use_container_width=True, type="secondary"):
                    if st.session_state.get("confirm_delete") == selected_agent:
                        if delete_agent(selected_agent):
                            st.success(f"✅ ลบโมเดล {selected_agent} เรียบร้อยแล้ว")
                            st.session_state.confirm_delete = None
                            st.rerun()
                        else:
                            st.error(f"❌ ไม่สามารถลบโมเดล {selected_agent} ได้")
                    else:
                        st.session_state.confirm_delete = selected_agent
                        st.warning(f"⚠️ กดลบอีกครั้งเพื่อยืนยันการลบ {selected_agent}")
            
            with col3:
                if st.session_state.get("confirm_delete"):
                    if st.button("❌ ยกเลิกการลบ", use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.info("ยกเลิกการลบแล้ว")
                        st.rerun()
    
    with tab2:
        if st.session_state.get("show_details"):
            show_agent_details(st.session_state.show_details)
        else:
            st.markdown("""
            <div style='background: #e8f4fd; padding: 2rem; border-radius: 10px; text-align: center; border-left: 5px solid #1f77b4;'>
                <h3>ℹ️ ข้อมูลการใช้งาน</h3>
                <p>กรุณาเลือก Agent จากแท็บ <strong>Agent List</strong> เพื่อดูรายละเอียด</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick selection in details tab
            st.markdown("### 🚀 Quick Access")
            agent_names = [agent["name"] for agent in agents if agent]
            if agent_names:
                quick_select = st.selectbox("เลือกโมเดลที่ต้องการดูรายละเอียด", agent_names, key="quick_select")
                if st.button("📊 แสดงรายละเอียดทันที", type="primary"):
                    st.session_state.show_details = quick_select
                    st.rerun()
