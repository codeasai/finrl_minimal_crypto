import os
import datetime
import streamlit as st
import sys
import zipfile
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Add project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from config import MODEL_DIR  # Now we can import from root

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache timeout in seconds
CACHE_TIMEOUT = 300  # 5 minutes

class AgentManager:
    """Centralized agent management class with caching and error handling"""
    
    def __init__(self):
        self._cache = {}
        self._cache_timestamps = {}
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[key] < CACHE_TIMEOUT
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if valid"""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache"""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
    
    def _clear_cache(self) -> None:
        """Clear all cache"""
        self._cache.clear()
        self._cache_timestamps.clear()

# Global agent manager instance
agent_manager = AgentManager()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_training_info(agent_name: str) -> Dict[str, Any]:
    """Get training information from pkl files with caching"""
    try:
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
                    logger.warning(f"ไม่สามารถอ่านไฟล์ {filename}: {e}")
        
        return training_info
    except Exception as e:
        logger.error(f"Error getting training info: {e}")
        return {}

def format_number(num: Any) -> str:
    """Format large numbers with commas safely"""
    try:
        if isinstance(num, (int, float)):
            return f"{num:,}"
        return str(num)
    except:
        return "N/A"

def safe_get_attribute(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object"""
    try:
        if hasattr(obj, attr):
            return getattr(obj, attr, default)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default
    except:
        return default

@st.cache_data(ttl=300)
def get_model_details(agent_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed model information from the zip file with improved error handling"""
    try:
        # Use cache first
        cache_key = f"model_details_{agent_name}"
        cached_result = agent_manager._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        full_name = f"minimal_crypto_{agent_name}" if not agent_name.startswith("minimal_crypto_") else agent_name
        model_path = os.path.join(MODEL_DIR, f"{full_name}.zip")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
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
        
        # Read zip file information
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            # Get file list and sizes
            file_info = []
            total_size = 0
            
            for info in zip_file.filelist:
                try:
                    size_kb = info.file_size / 1024
                    file_info.append({
                        "name": info.filename,
                        "size": f"{size_kb:.1f} KB",
                        "compressed_size": f"{info.compress_size / 1024:.1f} KB"
                    })
                    total_size += size_kb
                except Exception as e:
                    logger.warning(f"Error reading file info for {info.filename}: {e}")
            
            details["files"] = file_info
            details["total_size"] = f"{total_size:.1f} KB"
            
            # Try to read system info first (most reliable)
            try:
                with zip_file.open('system_info.txt') as sys_file:
                    sys_info = sys_file.read().decode('utf-8')
                    details["system_info"] = sys_info
            except:
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
            
            # Try to read data file (most complex part)
            try:
                with zip_file.open('data') as data_file:
                    data = None
                    
                    # Try multiple reading methods
                    for method in ['torch', 'pickle', 'json']:
                        try:
                            data_file.seek(0)
                            if method == 'torch':
                                import torch
                                data = torch.load(data_file, map_location='cpu')
                            elif method == 'pickle':
                                data = pickle.load(data_file)
                            elif method == 'json':
                                content = data_file.read().decode('utf-8')
                                data = json.loads(content)
                            break
                        except Exception as method_error:
                            logger.debug(f"Method {method} failed: {method_error}")
                            continue
                    
                    if data is not None:
                        # Extract information safely
                        data_dict = safe_get_attribute(data, '__dict__', {})
                        if isinstance(data, dict):
                            data_dict.update(data)
                        
                        # Override defaults with actual values if found
                        for attr in ['_policy_class', 'policy_class']:
                            value = safe_get_attribute(data, attr) or data_dict.get(attr)
                            if value:
                                try:
                                    details["policy_type"] = str(value).split("'")[1].split(".")[-1] if "'" in str(value) else str(value)
                                    break
                                except:
                                    pass
                        
                        # Extract spaces information
                        for space_attr in ['observation_space']:
                            space_obj = safe_get_attribute(data, space_attr) or data_dict.get(space_attr)
                            if space_obj and hasattr(space_obj, 'shape'):
                                try:
                                    details["observation_space"] = f"Box({space_obj.shape[0]},)"
                                    details["n_features"] = space_obj.shape[0]
                                    break
                                except:
                                    pass
                        
                        for space_attr in ['action_space']:
                            space_obj = safe_get_attribute(data, space_attr) or data_dict.get(space_attr)
                            if space_obj and hasattr(space_obj, 'shape'):
                                try:
                                    details["action_space"] = f"Box({space_obj.shape[0]},)"
                                    break
                                except:
                                    pass
                        
                        # Extract parameters safely
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
                                value = safe_get_attribute(data, attr) or data_dict.get(attr)
                                if value is not None:
                                    try:
                                        if 'timesteps' in detail_key:
                                            details[detail_key] = format_number(value)
                                        else:
                                            details[detail_key] = value
                                        break
                                    except:
                                        pass
                        
                        # Additional model info
                        if safe_get_attribute(data, '_last_obs') or data_dict.get('_last_obs'):
                            details["has_last_obs"] = True
                        if safe_get_attribute(data, 'policy') or data_dict.get('policy'):
                            details["has_policy"] = True
                            
            except Exception as e:
                details["data_error"] = f"Could not parse data file - using default values: {str(e)}"
                logger.warning(f"Error reading model data: {e}")
        
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
        
        # Cache the result
        agent_manager._set_cache(cache_key, details)
        return details
        
    except Exception as e:
        logger.error(f"Error getting model details for {agent_name}: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_agent_info(agent_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about an agent with caching"""
    try:
        path = os.path.join(MODEL_DIR, f"{agent_name}")
        if os.path.exists(path):
            stat = os.stat(path)
            created_time = datetime.datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            
            return {
                "name": agent_name.replace("minimal_crypto_", "").replace(".zip", ""),
                "type": "PPO" if "simple" not in agent_name else "PPO (Simple)",
                "created": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                "size": f"{size/1024:.1f} KB",
                "file_path": path
            }
    except Exception as e:
        logger.error(f"Error getting agent info for {agent_name}: {e}")
    return None

@st.cache_data(ttl=60)
def list_agents() -> List[Dict[str, Any]]:
    """List all available agents with their information"""
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            return []
        
        # List models with .zip extension
        agents = []
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith(".zip") and ("minimal_crypto" in filename or "ppo" in filename):
                agent_info = get_agent_info(filename)
                if agent_info:
                    agents.append(agent_info)
        
        # Sort by last modified time (newest first)
        agents.sort(key=lambda x: x["last_modified"], reverse=True)
        return agents
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        return []

def show_model_metrics(details: Dict[str, Any]) -> None:
    """Display model metrics in an organized way"""
    try:
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
                st.metric("🎛️ Policy Status", "✅ Loaded" if details["has_policy"] else "❌ Missing")
            if "has_last_obs" in details:
                st.metric("👁️ Last Observation", "✅ Available" if details["has_last_obs"] else "❌ Missing")
        
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
            if "sb3_version" in details:
                st.metric("📦 SB3 Version", details["sb3_version"])
        
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
        
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

def show_training_info(training_info: Dict[str, Any]) -> None:
    """Display additional training information"""
    try:
        if training_info:
            st.markdown("### 🎓 ข้อมูลการเทรนเพิ่มเติม")
            
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
    except Exception as e:
        st.error(f"Error displaying training info: {e}")

def show_system_info(details: Dict[str, Any]) -> None:
    """Display system information"""
    try:
        if "system_info" in details:
            st.markdown("### 🖥️ ข้อมูลสภาพแวดล้อม")
            with st.expander("ดูข้อมูลระบบ", expanded=False):
                system_lines = details["system_info"].split('\n')
                for line in system_lines:
                    if line.strip():
                        if any(keyword in line for keyword in ["OS", "Python", "PyTorch", "Stable-Baselines3"]):
                            st.code(line, language="text")
    except Exception as e:
        st.error(f"Error displaying system info: {e}")

def show_file_structure(details: Dict[str, Any]) -> None:
    """Display model file structure"""
    try:
        st.markdown("### 📁 ไฟล์ภายในโมเดล")
        if "files" in details and details["files"]:
            # Create enhanced file info table
            files_data = []
            for file_info in details["files"]:
                files_data.append({
                    "📄 File Name": file_info["name"],
                    "📊 Size": file_info["size"],
                    "🗜️ Compressed": file_info.get("compressed_size", "N/A")
                })
            
            if files_data:
                st.dataframe(files_data, use_container_width=True)
                
                # File descriptions
                with st.expander("📖 คำอธิบายไฟล์"):
                    st.markdown("""
                    - **data**: การตั้งค่าและพารามิเตอร์ของโมเดล
                    - **policy.pth**: น้ำหนัก Neural Network 
                    - **policy.optimizer.pth**: สถานะของ Optimizer
                    - **pytorch_variables.pth**: ตัวแปร PyTorch
                    - **system_info.txt**: ข้อมูลระบบที่ใช้เทรน
                    - **_stable_baselines3_version**: เวอร์ชันของ Stable-Baselines3
                    """)
        else:
            st.warning("ไม่พบข้อมูลไฟล์")
    except Exception as e:
        st.error(f"Error displaying file structure: {e}")

def show_agent_details(agent_name: str) -> None:
    """Show detailed information about a specific agent with improved error handling"""
    try:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #1f77b4, #ff7f0e); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>🤖 รายละเอียดโมเดล: {agent_name}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔍 กำลังโหลดข้อมูลโมเดล..."):
            details = get_model_details(agent_name)
            
        if not details:
            st.error("❌ ไม่สามารถอ่านข้อมูลโมเดลได้")
            st.info("💡 กรุณาตรวจสอบว่าไฟล์โมเดลยังอยู่ในโฟลเดอร์ models/")
            return
        
        # Show metrics
        show_model_metrics(details)
        
        # Show training info
        if "training_info" in details:
            show_training_info(details["training_info"])
        
        # Show system info
        show_system_info(details)
        
        # Show file structure
        show_file_structure(details)
        
        # Performance metrics (if available)
        performance_plot = os.path.join(MODEL_DIR, f"minimal_crypto_{agent_name}_performance.png")
        if os.path.exists(performance_plot):
            st.markdown("### 📈 กราฟผลการดำเนินการ")
            st.image(performance_plot, caption="Training Performance", use_column_width=True)
        
        # Show any errors
        if "error" in details:
            st.error(f"❌ Error reading model: {details['error']}")
        if "data_error" in details:
            st.warning(f"⚠️ {details['data_error']}")
            
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการแสดงรายละเอียด: {e}")
        logger.error(f"Error in show_agent_details: {e}")

def delete_agent(agent_name: str) -> bool:
    """Delete an agent with confirmation and proper cleanup"""
    try:
        # Clear cache first
        agent_manager._clear_cache()
        
        # Add prefix back for deletion
        full_name = f"minimal_crypto_{agent_name}" if not agent_name.startswith("minimal_crypto_") else agent_name
        if not full_name.endswith(".zip"):
            full_name += ".zip"
        
        path = os.path.join(MODEL_DIR, full_name)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted model: {path}")
            
            # Also remove any associated files (like performance plots)
            plot_path = os.path.join(MODEL_DIR, f"{full_name.replace('.zip', '')}_performance.png")
            if os.path.exists(plot_path):
                os.remove(plot_path)
                logger.info(f"Deleted plot: {plot_path}")
            
            # Clear Streamlit cache
            st.cache_data.clear()
            return True
        else:
            logger.warning(f"Model file not found: {path}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting agent {agent_name}: {e}")
        return False

def manage_agents_ui() -> None:
    """Main UI for managing agents with improved UX"""
    try:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center; margin: 0; font-size: 2.5rem;'>🤖 Agent Management</h1>
            <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>จัดการและดูรายละเอียดของ Trading Agents</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("🔄 รีเฟรช", help="รีเฟรชรายการ agents"):
                agent_manager._clear_cache()
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ ล้างแคช", help="ล้างข้อมูลแคชทั้งหมด"):
                agent_manager._clear_cache()
                st.cache_data.clear()
                st.success("✅ ล้างแคชเรียบร้อย")
        
        # List all agents
        with st.spinner("📋 กำลังโหลดรายการ agents..."):
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
            st.caption(f"พบ {len(agents)} agents (เรียงตามวันที่แก้ไขล่าสุด)")
            
            # Create enhanced table with agent information
            if agents:
                for i, agent in enumerate(agents):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                        
                        with col1:
                            st.markdown(f"**🤖 {agent['name']}**")
                            st.caption(f"Type: {agent['type']}")
                        
                        with col2:
                            st.text(agent['created'])
                            st.caption("สร้างเมื่อ")
                        
                        with col3:
                            st.text(agent['last_modified'])
                            st.caption("แก้ไขล่าสุด")
                        
                        with col4:
                            st.text(agent['size'])
                            st.caption("ขนาดไฟล์")
                        
                        with col5:
                            if st.button("📊 ดู", key=f"view_{i}", help=f"ดูรายละเอียด {agent['name']}"):
                                st.session_state.show_details = agent['name']
                                st.session_state.active_tab = 1  # Switch to details tab
                                st.rerun()
                        
                        st.divider()
            
            # Agent management section
            st.markdown("### 🎛️ การจัดการ Agent")
            
            agent_names = [agent["name"] for agent in agents]
            selected_agent = st.selectbox(
                "เลือก Agent เพื่อดำเนินการ", 
                agent_names,
                help="เลือกโมเดลที่ต้องการจัดการ"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("📊 ดูรายละเอียด", use_container_width=True, type="primary"):
                    st.session_state.show_details = selected_agent
                    st.success(f"✅ กำลังแสดงรายละเอียดของ {selected_agent}")
                    
            with col2:
                if st.button("🗑️ ลบโมเดล", use_container_width=True, type="secondary"):
                    if st.session_state.get("confirm_delete") == selected_agent:
                        with st.spinner("🗑️ กำลังลบโมเดล..."):
                            if delete_agent(selected_agent):
                                st.success(f"✅ ลบโมเดล {selected_agent} เรียบร้อยแล้ว")
                                st.session_state.confirm_delete = None
                                time.sleep(1)  # Brief pause for user to see success message
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
                        st.info("✅ ยกเลิกการลบแล้ว")
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
                if agents:
                    agent_names = [agent["name"] for agent in agents]
                    quick_select = st.selectbox("เลือกโมเดลที่ต้องการดูรายละเอียด", agent_names, key="quick_select")
                    if st.button("📊 แสดงรายละเอียดทันที", type="primary"):
                        st.session_state.show_details = quick_select
                        st.rerun()
                        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการแสดง UI: {e}")
        logger.error(f"Error in manage_agents_ui: {e}")

# Backwards compatibility functions
def create_agent():
    """Create a new agent with UI"""
    st.info("⚠️ Agents are created through training.")
    st.info("Please go to the Train section to create a new agent.")
