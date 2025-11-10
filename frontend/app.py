"""
Kaggle AI Agent System - Streamlit Frontend

Support multiple AI architectures (ReAct, RAG, Multi-Agent) to automatically solve Kaggle problems
"""

import streamlit as st
import asyncio
from pathlib import Path
from typing import Any
import time
import os
import sys
import tempfile
import shutil
import subprocess
# from datetime import datetime

# Ensure the repository root is on sys.path so top-level imports like `backend.*`
# work when running this file directly (for example: `streamlit run frontend/app.py`).
#
# We compute the repo root as the parent directory of the `frontend` folder and
# insert it at the head of sys.path if it's not already present. This is a
# lightweight fix for development and deployed Streamlit runs where the CWD
# might be `frontend` and Python would otherwise not find the `backend` package.
REPO_ROOT = Path(__file__).resolve().parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from backend.agents import ReactAgent
from backend.kaggle import KaggleDataFetcher, CompetitionInfo

# Try to import RAG framework (optional)
try:
    from backend.agents.rag_agent import RAGAgent
    from backend.RAG_tool.config import RAGConfig
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False
# from backend.evaluation import MetricsCalculator

# Page configuration
st.set_page_config(
    page_title="Kaggle AI Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .agent-card {
        border: 2px solid #e0e0e0;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    .agent-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .code-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'competition_info' not in st.session_state:
        st.session_state.competition_info = None
    if 'agent_result' not in st.session_state:
        st.session_state.agent_result = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = []


def parse_kaggle_url(url: str) -> str:
    """Parse Kaggle competition URL and extract competition name"""
    if '/competitions/' in url:
        parts = url.split('/competitions/')
        if len(parts) > 1:
            competition_name = parts[1].strip('/').split('/')[0]
            return competition_name
    return url.strip()


def fetch_competition_data(competition_name: str) -> CompetitionInfo:
    """Fetch competition data"""
    return KaggleDataFetcher().fetch_complete_info(competition_name)


async def run_rag_agent(competition_info: CompetitionInfo, config_params: dict):
    """Run RAG Agent and return a result-like object compatible with UI"""
    from types import SimpleNamespace

    if not RAG_AVAILABLE:
        # Return a minimal failure-like namespace
        ns = SimpleNamespace()
        ns.status = SimpleNamespace(value="failed")
        ns.generated_code = "# RAG system unavailable"
        ns.retrieved_knowledge = []
        ns.reasoning_steps = ["RAG framework not available"]
        ns.execution_time = 0.0
        ns.retrieval_count = 0
        ns.llm_calls = 0
        ns.error_message = "RAG framework not properly installed"
        ns.submission_file_path = None
        ns.thoughts = []
        ns.actions = []
        ns.observations = []
        ns.total_time = 0.0
        ns.code_generation_time = 0.0
        ns.code_lines = 0
        return ns

    # Require OpenAI API key for RAG
    if not config_params.get('openai_api_key'):
        ns = SimpleNamespace()
        ns.status = SimpleNamespace(value="failed")
        ns.generated_code = ""
        ns.retrieved_knowledge = []
        ns.reasoning_steps = ["OpenAI API key not provided"]
        ns.execution_time = 0.0
        ns.retrieval_count = 0
        ns.llm_calls = 0
        ns.error_message = "Please set your OpenAI API key in the sidebar"
        ns.submission_file_path = None
        ns.thoughts = []
        ns.actions = []
        ns.observations = []
        ns.total_time = 0.0
        ns.code_generation_time = 0.0
        ns.code_lines = 0
        return ns

    try:
        os.environ['OPENAI_API_KEY'] = config_params.get('openai_api_key')

        rag_config = RAGConfig(
            llm_model=config_params.get('llm_model', 'gpt-4o-mini'),
            temperature=config_params.get('temperature', 0.3),
            max_tokens=config_params.get('max_tokens', 4000),
            max_retries=config_params.get('max_retries', 1),
            openai_api_key=config_params.get('openai_api_key')
        )

        agent = RAGAgent(rag_config)

        data_info = {
            'train_files': competition_info.train_files,
            'test_files': competition_info.test_files,
            'columns': competition_info.columns,
            'all_files_info': competition_info.extra_info.get('all_files', {})
        }

        result_obj = await agent.run(
            problem_description=f"Kaggle Competition: {competition_info.competition_name}",
            data_info=data_info
        )

        # Normalize result into a SimpleNamespace so UI can access attributes
        def _get(r, k, default=None):
            try:
                return getattr(r, k)
            except Exception:
                try:
                    return r.get(k, default)
                except Exception:
                    return default

        ns = SimpleNamespace()
        status_raw = _get(result_obj, 'status', 'completed')
        ns.status = SimpleNamespace(value=(status_raw.value if hasattr(status_raw, 'value') else str(status_raw)))
        ns.generated_code = _get(result_obj, 'generated_code', '')
        ns.retrieved_knowledge = _get(result_obj, 'retrieved_knowledge', [])
        ns.reasoning_steps = _get(result_obj, 'reasoning_steps', [])
        ns.execution_time = _get(result_obj, 'execution_time', 0.0)
        ns.retrieval_count = _get(result_obj, 'retrieval_count', 0)
        ns.llm_calls = _get(result_obj, 'llm_calls', 0)
        ns.error_message = _get(result_obj, 'error_message', _get(result_obj, 'execution_error', None))
        ns.submission_file_path = _get(result_obj, 'submission_path', _get(result_obj, 'submission_file_path', None))
        ns.thoughts = _get(result_obj, 'thoughts', []) or []
        ns.actions = _get(result_obj, 'actions', []) or []
        ns.observations = _get(result_obj, 'observations', []) or []
        ns.total_time = _get(result_obj, 'total_time', 0.0)
        ns.code_generation_time = _get(result_obj, 'code_generation_time', 0.0)
        ns.code_lines = _get(result_obj, 'code_lines', 0)

        return ns

    except Exception as e:
        ns = SimpleNamespace()
        ns.status = SimpleNamespace(value="failed")
        ns.generated_code = ""
        ns.retrieved_knowledge = []
        ns.reasoning_steps = [f'Error: {str(e)}']
        ns.execution_time = 0.0
        ns.retrieval_count = 0
        ns.llm_calls = 0
        ns.error_message = str(e)
        ns.submission_file_path = None
        ns.thoughts = []
        ns.actions = []
        ns.observations = []
        ns.total_time = 0.0
        ns.code_generation_time = 0.0
        ns.code_lines = 0
        return ns


async def _try_generate_submission(generated_code: str, competition_name: str):
    """Execute generated code in a temp dir to try to produce submission.csv (returns tuple)
    This mirrors the helper from app_only_for_RAG.py but is lightweight.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        data_files = ["train.csv", "test.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv", "sample_submission.csv"]
        for f in data_files:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(temp_dir, f))

        temp_file = os.path.join(temp_dir, 'generated_solution.py')
        debug_code = """
import traceback
import sys

def debug_hook(type, value, tb):
    print("Exception occurred:", file=sys.stderr)
    traceback.print_exception(type, value, tb, file=sys.stderr)

sys.excepthook = debug_hook

""" + generated_code

        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(debug_code)

        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=300, cwd=temp_dir)

        submission_path = os.path.join(temp_dir, 'submission.csv')
        if os.path.exists(submission_path):
            with open(submission_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # return True and content
            shutil.rmtree(temp_dir)
            return True, content
        else:
            # cleanup and return False
            shutil.rmtree(temp_dir)
            return False, None

    except Exception:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        return False, None


async def run_agent(
    agent_type: str,
    competition_info: CompetitionInfo,
    config_params: dict
 ) -> Any:
    """Run AI Agent"""

    # Initialize agent (minimal ReactAgent)
    agent = ReactAgent(
        llm_model=config_params.get('llm_model', 'gpt-4o-mini'),
        temperature=config_params.get('temperature', 0.3),
        max_tokens=config_params.get('max_tokens', 4000),
        timeout=config_params.get('max_execution_time', 600),
        competition_name=competition_info.competition_name,
    )

    # Build a minimal problem analysis and data_info to pass into the agent
    problem_analysis = {
        "problem_type": "unknown",
        "key_insights": [],
        "suggested_approach": "",
        "data_requirements": [],
        "problem_description": f"Kaggle Competition: {competition_info.competition_name}"
    }

    data_info = {
        'train_files': getattr(competition_info, 'train_files', []),
        'test_files': getattr(competition_info, 'test_files', []),
        'columns': getattr(competition_info, 'columns', {}),
        'all_files_info': (competition_info.extra_info.get('all_files') if getattr(competition_info, 'extra_info', None) else {})
    }

    # Run the full agent pipeline (generation + execution + fixes)
    raw = await agent.run(problem_description=problem_analysis['problem_description'], data_info=data_info)

    # Normalize to a SimpleNamespace expected by the UI
    from types import SimpleNamespace
    ns = SimpleNamespace()
    ns.status = SimpleNamespace(value=("completed" if raw.success else "failed"))
    ns.generated_code = raw.generated_code or ''
    ns.retrieved_knowledge = []
    ns.reasoning_steps = []
    ns.execution_time = raw.execution_time
    ns.retrieval_count = 0
    ns.llm_calls = raw.llm_calls
    ns.error_message = raw.error
    ns.submission_file_path = raw.submission_path
    ns.thoughts = []
    ns.actions = []
    ns.observations = raw.observations or []
    ns.total_time = raw.total_time
    ns.code_generation_time = raw.code_generation_time
    ns.code_lines = len(raw.generated_code.split('\n')) if raw.generated_code else 0

    return ns


def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.markdown('<div class="main-header">🤖 Kaggle AI Agent System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Agent System for Automatically Solving Kaggle Data Analysis Problems</div>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Agent type selection
        st.subheader("1. Select AI Architecture")
        agent_type = st.selectbox(
            "Agent Type",
            ["ReAct", "RAG"],
            help="Choose different AI agent architectures"
        )
        agent_type = agent_type.split()[0]  # Remove "(In Development)" tag
        
        # Agent introduction
        if agent_type == "ReAct":
            st.info("""
            **ReAct Architecture**
            - Reasoning-Action Loop
            - Automatic code generation and execution
            - Support automatic error fixing
            - Suitable for exploratory analysis
            """)
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            llm_model = st.selectbox(
                "LLM Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                index=0
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Control generation randomness, lower values are more deterministic"
            )
            
            max_retries = st.number_input(
                "Max Fix Attempts",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum attempts to fix code when execution fails"
            )
            
            max_execution_time = st.number_input(
                "Max Execution Time (seconds)",
                min_value=60,
                max_value=1200,
                value=600,
                help="Maximum time for single code execution"
            )
        
        config_params = {
            'llm_model': llm_model,
            'temperature': temperature,
            'max_retries': max_retries,
            'max_execution_time': max_execution_time
        }

        # If RAG is selected, ask for OpenAI API key
        if agent_type == "RAG":
            openai_api_key = st.text_input("OpenAI API Key", value=os.environ.get('OPENAI_API_KEY',''), type="password", help="Required for RAG operations")
            config_params['openai_api_key'] = openai_api_key
        
        st.divider()
        st.caption("💡 Tip: First run requires downloading Kaggle data, may take a few minutes")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📥 Input Task", "🚀 Execution Process", "📊 Results Analysis"])
    
    with tab1:
        st.header("Input Kaggle Competition Information")
        
        col1, col2 = st.columns([3, 1])
        
        def on_input_change():
            if st.session_state.competition_input:
                competition_name = parse_kaggle_url(st.session_state.competition_input)
                with st.spinner(f"Fetching competition data: {competition_name}..."):
                    try:
                        competition_info = fetch_competition_data(competition_name)
                        st.session_state.competition_info = competition_info
                        st.success("✅ Data fetched successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to fetch data: {str(e)}")

        with col1:
            competition_input = st.text_input(
                "Kaggle Competition URL or Name",
                placeholder="https://www.kaggle.com/competitions/your_kaggle_problem  or  your_kaggle_problem",
                help="Enter full URL or just the competition name and press Enter",
                key="competition_input",
                on_change=on_input_change
            )
        
        with col2:
            st.write("")  # Spacing for button alignment
            st.write("")
            fetch_button = st.button("📥 Fetch Data", type="primary", use_container_width=True)
        
        # Fetch competition data
        if fetch_button and st.session_state.competition_input:
            competition_name = parse_kaggle_url(st.session_state.competition_input)
            with st.spinner(f"Fetching competition data: {competition_name}..."):
                try:
                    competition_info = fetch_competition_data(competition_name)
                    st.session_state.competition_info = competition_info
                    st.success("✅ Data fetched successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to fetch data: {str(e)}")
        
        # Display competition information
        if st.session_state.competition_info:
            info = st.session_state.competition_info
            
            st.markdown("---")
            st.subheader("📋 Competition Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Competition Name", info.competition_name)
            with col2:
                st.metric("Training Rows", f"{info.train_shape[0]:,}" if info.train_shape else "N/A")
            with col3:
                st.metric("Test Rows", f"{info.test_shape[0]:,}" if info.test_shape else "N/A")
            with col4:
                st.metric("Features", info.train_shape[1] if info.train_shape else "N/A")
            
            # Data files
            with st.expander("📁 Data Files Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Training Files:**")
                    for f in info.train_files:
                        st.write(f"- {f}")
                
                with col2:
                    st.write("**Test Files:**")
                    for f in info.test_files:
                        st.write(f"- {f}")
                
                if info.extra_info and 'all_files' in info.extra_info:
                    st.markdown("---")
                    st.write("**📋 All Data Files Details:**")
                    all_files = info.extra_info['all_files']
                    for filename, file_info in all_files.items():
                        st.markdown(f"**📄 {filename}**")
                        st.write(f"- Columns: {len(file_info['columns'])}")
                        st.write(f"- Column Names: {', '.join(file_info['columns'][:10])}{'...' if len(file_info['columns']) > 10 else ''}")
                        if 'sample_data' in file_info and file_info['sample_data']:
                            with st.container():
                                st.caption("Sample Data (first 3 rows):")
                                st.json(file_info['sample_data'][:3], expanded=False)
                        st.write("")  # Empty line for separation
            
            # Start button
            st.markdown("---")
            if st.button("🚀 Start Generating Solution", type="primary", use_container_width=True, disabled=st.session_state.is_running):
                st.session_state.is_running = True
                st.success("🚀 Task started! Please switch to 'Execution Process' tab to view progress.")
                st.info("💡 Tip: System will automatically switch to execution process tab")
                st.rerun()
    
    with tab2:
        st.header("Execution Process")
        
        # Display current status
        if st.session_state.is_running:
            st.info("🔄 Task is running...")
        elif st.session_state.agent_result:
            st.success("✅ Task completed!")
        else:
            st.info("👈 Please start task execution in Input Task tab first")
        
        if st.session_state.is_running and st.session_state.competition_info:
            # Execute Agent
            info = st.session_state.competition_info
            
            # Progress display
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()
            
            with log_container:
                st.subheader("📝 Execution Log")
                log_area = st.empty()
            
            try:
                # Stage 1: Initialization
                status_text.info("🔧 Initializing Agent...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Stage 2: Run Agent
                status_text.info(f"🤖 {agent_type} Agent is analyzing problem...")
                progress_bar.progress(30)
                
                # Show execution prompt
                with st.spinner("Generating code, please wait..."):
                    if agent_type == "ReAct":
                        result = asyncio.run(run_agent(agent_type, info, config_params))
                    elif agent_type == "RAG":
                        result = asyncio.run(run_rag_agent(info, config_params))
                    else:
                        st.error(f"Agent type {agent_type} not supported")
                        result = None
                
                if result:
                    progress_bar.progress(90)
                    status_text.success("✅ Execution completed!")
                    progress_bar.progress(100)
                    
                    st.session_state.agent_result = result
                    st.session_state.is_running = False
                    
                    st.balloons()
                    st.success("🎉 Task completed! Please check Results Analysis tab.")
                    
                else:
                    st.error("❌ Agent execution failed")
                    st.session_state.is_running = False
                    
            except Exception as e:
                st.error(f"❌ Execution error: {str(e)}")
                st.session_state.is_running = False
        
        elif st.session_state.agent_result:
            st.info("✅ Task completed, please check Results Analysis tab")
        
        else:
            st.info("👈 Please fetch competition data and start execution in Input Task tab first")
    
    with tab3:
        st.header("Results Analysis")
        
        if st.session_state.agent_result:
            result = st.session_state.agent_result
            
            # Status overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card"><h3>Status</h3><h2>{}</h2></div>'.format(
                    "✅ Success" if result.status.value == "completed" else "❌ Failed"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card"><h3>Total Time</h3><h2>{:.1f}s</h2></div>'.format(
                    result.total_time
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card"><h3>LLM Calls</h3><h2>{}</h2></div>'.format(
                    result.llm_calls
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card"><h3>Code Lines</h3><h2>{}</h2></div>'.format(
                    result.code_lines
                ), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Generated code
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("💻 Generated Code")
                if result.generated_code:
                    st.code(result.generated_code, language='python', line_numbers=True)
                    
                    # Download code
                    st.download_button(
                        label="⬇️ Download Code",
                        data=result.generated_code,
                        file_name=f"{st.session_state.competition_info.competition_name}_solution.py",
                        mime="text/x-python"
                    )
                else:
                    st.warning("No code generated")
            
            with col2:
                st.subheader("📈 Execution Metrics")
                
                metrics_data = {
                    "Code Gen Time": f"{result.code_generation_time:.2f}s",
                    "Execution Time": f"{result.execution_time:.2f}s",
                    "Thought Steps": len(result.thoughts),
                    "Actions": len(result.actions),
                    "Observations": len(result.observations)
                }
                
                for key, value in metrics_data.items():
                    st.metric(key, value)
            
            # Submission file
            st.markdown("---")
            st.subheader("📦 Submission File")
            
            if result.submission_file_path and Path(result.submission_file_path).exists():
                st.success(f"✅ Submission generated: {result.submission_file_path}")
                
                # Read and display submission
                import pandas as pd
                try:
                    submission_df = pd.read_csv(result.submission_file_path)
                    st.dataframe(submission_df.head(10), use_container_width=True)
                    
                    # Download submission
                    with open(result.submission_file_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Submission.csv",
                            data=f,
                            file_name="submission.csv",
                            mime="text/csv",
                            type="primary"
                        )
                except Exception as e:
                    st.error(f"Failed to read submission file: {e}")
            else:
                st.warning("⚠️ No submission file generated")
            
            # Error information
            if result.error_message:
                st.markdown("---")
                st.subheader("⚠️ Error Information")
                st.error(result.error_message)
            
            # Detailed logs
            with st.expander("📋 Detailed Execution Log"):
                st.write("**Thought Process:**")
                for i, thought in enumerate(result.thoughts, 1):
                    st.write(f"{i}. {thought}")
                
                st.write("**Actions:**")
                for i, action in enumerate(result.actions, 1):
                    st.json(action)
                
                st.write("**Execution & Fix Process:**")
                fix_attempt = 0
                for i, obs in enumerate(result.observations, 1):
                    if "执行代码，尝试 #" in obs:
                        fix_attempt += 1
                        st.markdown(f"**Fix Attempt #{fix_attempt}:**")
                        continue
                    if "代码执行成功" in obs:
                        st.success(obs)
                    elif "执行失败" in obs:
                        st.error(obs)
                    elif "LLM 返回修正代码" in obs:
                        st.info(obs)
                    else:
                        st.write(f"- {obs}")
        
        else:
            st.info("👈 Please execute task first to view results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Kaggle AI Agent System | Powered by OpenAI & Streamlit</p>
        <p>Supported Architectures: ReAct ✅ | RAG 🔨 | Multi-Agent 🔨</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
