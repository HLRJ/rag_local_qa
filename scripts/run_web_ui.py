# scripts/run_web_ui.py
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
app_file = current_dir / "app.py"

print("🚀 启动 Streamlit Web UI（单页面模式）...")
os.system(f"streamlit run {app_file}")

