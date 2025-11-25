#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run the ICONNET Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        print("✅ All core dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def install_dependencies():
    """Install dependencies from requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully")
    else:
        print("⚠️ requirements.txt not found")

def run_dashboard():
    """Run the Streamlit dashboard"""
    main_file = Path(__file__).parent / "main.py"
    
    if not main_file.exists():
        print("❌ main.py not found")
        return
    
    print("🚀 Starting ICONNET Dashboard...")
    print("📊 Dashboard will open in your browser...")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(main_file),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--browser.gatherUsageStats=false"
    ])

def main():
    """Main function"""
    print("🌟 ICONNET Predictive Analytics Dashboard")
    print("=" * 50)
    
    # Check if dependencies are available
    if not check_dependencies():
        print("🔧 Installing missing dependencies...")
        install_dependencies()
        
        # Check again
        if not check_dependencies():
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            return
    
    # Run the dashboard
    run_dashboard()

if __name__ == "__main__":
    main()