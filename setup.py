#!/usr/bin/env python3
"""
Setup script for the Ollama Weather Agent.
This script helps users set up their environment and install dependencies.
"""
import os
import sys
import subprocess
import platform
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    print(f"Python version {current_version[0]}.{current_version[1]}.{current_version[2]} is compatible.")
    return True

def check_pip():
    """Check if pip is installed."""
    pip_path = shutil.which("pip") or shutil.which("pip3")
    
    if not pip_path:
        print("Error: pip is not installed or not in PATH.")
        print("Please install pip and try again.")
        return False
    
    print(f"Found pip at: {pip_path}")
    return True

def install_dependencies():
    """Install Python dependencies."""
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        return False
    
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {str(e)}")
        return False

def check_ollama():
    """Check if Ollama is installed."""
    ollama_path = shutil.which("ollama")
    
    if not ollama_path:
        print("Warning: Ollama is not installed or not in PATH.")
        print("You'll need to install Ollama to use this agent.")
        print("Visit https://ollama.ai/ for installation instructions.")
        return False
    
    print(f"Found Ollama at: {ollama_path}")
    return True

def setup_env_file():
    """Set up the .env file if it doesn't exist."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env_example = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
    
    if os.path.exists(env_path):
        print(".env file already exists.")
        return True
    
    # Create a basic .env file
    try:
        with open(env_path, "w") as f:
            f.write("# OpenWeather API key - get one from https://openweathermap.org/api\n")
            f.write("OPENWEATHER_API_KEY=your_api_key_here\n\n")
            f.write("# Ollama configuration\n")
            f.write("OLLAMA_BASE_URL=http://localhost:11434\n")
            f.write("OLLAMA_MODEL=llama3\n")
        
        print(f".env file created at {env_path}")
        print("Please edit the file to add your OpenWeather API key.")
        return True
    except Exception as e:
        print(f"Error creating .env file: {str(e)}")
        return False

def main():
    """Main function."""
    print("Setting up Ollama Weather Agent...")
    print("-" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check Ollama
    check_ollama()
    
    # Set up .env file
    setup_env_file()
    
    print("-" * 50)
    print("Setup completed successfully!")
    print("Next steps:")
    print("1. Edit the .env file to add your OpenWeather API key")
    print("2. Make sure Ollama is running (ollama serve)")
    print("3. Run the agent with: python run.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


