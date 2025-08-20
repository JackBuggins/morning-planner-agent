#!/usr/bin/env python3
"""
Script to check if Ollama is running and available.
This is useful to run before starting the agent to ensure Ollama is ready.
"""
import sys
import requests
import time
from src.config import config

def check_ollama():
    """Check if Ollama is running and available."""
    ollama_url = config.get("ollama.base_url", "http://localhost:11434")
    ollama_model = config.get("ollama.default_model", "llama3")
    
    print(f"Checking Ollama at {ollama_url}...")
    
    # Try to connect to Ollama
    try:
        # Check if Ollama API is responding
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code != 200:
            print(f"Error: Ollama API returned status code {response.status_code}")
            return False
        
        # Check if the configured model is available
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]
        
        if ollama_model not in model_names:
            print(f"Warning: Model '{ollama_model}' not found in available models.")
            print(f"Available models: {', '.join(model_names)}")
            print(f"You may need to run: ollama pull {ollama_model}")
            return False
        
        print(f"Ollama is running and model '{ollama_model}' is available.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama.")
        print("Make sure Ollama is running. You can start it with:")
        print("  ollama serve")
        return False
    except Exception as e:
        print(f"Error checking Ollama: {str(e)}")
        return False

def main():
    """Main function."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(1, max_retries + 1):
        if check_ollama():
            sys.exit(0)
        
        if attempt < max_retries:
            print(f"Retrying in {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
            time.sleep(retry_delay)
    
    print("Failed to connect to Ollama after multiple attempts.")
    sys.exit(1)

if __name__ == "__main__":
    main()
