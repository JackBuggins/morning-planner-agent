#!/usr/bin/env python3
"""
Entry point script to run the Ollama Weather Agent API server.
"""
import uvicorn
import os
from src.config import config

if __name__ == "__main__":
    # Get configuration values
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    debug = config.get("api.debug", True)
    
    print(f"Starting Ollama Weather Agent API on {host}:{port}...")
    print(f"Using Ollama model: {config.get('ollama.default_model', 'llama3')}")
    print(f"Debug mode: {'enabled' if debug else 'disabled'}")
    print("Press Ctrl+C to stop the server")
    
    # Run the FastAPI application
    uvicorn.run("src.app:app", host=host, port=port, reload=debug)


