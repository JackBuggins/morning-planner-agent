import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """
    Configuration manager that loads settings from config files and environment variables.
    Environment variables take precedence over config file values.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Default config path
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "default_config.json")
        
        # Load config from file
        self.config = self._load_config_file(config_path)
        
        # Override with environment variables
        self._override_from_env()
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Config file not found at {config_path}")
                return {}
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return {}
    
    def _override_from_env(self):
        """Override configuration with environment variables."""
        # API settings
        if os.getenv("API_PORT"):
            self.config.setdefault("api", {})["port"] = int(os.getenv("API_PORT"))
        
        # Ollama settings
        if os.getenv("OLLAMA_BASE_URL"):
            self.config.setdefault("ollama", {})["base_url"] = os.getenv("OLLAMA_BASE_URL")
        if os.getenv("OLLAMA_MODEL"):
            self.config.setdefault("ollama", {})["default_model"] = os.getenv("OLLAMA_MODEL")
        
        # Weather settings
        if os.getenv("OPENWEATHER_API_KEY"):
            self.config.setdefault("weather", {})["api_key"] = os.getenv("OPENWEATHER_API_KEY")
    
    def get(self, key: str, default=None):
        """
        Get a configuration value by key path.
        
        Args:
            key: Dot-separated path to the config value (e.g., "api.port")
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value

# Create a singleton instance
config = Config()


