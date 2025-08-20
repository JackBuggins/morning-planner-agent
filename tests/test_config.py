import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import sys
import json

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config

class TestConfig(unittest.TestCase):
    
    def setUp(self):
        # Sample config data for testing
        self.sample_config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "default_model": "llama3"
            },
            "weather": {
                "api_url": "https://api.openweathermap.org/data/2.5/weather",
                "units": "metric"
            }
        }
    
    @patch("dotenv.load_dotenv")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("json.load")
    def test_load_config_file(self, mock_json_load, mock_exists, mock_file_open, mock_load_dotenv):
        # Configure mocks
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_config
        
        # Create config instance
        config = Config("/fake/path/config.json")
        
        # Verify file was opened with the correct path
        mock_file_open.assert_called_with("/fake/path/config.json", "r")
        
        # Verify config was loaded
        self.assertEqual(config.get("api.port"), 8000)
        self.assertEqual(config.get("ollama.default_model"), "llama3")
        self.assertEqual(config.get("weather.units"), "metric")
    
    @patch("dotenv.main.find_dotenv", return_value="")
    @patch("dotenv.load_dotenv")
    @patch("os.path.exists")
    def test_missing_config_file(self, mock_exists, mock_load_dotenv, mock_find_dotenv):
        # Configure mock to simulate missing file
        mock_exists.return_value = False
        
        # Create config instance with non-existent file
        config = Config("/nonexistent/config.json")
        
        # Verify default values are used
        self.assertEqual(config.get("api.port", 9000), 9000)
        self.assertEqual(config.get("nonexistent.key", "default"), "default")
    
    @patch("dotenv.load_dotenv")
    @patch("os.getenv")
    @patch("os.path.exists")
    @patch("json.load")
    def test_env_override(self, mock_json_load, mock_exists, mock_getenv, mock_load_dotenv):
        # Configure mocks
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_config
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "OLLAMA_MODEL": "mistral",
            "API_PORT": "9000",
            "OPENWEATHER_API_KEY": "test_key"
        }.get(key, default)
        
        # Create config instance
        config = Config("/fake/path/config.json")
        
        # Verify environment variables override config file
        self.assertEqual(config.get("ollama.default_model"), "mistral")
        self.assertEqual(config.get("api.port"), 9000)
        self.assertEqual(config.get("weather.api_key"), "test_key")

if __name__ == "__main__":
    unittest.main()


