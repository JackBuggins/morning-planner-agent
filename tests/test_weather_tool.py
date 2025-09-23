import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import requests

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.weather_tool import WeatherTool

class TestWeatherTool(unittest.TestCase):
    
    def setUp(self):
        # Create a WeatherTool instance for testing
        with patch('src.config.config.get') as mock_config_get:
            # Mock configuration values
            mock_config_get.side_effect = lambda key, default=None: {
                "weather.api_key": "test_api_key",
                "weather.api_url": "https://api.openweathermap.org/data/2.5/weather",
                "weather.units": "metric"
            }.get(key, default)
            
            self.weather_tool = WeatherTool()
        
        # Sample weather data for mocking responses
        self.sample_weather_data = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.8,
                "humidity": 76
            },
            "weather": [{"description": "light rain"}],
            "wind": {"speed": 4.1}
        }
    
    @patch('src.tools.weather_tool.requests.get')
    def test_get_weather_success(self, mock_get):
        # Configure the mock to return a successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = self.sample_weather_data
        mock_get.return_value = mock_response
        
        # Call the method under test
        result = self.weather_tool.get_weather("London")
        
        # Verify the result contains expected weather information
        self.assertIn("London, GB", result)
        self.assertIn("light rain", result)
        self.assertIn("15.5°C", result)
        self.assertIn("76%", result)
        self.assertIn("4.1 m/s", result)
        
        # Verify the API was called with the correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs['params']['q'], "London")
        self.assertEqual(kwargs['params']['units'], "metric")
        self.assertEqual(kwargs['params']['appid'], "test_api_key")
    
    @patch('src.tools.weather_tool.requests.get')
    def test_get_weather_api_error(self, mock_get):
        # Configure the mock to raise a RequestException
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        # Call the method under test
        result = self.weather_tool.get_weather("InvalidCity")
        
        # Verify the result contains error information
        self.assertIn("Error fetching weather data", result)
    
    @patch('src.config.config.get')
    def test_missing_api_key(self, mock_config_get):
        # Mock configuration to return no API key
        mock_config_get.return_value = None
        
        # Create a new instance with no API key
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": ""}):
            tool_without_key = WeatherTool()
            result = tool_without_key.get_weather("London")
            
            # Verify the result contains error information
            self.assertIn("Error: OpenWeather API key not configured", result)
    
    def test_format_weather_data(self):
        """Test the _format_weather_data method directly."""
        # Call the method under test
        result = self.weather_tool._format_weather_data(self.sample_weather_data)
        
        # Verify the formatted string contains all expected elements
        self.assertIn("London, GB", result)
        self.assertIn("light rain", result)
        self.assertIn("15.5°C", result)
        self.assertIn("feels like 14.8°C", result)
        self.assertIn("76%", result)
        self.assertIn("4.1 m/s", result)
    
    def test_format_weather_data_missing_keys(self):
        """Test the _format_weather_data method with missing data."""
        # Create incomplete weather data
        incomplete_data = {
            "name": "London",
            "sys": {},  # Missing country
            "main": {
                "temp": 15.5
                # Missing feels_like and humidity
            },
            "weather": [{"description": "light rain"}]
            # Missing wind
        }
        
        # Call the method under test
        result = self.weather_tool._format_weather_data(incomplete_data)
        
        # Verify the result contains error information
        self.assertIn("Error parsing weather data", result)

if __name__ == '__main__':
    unittest.main()


