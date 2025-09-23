import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import re
from fastapi.testclient import TestClient

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after path setup
from src.app import app

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root endpoint returns the expected message."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Ollama Weather Agent API is running", response.json()["message"])
    
    @patch('src.app.LLMChain')
    @patch('src.app.get_weather_by_coordinates')
    @patch('src.app.llm_geocode_location')
    def test_chat_endpoint_weather_query(self, mock_geocode, mock_get_weather, mock_llm_chain):
        """Test the chat endpoint with a weather query."""
        # Configure mocks
        mock_geocode.return_value = (51.5074, -0.1278)  # London coordinates
        
        # Mock weather data
        weather_data = {
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
        
        mock_get_weather.return_value = (
            "Weather in London, GB: light rain. Temperature: 15.5°C (feels like 14.8°C). Humidity: 76%. Wind speed: 4.1 m/s.",
            weather_data
        )
        
        # Mock LLMChain for location resolution
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = '{"location": "London"}'
        mock_llm_chain.return_value = mock_chain_instance
        
        # Make request
        response = self.client.post(
            "/chat",
            json={"text": "What's the weather in London?"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())
        self.assertIn("London", response.json()["response"])
        self.assertIn("light rain", response.json()["response"])
        
        # Verify location resolution was called
        mock_llm_chain.assert_called()
        
        # Verify geocoding was called with London
        mock_geocode.assert_called_once()
        self.assertEqual(mock_geocode.call_args[0][0], "London")
        
        # Verify weather data was fetched with the correct coordinates
        mock_get_weather.assert_called_once()
        self.assertEqual(mock_get_weather.call_args[0][0], 51.5074)
        self.assertEqual(mock_get_weather.call_args[0][1], -0.1278)
    
    @patch('src.app.LLMChain')
    def test_chat_endpoint_non_weather_query(self, mock_llm_chain):
        """Test the chat endpoint with a non-weather query."""
        # Mock LLMChain
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = "Python is a popular programming language known for its readability and versatility."
        mock_llm_chain.return_value = mock_chain_instance
        
        # Make request
        response = self.client.post(
            "/chat",
            json={"text": "Tell me about Python programming language"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())
        self.assertIn("Python", response.json()["response"])
        
        # Verify correct prompt was used
        mock_llm_chain.assert_called_once()
        self.assertEqual(mock_chain_instance.run.call_args[1]["query"], "Tell me about Python programming language")
    
    def test_chat_endpoint_invalid_request(self):
        """Test the chat endpoint with invalid request data."""
        # Missing required field
        response = self.client.post(
            "/chat",
            json={"invalid": "data"}
        )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
    
    @patch('src.app.LLMChain')
    @patch('src.app.api_geocode_location')
    @patch('src.app.llm_geocode_location')
    def test_chat_endpoint_location_not_found(self, mock_llm_geocode, mock_api_geocode, mock_llm_chain):
        """Test the chat endpoint when location cannot be geocoded."""
        # Configure mocks
        mock_llm_geocode.return_value = None
        mock_api_geocode.return_value = None
        
        # Mock LLMChain for location resolution
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = '{"location": "NonExistentPlace"}'
        mock_llm_chain.return_value = mock_chain_instance
        
        # Make request
        response = self.client.post(
            "/chat",
            json={"text": "What's the weather in NonExistentPlace?"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())
        self.assertIn("couldn't find", response.json()["response"].lower())
        self.assertIn("NonExistentPlace", response.json()["response"])
        
        # Verify geocoding was attempted
        mock_llm_geocode.assert_called_once()
        mock_api_geocode.assert_called_once()
    
    @patch('src.app.get_forecast_by_coordinates')
    @patch('src.app.get_weather_by_coordinates')
    @patch('src.app.llm_geocode_location')
    @patch('src.app.LLMChain')
    def test_chat_endpoint_with_clothing_recommendations(self, mock_llm_chain, mock_geocode,
                                                       mock_get_weather, mock_get_forecast):
        """Test the chat endpoint with weather query including clothing recommendations."""
        # Configure mocks
        mock_geocode.return_value = (51.5074, -0.1278)  # London coordinates
        
        # Mock weather data
        weather_data = {
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
        
        # Mock forecast data
        forecast_data = [
            {
                "dt": 1632571200,
                "main": {
                    "temp": 16.2,
                    "feels_like": 15.8,
                    "humidity": 72
                },
                "weather": [{"description": "scattered clouds"}],
                "wind": {"speed": 3.5},
                "formatted_time": "15:00"
            }
        ]
        
        mock_get_weather.return_value = (
            "Weather in London, GB: light rain. Temperature: 15.5°C (feels like 14.8°C). Humidity: 76%. Wind speed: 4.1 m/s.",
            weather_data
        )
        
        mock_get_forecast.return_value = (forecast_data, {"list": forecast_data})
        
        # Mock LLMChain for location resolution
        mock_chain_instance = MagicMock()
        # First call is for location resolution, second call might be for general query
        mock_chain_instance.run.side_effect = ['{"location": "London"}', "Weather information for London"]
        mock_llm_chain.return_value = mock_chain_instance
        
        # Make request
        response = self.client.post(
            "/chat",
            json={"text": "What's the weather in London today?"}  # Changed to explicit weather query
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())
        self.assertIn("London", response.json()["response"])
        self.assertIn("light rain", response.json()["response"])
        self.assertIn("Clothing recommendations", response.json()["response"])
        
        # Verify forecast was fetched
        mock_get_forecast.assert_called_once()

if __name__ == "__main__":
    unittest.main()

