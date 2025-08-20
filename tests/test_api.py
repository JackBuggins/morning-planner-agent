import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
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
    @patch('src.app.weather_tool.get_weather')
    def test_chat_endpoint_weather_query(self, mock_get_weather, mock_llm_chain):
        """Test the chat endpoint with a weather query."""
        # Configure mocks
        mock_get_weather.return_value = "Sunny, 25°C, humidity 60%"
        
        # Mock LLMChain
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = "The weather in London is sunny with a temperature of 25°C and humidity of 60%."
        mock_llm_chain.return_value = mock_chain_instance
        
        # Make request
        response = self.client.post(
            "/chat",
            json={"text": "What's the weather in London?"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())
        
        # Verify weather tool was called
        mock_get_weather.assert_called_once()
        self.assertIn("london", mock_get_weather.call_args[0][0])
    
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

if __name__ == "__main__":
    unittest.main()


