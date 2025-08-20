import os
import requests
from typing import Dict, Any, Optional
from src.config import config

class WeatherTool:
    """Tool for fetching weather data from OpenWeatherMap API."""
    
    def __init__(self):
        """Initialize the weather tool with API key from configuration."""
        self.api_key = config.get("weather.api_key") or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            print("Warning: OpenWeather API key not found in configuration or environment variables.")
        
        self.base_url = config.get("weather.api_url", "https://api.openweathermap.org/data/2.5/weather")
        self.units = config.get("weather.units", "metric")
    
    def get_weather(self, location: str) -> str:
        """
        Get current weather for a location.
        
        Args:
            location: City name or location
            
        Returns:
            String with formatted weather information
        """
        if not self.api_key:
            return "Error: OpenWeather API key not configured."
        
        try:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": self.units
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            return self._format_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
    
    def _format_weather_data(self, data: Dict[str, Any]) -> str:
        """Format the weather data into a readable string."""
        try:
            city = data["name"]
            country = data["sys"]["country"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            description = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            # Use °C or °F based on units
            temp_unit = "°C" if self.units == "metric" else "°F"
            wind_unit = "m/s" if self.units == "metric" else "mph"
            
            return (
                f"Weather in {city}, {country}: {description}. "
                f"Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit}). "
                f"Humidity: {humidity}%. Wind speed: {wind_speed} {wind_unit}."
            )
        except KeyError as e:
            return f"Error parsing weather data: {str(e)}"

if __name__ == "__main__":
    # Simple test if run directly
    tool = WeatherTool()
    print(tool.get_weather("London"))


