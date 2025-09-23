#!/usr/bin/env python3
"""
Example script showing how to use the Ollama Weather Agent directly from Python code.
This is useful for integrating the agent into other applications without using the API.
"""
import os
import sys
import json
import re
import requests
import urllib.parse
import warnings
import logging
from datetime import datetime, timedelta
import io

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress urllib3 warnings about OpenSSL
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Disable logging for specific modules
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)

# Redirect stdout and stderr to suppress print statements during imports
original_stdout = sys.stdout
original_stderr = sys.stderr
null_output = io.StringIO()
sys.stdout = null_output
sys.stderr = null_output

try:
    # Add the project root to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Import LangChain components
    try:
        # Try to import OllamaLLM from langchain_ollama (recommended)
        from langchain_ollama import OllamaLLM
        # Suppressed: print("Using OllamaLLM from langchain_ollama")
    except ImportError:
        # Fall back to deprecated import if langchain_ollama is not installed
        from langchain_community.llms import Ollama as OllamaLLM
        # Suppressed: print("Warning: Using deprecated Ollama import...")

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableSequence

    # Import project components
    from src.tools.weather_tool import WeatherTool
    from src.config import config
finally:
    # Restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

def preprocess_query(query):
    """
    Preprocess the user query to make it more robust to special characters and formatting.
    
    Args:
        query: The user's query string
        
    Returns:
        Preprocessed query string
    """
    # Remove any leading/trailing whitespace
    query = query.strip()
    
    # Replace multiple spaces with a single space
    query = re.sub(r'\s+', ' ', query)
    
    # Ensure the query ends with a question mark if it's a question
    if any(query.lower().startswith(q) for q in ["how", "what", "when", "where", "why", "is", "can", "will", "should"]) and not query.endswith("?"):
        query = query + "?"
    
    return query

def extract_location_from_text(text):
    """
    Manually extract location from text in case JSON parsing fails.
    Attempts to find JSON structure or extract location using regex patterns.
    """
    # Try to parse as JSON first
    try:
        # Find JSON-like structure in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if "location" in data:
                return data["location"]
    except Exception as e:
        print(f"JSON extraction failed: {str(e)}")
    
    # Fallback: Try to extract location using regex patterns
    try:
        # Look for patterns like 'location: "New York"' or 'location: New York'
        location_match = re.search(r'location["\s:]+([^"}\s]+|"[^"]+")(?:\s*})?', text, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip('"')
            return location
    except Exception:
        pass
    
    # Try to extract location from common weather query patterns
    try:
        # Look for patterns like "weather in [location]" or "weather for [location]"
        weather_match = re.search(r'weather\s+(?:in|for|at|of)\s+([A-Za-z\s,]+)(?:\s|$|\.|\?)', text, re.IGNORECASE)
        if weather_match:
            return weather_match.group(1).strip()
    except Exception:
        pass
    
    # If all else fails, try to extract any city name from the text
    # This is a very simple approach and might not be accurate
    return None

def llm_geocode_location(location, llm):
    """
    Use the LLM to get latitude and longitude coordinates for a location.
    
    Args:
        location: Location name (city, address, etc.)
        llm: The LLM instance to use
        
    Returns:
        Tuple of (latitude, longitude) if successful, None if failed
    """
    if not location:
        print("Error: No location provided for geocoding")
        return None
    
    try:
        # Create a prompt for geocoding
        geocoding_prompt = f"""
        You are a helpful assistant that provides accurate latitude and longitude coordinates for locations.
        
        Location: {location}
        
        Please provide the latitude and longitude coordinates for this location in the following JSON format:
        {{"latitude": LATITUDE_VALUE, "longitude": LONGITUDE_VALUE}}
        
        Replace LATITUDE_VALUE and LONGITUDE_VALUE with the actual numerical coordinates.
        Use decimal degrees with 6 decimal places of precision.
        Do not include any explanations or additional text, only return the JSON object.
        
        If you're not sure about the exact coordinates, provide your best estimate.
        """
        
        print(f"Asking LLM for coordinates of '{location}'...")
        
        # Get the response from the LLM
        response = llm.invoke(geocoding_prompt)
        print(f"LLM geocoding response: {response}")
        
        # Try to extract JSON from the response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Extract latitude and longitude
                lat = data.get("latitude")
                lon = data.get("longitude")
                
                if lat is not None and lon is not None:
                    # Convert to float if they're strings
                    if isinstance(lat, str):
                        lat = float(lat)
                    if isinstance(lon, str):
                        lon = float(lon)
                        
                    print(f"LLM geocoded '{location}' to coordinates: ({lat}, {lon})")
                    return (lat, lon)
            
            # If JSON pattern not found or missing keys, try to parse the whole response
            data = json.loads(response)
            lat = data.get("latitude")
            lon = data.get("longitude")
            
            if lat is not None and lon is not None:
                # Convert to float if they're strings
                if isinstance(lat, str):
                    lat = float(lat)
                if isinstance(lon, str):
                    lon = float(lon)
                    
                print(f"LLM geocoded '{location}' to coordinates: ({lat}, {lon})")
                return (lat, lon)
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM geocoding response: {str(e)}")
            
            # Try to extract coordinates using regex
            coords_match = re.search(r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)', response)
            if coords_match:
                try:
                    lat = float(coords_match.group(1))
                    lon = float(coords_match.group(2))
                    print(f"Extracted coordinates from text: ({lat}, {lon})")
                    return (lat, lon)
                except ValueError:
                    pass
        
        print(f"LLM could not provide valid coordinates for '{location}'")
        return None
        
    except Exception as e:
        print(f"Error during LLM geocoding: {str(e)}")
        return None

def normalize_location(location):
    """
    Normalize location names to improve geocoding success rate.
    
    Args:
        location: The location name to normalize
        
    Returns:
        A list of location variations to try, from most specific to most general
    """
    if not location:
        return []
    
    # Clean the location string
    location = location.strip()
    
    # Create a list of location variations to try
    variations = [location]  # Start with the original location
    
    # Add variations without special characters
    clean_location = re.sub(r'[^\w\s]', '', location)
    if clean_location != location:
        variations.append(clean_location)
    
    # If the location has multiple words, add the first word (usually the city name)
    parts = location.split()
    if len(parts) > 1:
        variations.append(parts[0])  # Add just the first word (usually the city)
        
        # If it looks like "City, Country" format, add both parts separately
        if ',' in location:
            city_country = location.split(',')
            if len(city_country) == 2:
                variations.append(city_country[0].strip())  # Just the city
                variations.append(city_country[1].strip())  # Just the country
    
    # Remove duplicates while preserving order
    unique_variations = []
    for var in variations:
        if var not in unique_variations:
            unique_variations.append(var)
    
    return unique_variations

def api_geocode_location(location, api_key):
    """
    Convert a location name to latitude and longitude coordinates using OpenWeatherMap's geocoding API.
    Tries multiple variations of the location name if the first attempt fails.
    
    Args:
        location: Location name (city, address, etc.)
        api_key: OpenWeatherMap API key
        
    Returns:
        Tuple of (latitude, longitude) if successful, None if failed
    """
    if not location or not api_key:
        print("Error: Location or API key not provided for geocoding")
        return None
    
    # Get location variations to try
    location_variations = normalize_location(location)
    print(f"Trying location variations with API: {location_variations}")
    
    # Try each location variation
    for variation in location_variations:
        try:
            print(f"Trying to geocode with API: '{variation}'")
            
            # OpenWeatherMap geocoding API endpoint
            geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
            
            # URL encode the location parameter to handle spaces and special characters
            encoded_location = urllib.parse.quote(variation)
            print(f"URL encoded location: '{encoded_location}'")
            
            # Parameters for the geocoding API
            params = {
                "q": variation,  # The requests library will handle URL encoding
                "limit": 1,      # Get only the top result
                "appid": api_key
            }
            
            # Make the API request
            response = requests.get(geocoding_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the response
            data = response.json()
            print(f"Geocoding API response for '{variation}': {data}")
            
            # Check if we got any results
            if data and len(data) > 0:
                lat = data[0].get("lat")
                lon = data[0].get("lon")
                if lat is not None and lon is not None:
                    print(f"Successfully geocoded '{variation}' to coordinates: ({lat}, {lon})")
                    return (lat, lon)
            
            print(f"No results found for '{variation}'")
            
        except Exception as e:
            print(f"Error geocoding '{variation}': {str(e)}")
    
    print(f"Could not geocode any variation of '{location}' with API")
    return None

def get_weather_by_coordinates(lat, lon, api_key, units="metric"):
    """
    Get current weather data for specific coordinates using OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeatherMap API key
        units: Units for temperature (metric, imperial)
        
    Returns:
        Weather data as a formatted string and the raw data
    """
    if not api_key:
        return "Error: OpenWeather API key not configured.", None
    
    try:
        # OpenWeatherMap current weather API endpoint
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        
        # Parameters for the weather API
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": units
        }
        
        # Make the API request
        response = requests.get(weather_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the response
        data = response.json()
        
        # Format the weather data
        return format_weather_data(data, units), data
        
    except Exception as e:
        return f"Error fetching weather data: {str(e)}", None

def get_forecast_by_coordinates(lat, lon, api_key, units="metric"):
    """
    Get weather forecast data for specific coordinates using OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeatherMap API key
        units: Units for temperature (metric, imperial)
        
    Returns:
        Forecast data for the rest of the day
    """
    if not api_key:
        return "Error: OpenWeather API key not configured.", None
    
    try:
        # OpenWeatherMap forecast API endpoint
        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        
        # Parameters for the forecast API
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": units
        }
        
        # Make the API request
        response = requests.get(forecast_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the response
        data = response.json()
        
        # Extract forecast for the rest of the day
        today_forecast = extract_today_forecast(data)
        
        return today_forecast, data
        
    except Exception as e:
        return f"Error fetching forecast data: {str(e)}", None

def extract_today_forecast(forecast_data):
    """
    Extract forecast data for the rest of the current day.
    
    Args:
        forecast_data: Raw forecast data from OpenWeatherMap API
        
    Returns:
        List of forecast periods for the rest of the day
    """
    if not forecast_data or "list" not in forecast_data:
        return []
    
    # Get the current date
    current_date = datetime.now().date()
    
    # Extract forecast entries for today
    today_forecast = []
    for entry in forecast_data["list"]:
        # Convert timestamp to datetime
        timestamp = entry["dt"]
        entry_date = datetime.fromtimestamp(timestamp).date()
        entry_time = datetime.fromtimestamp(timestamp).time()
        
        # Check if this entry is for today and in the future
        if entry_date == current_date and datetime.fromtimestamp(timestamp) > datetime.now():
            # Add a formatted time to the entry
            entry["formatted_time"] = datetime.fromtimestamp(timestamp).strftime("%H:%M")
            today_forecast.append(entry)
    
    return today_forecast

def format_weather_data(data, units="metric"):
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
        temp_unit = "°C" if units == "metric" else "°F"
        wind_unit = "m/s" if units == "metric" else "mph"
        
        return (
            f"Weather in {city}, {country}: {description}. "
            f"Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit}). "
            f"Humidity: {humidity}%. Wind speed: {wind_speed} {wind_unit}."
        )
    except KeyError as e:
        return f"Error parsing weather data: {str(e)}"

def get_clothing_recommendation(weather_data, forecast_data=None):
    """
    Generate clothing recommendations based on weather conditions for now and the rest of the day.
    
    Args:
        weather_data: Current weather data from OpenWeatherMap API
        forecast_data: Forecast data for the rest of the day (optional)
        
    Returns:
        String with clothing recommendations
    """
    if not weather_data:
        return "I can't provide clothing recommendations without weather data."
    
    try:
        # Get current time
        current_time = datetime.now()
        
        # Initialize recommendations
        current_recommendations = []
        forecast_recommendations = {}
        
        # Generate recommendations for current weather
        # Extract relevant weather information
        temp = weather_data["main"]["temp"]
        description = weather_data["weather"][0]["description"].lower()
        wind_speed = weather_data["wind"]["speed"]
        
        # Temperature-based recommendations
        if temp < 0:
            current_recommendations.append("heavy winter coat")
            current_recommendations.append("hat, scarf, and gloves")
            current_recommendations.append("thermal layers")
            current_recommendations.append("insulated boots")
        elif temp < 10:
            current_recommendations.append("winter coat or heavy jacket")
            current_recommendations.append("hat and gloves")
            current_recommendations.append("warm layers")
        elif temp < 15:
            current_recommendations.append("light jacket or heavy sweater")
            current_recommendations.append("long sleeves")
        elif temp < 20:
            current_recommendations.append("light sweater or long-sleeved shirt")
        elif temp < 25:
            current_recommendations.append("t-shirt or light top")
            current_recommendations.append("light pants or jeans")
        else:
            current_recommendations.append("light, breathable clothing")
            current_recommendations.append("shorts or light pants")
            current_recommendations.append("sun protection")
        
        # Weather condition-based recommendations
        if "rain" in description or "drizzle" in description or "shower" in description:
            current_recommendations.append("raincoat or umbrella")
            current_recommendations.append("waterproof shoes")
        elif "snow" in description or "sleet" in description:
            current_recommendations.append("waterproof boots")
            current_recommendations.append("warm, waterproof jacket")
        elif "thunderstorm" in description:
            current_recommendations.append("stay indoors if possible")
            current_recommendations.append("raincoat and umbrella if you must go out")
        elif "clear" in description and temp > 20:
            current_recommendations.append("sunglasses")
            current_recommendations.append("sunscreen")
            current_recommendations.append("hat for sun protection")
        
        # Wind-based recommendations
        if wind_speed > 10:
            current_recommendations.append("windbreaker or wind-resistant jacket")
        
        # Generate recommendations for forecast periods if available
        if forecast_data:
            # Group forecast periods into meaningful time blocks
            time_blocks = {
                "Morning": [],
                "Afternoon": [],
                "Evening": []
            }
            
            for entry in forecast_data:
                entry_time = datetime.fromtimestamp(entry["dt"]).time()
                
                if entry_time.hour < 12:
                    time_blocks["Morning"].append(entry)
                elif entry_time.hour < 18:
                    time_blocks["Afternoon"].append(entry)
                else:
                    time_blocks["Evening"].append(entry)
            
            # Generate recommendations for each time block
            for block_name, entries in time_blocks.items():
                if not entries:
                    continue
                    
                # Use the middle entry as representative for the time block
                middle_idx = len(entries) // 2
                representative_entry = entries[middle_idx]
                
                block_recommendations = []
                
                # Extract weather information
                temp = representative_entry["main"]["temp"]
                description = representative_entry["weather"][0]["description"].lower()
                wind_speed = representative_entry["wind"]["speed"]
                
                # Temperature-based recommendations
                if temp < 0:
                    block_recommendations.append("heavy winter coat")
                    block_recommendations.append("hat, scarf, and gloves")
                    block_recommendations.append("thermal layers")
                elif temp < 10:
                    block_recommendations.append("winter coat or heavy jacket")
                    block_recommendations.append("hat and gloves")
                elif temp < 15:
                    block_recommendations.append("light jacket or heavy sweater")
                elif temp < 20:
                    block_recommendations.append("light sweater or long-sleeved shirt")
                elif temp < 25:
                    block_recommendations.append("t-shirt or light top")
                else:
                    block_recommendations.append("light, breathable clothing")
                    block_recommendations.append("sun protection")
                
                # Weather condition-based recommendations
                if "rain" in description or "drizzle" in description:
                    block_recommendations.append("raincoat or umbrella")
                elif "snow" in description:
                    block_recommendations.append("waterproof boots")
                elif "clear" in description and temp > 20:
                    block_recommendations.append("sunglasses")
                
                # Wind-based recommendations
                if wind_speed > 10:
                    block_recommendations.append("windbreaker")
                
                forecast_recommendations[block_name] = block_recommendations
        
        # Format the recommendations
        result = "Clothing recommendations:\n\n"
        
        # Current recommendations
        result += "For now: " + ", ".join(current_recommendations) + ".\n\n"
        
        # Forecast recommendations
        if forecast_recommendations:
            result += "For the rest of the day:\n"
            for block_name, recommendations in forecast_recommendations.items():
                if recommendations:
                    result += f"- {block_name}: " + ", ".join(recommendations) + ".\n"
        
        return result
            
    except (KeyError, TypeError) as e:
        return f"Error generating clothing recommendations: {str(e)}"

def main():
    # Redirect stdout and stderr to suppress initialization messages
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    null_output = io.StringIO()
    sys.stdout = null_output
    sys.stderr = null_output
    
    try:
        # Initialize Ollama
        ollama_url = config.get("ollama.base_url", "http://localhost:11434")
        ollama_model = config.get("ollama.default_model", "llama3")
        # Suppressed: print(f"Initializing Ollama with model: {ollama_model}")
        llm = OllamaLLM(base_url=ollama_url, model=ollama_model)
        
        # Initialize weather tool and get API key
        weather_tool = WeatherTool()
        api_key = weather_tool.api_key
        units = "metric"  # Default to metric units if not specified
        if hasattr(weather_tool, "units") and isinstance(weather_tool.units, str):
            units = weather_tool.units
    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    # Create a prompt template for general queries
    prompt_template = """
    You are a helpful AI assistant with access to weather information.
    If the user asks about the weather, use the weather tool to provide accurate information.
    Otherwise, respond helpfully to their query.

    User query: {query}

    Think step by step:
    1. Determine if this is a weather-related query
    2. If it is weather-related, extract the location and use the weather tool
    3. If not weather-related, respond based on your knowledge

    Your response:
    """

    # Create a prompt template for location extraction with improved JSON formatting
    # Note: Double curly braces {{ }} are used to escape actual curly braces in the template
    weather_location_resolution_prompt = """
    Your task is to extract ONLY the location from the user query.
    
    User query: {query}
    
    INSTRUCTIONS:
    1. Identify the location mentioned in the query
    2. Return ONLY a valid JSON object with this exact format: {{"location": "EXTRACTED_LOCATION"}}
    3. Replace EXTRACTED_LOCATION with the actual location from the query
    4. If no location is found, use: {{"location": "Unknown"}}
    5. Do not include any explanations, notes, or additional text
    6. Ensure the JSON is properly formatted with double quotes
    7. Ignore punctuation like commas, periods, and question marks when extracting the location
    8. Keep multi-word locations together (e.g., "New York", "London Waterloo")
    9. For specific locations like stations or neighborhoods, include the city name (e.g., "London Waterloo" or "Waterloo, London")
    
    EXAMPLE OUTPUTS:
    {{"location": "London"}}
    {{"location": "New York City"}}
    {{"location": "London Waterloo"}}
    {{"location": "London, UK"}}
    {{"location": "Unknown"}}
    
    EXAMPLE INPUTS AND EXPECTED OUTPUTS:
    Input: "What's the weather like in Paris?"
    Output: {{"location": "Paris"}}
    
    Input: "Hey, how is the weather in London Waterloo today?"
    Output: {{"location": "London Waterloo"}}
    
    Input: "Will it rain tomorrow in San Francisco, California?"
    Output: {{"location": "San Francisco, California"}}
    
    YOUR RESPONSE (ONLY JSON):
    """

    # Create prompt templates
    location_resolution_prompt = PromptTemplate(
        template=weather_location_resolution_prompt, 
        input_variables=["query"]
    )
    
    general_prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["query"]
    )
    
    # Create output parsers
    location_parser = JsonOutputParser()
    str_parser = StrOutputParser()
    
    # Create the location resolution chain
    location_resolution_chain = (
        {"query": RunnablePassthrough()} 
        | location_resolution_prompt
        | llm
        | str_parser  # Use StrOutputParser first to get the raw text
    )
    
    # Create the general query chain
    general_query_chain = (
        {"query": RunnablePassthrough()}
        | general_prompt
        | llm
        | str_parser
    )
    
    # Set debug mode (set to False to hide debug prints)
    DEBUG = False
    
    # Define a debug print function that only prints if DEBUG is True
    def debug_print(*args, **kwargs):
        if DEBUG:
            print(*args, **kwargs)
    
    print("Ollama Weather Agent Example")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        # Get user input
        user_query = input("\nYour query: ")
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        try:
            # Initialize response variable
            response = "I don't understand your query. Please try again."
            
            # Preprocess the user query
            processed_query = preprocess_query(user_query)
            
            # Check if it's a weather query
            if "weather" in processed_query.lower():
                debug_print("Processing weather query...")
                
                # Extract location using the location resolution chain
                try:
                    # Get the raw text response from the LLM
                    llm_response = location_resolution_chain.invoke(processed_query)
                    debug_print(f"Raw LLM response: {llm_response}")
                    
                    # Try to parse the JSON response
                    location = None
                    try:
                        # Try to clean up the response if it contains extra text
                        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            location_data = json.loads(json_str)
                            location = location_data.get("location", "Unknown")
                        else:
                            # If no JSON found, try to parse the whole response
                            location_data = json.loads(llm_response)
                            location = location_data.get("location", "Unknown")
                    except json.JSONDecodeError:
                        # Fallback to manual extraction if JSON parsing fails
                        debug_print("JSON parsing failed, trying manual extraction...")
                        location = extract_location_from_text(llm_response)
                        
                        # If manual extraction fails, try to extract from the original query
                        if not location:
                            debug_print("Manual extraction from LLM response failed, trying to extract from query...")
                            location = extract_location_from_text(processed_query)
                            
                        if not location:
                            location = "Unknown"
                    
                    debug_print(f"Extracted location: '{location}'")
                    
                    if location and location.lower() != "unknown":
                        # First try to geocode using the LLM
                        debug_print(f"Using LLM to geocode location: '{location}'")
                        coordinates = llm_geocode_location(location, llm)
                        
                        # If LLM geocoding fails, fall back to API
                        if not coordinates:
                            debug_print(f"LLM geocoding failed, falling back to API for: '{location}'")
                            coordinates = api_geocode_location(location, api_key)
                        
                        if coordinates:
                            lat, lon = coordinates
                            debug_print(f"Fetching weather for coordinates: ({lat}, {lon})")
                            
                            # Get current weather data
                            weather_text, raw_weather_data = get_weather_by_coordinates(lat, lon, api_key, units)
                            debug_print(f"Current weather data: {weather_text}")
                            
                            # Get forecast data for the rest of the day
                            debug_print(f"Fetching forecast for the rest of the day...")
                            forecast_data, raw_forecast_data = get_forecast_by_coordinates(lat, lon, api_key, units)
                            
                            # Check if we got an error message back for current weather
                            if isinstance(weather_text, str) and "Error" in weather_text:
                                response = f"I couldn't get the weather for {location}. {weather_text}"
                            else:
                                # Get clothing recommendations for current weather and forecast
                                clothing_recommendations = get_clothing_recommendation(raw_weather_data, forecast_data)
                                
                                # Use a simple string for the weather response with clothing recommendations
                                response = f"Based on your query about the weather in {location}, here's what I found:\n\n{weather_text}\n\n{clothing_recommendations}"
                        else:
                            # If we couldn't geocode the specific location, provide a helpful message
                            city_match = re.search(r'([A-Za-z]+)', location)
                            if city_match:
                                city = city_match.group(1)
                                response = f"I couldn't find the specific location '{location}'. Try asking about the weather in '{city}' instead."
                            else:
                                response = f"I couldn't find the location '{location}'. Please try a different location."
                    else:
                        response = "I need a location to check the weather. Please specify a city or place."
                except Exception as e:
                    debug_print(f"Error processing location: {str(e)}")
                    
                    # Try a direct approach if the chain fails
                    try:
                        debug_print("Trying direct location extraction from query...")
                        location = extract_location_from_text(processed_query)
                        
                        if location and location.lower() != "unknown":
                            # First try to geocode using the LLM
                            debug_print(f"Using LLM to geocode location: '{location}'")
                            coordinates = llm_geocode_location(location, llm)
                            
                            # If LLM geocoding fails, fall back to API
                            if not coordinates:
                                debug_print(f"LLM geocoding failed, falling back to API for: '{location}'")
                                coordinates = api_geocode_location(location, api_key)
                            
                            if coordinates:
                                lat, lon = coordinates
                                debug_print(f"Fetching weather for coordinates: ({lat}, {lon})")
                                
                                # Get current weather data
                                weather_text, raw_weather_data = get_weather_by_coordinates(lat, lon, api_key, units)
                                
                                # Check if we got an error message back
                                if isinstance(weather_text, str) and "Error" in weather_text:
                                    response = f"I couldn't get the weather for {location}. {weather_text}"
                                else:
                                    response = f"Based on your query about the weather in {location}, here's what I found:\n\n{weather_text}"
                            else:
                                response = f"I couldn't find the location '{location}'. Please try a different location."
                    except Exception as e:
                        print(f"Error during direct location extraction: {str(e)}")
                        response = "I had trouble processing your weather query. Please try again with a clearer location."
            else:
                # For non-weather queries, use the general query chain
                print("Processing general query...")
                response = general_query_chain.invoke(processed_query)
                
            # Print the response
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()

