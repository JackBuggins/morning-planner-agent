#!/usr/bin/env python3
"""
Example script showing how to use the Ollama Weather Agent directly from Python code.
This is useful for integrating the agent into other applications without using the API.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from src.tools.weather_tool import WeatherTool
from src.config import config

def main():
    # Initialize Ollama
    ollama_url = config.get("ollama.base_url", "http://localhost:11434")
    ollama_model = config.get("ollama.default_model", "llama3")
    print(f"Initializing Ollama with model: {ollama_model}")
    llm = Ollama(base_url=ollama_url, model=ollama_model)
    
    # Initialize weather tool
    weather_tool = WeatherTool()
    
    # Create a prompt template
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
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
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
            # Check if it's a weather query
            if "weather" in user_query.lower():
                # Extract location (simplified - in production you'd use NER or similar)
                location = user_query.lower().replace("weather", "").replace("what's the", "").replace("how is the", "").replace("in", "").strip()
                if location:
                    print(f"Fetching weather for: {location}")
                    weather_data = weather_tool.get_weather(location)
                    print(f"Weather data: {weather_data}")
                    
                    # Format weather data for the LLM
                    weather_context = f"Weather in {location}: {weather_data}"
                    full_prompt = f"{prompt_template}\n\nWeather data: {weather_context}"
                    chain = LLMChain(llm=llm, prompt=PromptTemplate(template=full_prompt, input_variables=[]))
                    response = chain.run({})
                else:
                    response = "I need a location to check the weather. Please specify a city or place."
            else:
                # For non-weather queries, use the standard prompt
                response = chain.run(query=user_query)
                
            print("\nResponse:", response)
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


