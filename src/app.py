from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from src.tools.weather_tool import WeatherTool
from src.config import config

app = FastAPI(title="Ollama Weather Agent")

# Initialize Ollama
ollama_url = config.get("ollama.base_url", "http://localhost:11434")
ollama_model = config.get("ollama.default_model", "llama3")
llm = Ollama(base_url=ollama_url, model=ollama_model)

# Initialize tools
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

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        user_query = query.text
        
        # Check if it's a weather query
        if "weather" in user_query.lower():
            # Extract location (simplified - in production you'd use NER or similar)
            location = user_query.lower().replace("weather", "").replace("what's the", "").replace("how is the", "").replace("in", "").strip()
            if location:
                weather_data = weather_tool.get_weather(location)
                # Format weather data for the LLM
                weather_context = f"Weather in {location}: {weather_data}"
                full_prompt = f"{prompt_template}\n\nWeather data: {weather_context}"
                chain = LLMChain(llm=llm, prompt=PromptTemplate(template=full_prompt, input_variables=[]))
                response = chain.run({})
            else:
                response = "I need a location to check the weather. Please specify a city or place."
        else:
            # For non-weather queries, use the standard prompt
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(query=user_query)
            
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Ollama Weather Agent API is running. Send POST requests to /chat endpoint."}

if __name__ == "__main__":
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    debug = config.get("api.debug", True)
    uvicorn.run("src.app:app", host=host, port=port, reload=debug)


