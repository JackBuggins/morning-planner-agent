# Morning Planner Agent

This project creates an AI agent that uses a local Ollama model with added functionality to fetch weather data. The agent can respond to general queries and has special handling for weather-related questions.

## Features

- Integration with local Ollama models
- Weather data fetching using OpenWeatherMap API
- FastAPI web server for easy interaction
- Extensible architecture for adding more tools
- Configuration management with environment variables and config files
- Comprehensive test suite

## Prerequisites

- Python 3.8+
- Ollama installed and running locally (https://ollama.ai/)
- OpenWeatherMap API key (https://openweathermap.org/api)

## Python Environment Setup

This project uses Python 3.8.13 and includes a `.python-version` file for use with pyenv.

For pyenv installation and setup instructions, please refer to the official documentation:
https://github.com/pyenv/pyenv#installation

After installing pyenv, you can install Python 3.8.13 with:
```
pyenv install 3.8.13
```

Then navigate to the project directory, and pyenv will automatically use Python 3.8.13.

## Installation

1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd ollama-weather-agent
   ```

2. Run the setup script to install dependencies and create configuration files:
   ```
   ./setup.py
   ```

3. Edit the `.env` file to add your OpenWeatherMap API key

## Usage

### Running the API Server

Start the API server:
```
./run.py
```

The API will be available at `http://localhost:8000`

Send requests to the `/chat` endpoint:
```
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"text": "What's the weather in London?"}'
```

### Using the Agent Directly

You can also use the agent directly from Python code:
```
./examples/direct_usage.py
```

This will start an interactive session where you can chat with the agent.

## Project Structure

```
ollama-weather-agent/
├── config/                 # Configuration files
│   └── default_config.json # Default configuration
├── examples/               # Example usage scripts
│   └── direct_usage.py     # Example of using the agent directly
├── src/                    # Source code
│   ├── tools/              # Tool implementations
│   │   ├── __init__.py
│   │   └── weather_tool.py # Weather tool implementation
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   └── config.py           # Configuration management
├── tests/                  # Test files
│   ├── test_api.py         # API tests
│   ├── test_config.py      # Configuration tests
│   └── test_weather_tool.py # Weather tool tests
├── .env                    # Environment variables
├── .python-version         # Python version for pyenv
├── check_ollama.py         # Script to check if Ollama is running
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── run.py                  # Script to run the API server
├── run_tests.py            # Script to run tests
└── setup.py                # Setup script
```

## Configuration

The agent can be configured using:

1. Environment variables in the `.env` file
2. JSON configuration in `config/default_config.json`

Environment variables take precedence over configuration file values.

### Key Configuration Options

- `OLLAMA_BASE_URL`: URL of the Ollama server (default: http://localhost:11434)
- `OLLAMA_MODEL`: Ollama model to use (default: llama3)
- `OPENWEATHER_API_KEY`: API key for OpenWeatherMap
- `API_PORT`: Port for the FastAPI server (default: 8000)

## Testing

Run the test suite:
```
./run_tests.py
```

This will run all tests and generate a coverage report.

## Extending with New Tools

The agent is designed to be easily extensible with new tools. To add a new tool:

1. Create a new file in the `src/tools/` directory
2. Implement your tool class with appropriate methods
3. Import and initialize your tool in `app.py`
4. Update the prompt template to handle queries related to your tool

### Example: Adding a Calculator Tool

1. Create `src/tools/calculator_tool.py`:
   ```python
   class CalculatorTool:
       def calculate(self, expression):
           try:
               # Safely evaluate the expression
               result = eval(expression, {"__builtins__": {}}, {})
               return f"The result of {expression} is {result}"
           except Exception as e:
               return f"Error calculating {expression}: {str(e)}"
   ```

2. Update `src/app.py` to use the new tool:
   ```python
   from src.tools.calculator_tool import CalculatorTool
   
   # Initialize tools
   weather_tool = WeatherTool()
   calculator_tool = CalculatorTool()
   ```

3. Update the prompt template to handle calculator queries:
   ```
   You are a helpful AI assistant with access to weather information and a calculator.
   If the user asks about the weather, use the weather tool.
   If the user asks for a calculation, use the calculator tool.
   Otherwise, respond helpfully to their query.
   ```

## License

UNLICENSED
