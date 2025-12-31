# c:\Users\Sai Krishna -INT-363\Desktop\Mcp-server\server.py
import os
import math
import logging
import requests
import json
from typing import Any
from dotenv import load_dotenv
from mcp.server import FastMCP

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the MCP server with a name
mcp = FastMCP("ToolServer")

# --- Tool Definitions ---

@mcp.tool()
def get_weather(city: str) -> str:
    """Gets the current weather for a specified city."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: Weather API key not configured."
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric" 
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        weather_description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        
        return (f"Weather in {city}: {weather_description}, "
                f"Temperature: {temp}°C, "
                f"Humidity: {humidity}%, "
                f"Wind Speed: {wind_speed} m/s")
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except (KeyError, IndexError):
        return "Error: Could not parse weather data. The API response may be malformed or the city is invalid."

@mcp.tool()
def internet_search(query: str, num_results: Any = 5) -> str:
    """Performs an internet search for general world knowledge. Do NOT use for internal company policy."""
    num_results = int(num_results) if num_results else 5
    logging.info(f"Performing Google search for query: '{query}'")
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")

    if not api_key or not cx:
        return "Error: Google API Key or Search Engine ID (CX) is not configured."

    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': cx,
            'q': query,
            'num': num_results
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        search_results = response.json()

        if 'items' not in search_results or not search_results['items']:
            return "No search results found."

        # Format the results into a readable string.
        formatted_output = []
        for i, item in enumerate(search_results['items'], 1):
            title = item.get('title', 'No Title')
            snippet = item.get('snippet', 'No snippet available.').replace('\n', '')
            link = item.get('link', 'No URL available.')
            formatted_output.append(f"{i}. {title}\n   - {snippet}\n   - URL: {link}")

        return "\n\n".join(formatted_output)
    except requests.exceptions.RequestException as e:
        return f"An error occurred during the search: {e}"

@mcp.tool()
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtracts the second number from the first."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divides the first number by the second. Raises ValueError if the divisor is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@mcp.tool()
def sqrt(a: float) -> float:
    """Calculates the square root of a number. Raises ValueError for negative numbers."""
    return math.sqrt(a)

# --- Run the Server ---

if __name__ == "__main__":
    logging.info("Starting MCP server...")
    # The server communicates over standard input/output (stdio)
    mcp.run(transport="stdio")
