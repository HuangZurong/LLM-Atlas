from flashboot_core.utils import project_utils
import os
import dotenv
from loguru import logger

dotenv.load_dotenv(project_utils.get_root_path() / ".env", override=True)
print = logger.info

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm


def get_weather(city: str) -> dict:
    print(f"--工具：get_weather被调用，城市：{city}")
    city_normalized = city.lower().replace(" ", "")

    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "New York City has a mild temperature of 25 degrees Celsius and clear sky.",
        },
        "beijing": {
            "status": "success",
            "report": "Beijing has a mild temperature of 15 degrees Celsius and clear sky.",
        },
        "hangzhou": {
            "status": "success",
            "report": "Hangzhou has a mild temperature of 18 degrees Celsius and clear sky.",
        },
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {
            "status": "error",
            "report": f"City {city} not found in weather database.",
        }


print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_BASE"))

root_agent = Agent(
    name="weather_agent",
    model=LiteLlm(
        model="openai/gemini-3-flash-preview",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    ),
    description=("Agent to answer questions about weather in a city."),
    instruction=(
        "You are a helpful agent who can answer user questions about the weather in a city."
    ),
    tools=[get_weather],
)
