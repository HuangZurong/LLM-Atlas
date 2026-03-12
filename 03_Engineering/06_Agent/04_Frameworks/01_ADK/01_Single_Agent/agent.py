from flashboot_core.utils import project_utils
from dotenv import load_dotenv
from loguru import logger

load_dotenv(project_utils.get_root_path() / ".env", override=True)
print = logger.info

from google.adk.agents.llm_agent import Agent


def get_current_time(city: str) -> dict:
    return {"status": "success", "city": city, "time": "10:30 AM"}


root_agent = Agent(
    model="gemini-3-flash-preview",
    name="root_agent",
    description="报告指定城市的当前时间。",
    instruction="你是一个有用的助手，可以报告城市的当前时间。为此使用 'get_current_time' 工具。",
    tools=[get_current_time],
)
