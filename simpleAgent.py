from dotenv import load_dotenv
import openai
from tavily import TavilyClient
import requests
import re
import os

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

def get_weather(city: str) -> str:
    """根据城市名称返回规整化的实时天气信息
        @Parameter:
            - city: str  
                城市名称
        @Return:
            - formatted_result: str
                规整后的天气信息，包括温度，云覆盖度，紫外线强度，可见度
    """

    url = f"https://wttr.in/{city}?format=j1"
    try:
        # 查询天气信息
        response = requests.get(url)
        response.raise_for_status()
        jsonResponse = response.json()

        # 规整天气信息
        current_condition = jsonResponse["current_condition"][0]
        tempC = current_condition["temp_C"]
        cloudcover = current_condition["cloudcover"]
        uvIndex = current_condition["uvIndex"]
        visibility = current_condition["visibility"]

        formatted_result = f"""
                            temperature: {tempC}celsius,
                            cloudcover: {cloudcover},
                            uvIndex: {uvIndex},
                            visibility: {visibility}
                            """
        
        return formatted_result
    except requests.exceptions.RequestException as e:
        return f"天气信息访问失败，错误信息:{e}"
    except (KeyError, IndexError) as e:
        return f"触发{e}, 可能是城市名输入错误"
        
