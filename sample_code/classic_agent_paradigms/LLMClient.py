from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from os import getenv
from typing import List, Dict

class OpenAICompatibleClient:
    """Encapsulated openai compatible LLM client"""
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = 120):
        self.model = model
        if not all([model, api_key, base_url]):
            raise ValueError("Missing key information, please make sure you set model, api_key, and base_url in the config.")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(self, message: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        Call LLM through api to get response.
        @Parameters:
            - message: List[Dict[str, str]]
                - message sends to the LLM
        @Return:
            - completeResponse: str
                - complete response of the LLM
        """
        print(f"🧠---正在调用LLM service, model_id: {self.model}---")
        try:
            response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=temperature,
                    stream=True
                )
            
            collected_content = []

            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)

            print()
            print("✅Successfully called LLM service.")
            completeResponse = "".join(collected_content)
            return completeResponse
        
        except OpenAIError as e:
            print(f"Error occured when calling LLM service, error message: {e}")
        except Exception as e:
            print(f"Unexpected error, error message{e}")

if __name__ == "__main__":
    load_dotenv()
    api_key = getenv("API_KEY")
    model = getenv("MODEL_ID")
    base_url = getenv("BASE_URL")
    client = OpenAICompatibleClient(model, api_key, base_url)

    message = [
            {"role": "system", "content": "你是一名AI助手"},
            {"role": "user", "content": "你好，介绍一下你自己"}
        ]
    
    client.generate(message, 0.1)