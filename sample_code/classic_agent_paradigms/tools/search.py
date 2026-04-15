from dotenv import load_dotenv
from os import getenv
import serpapi
import tavily
from tavily.errors import *
from serpapi.exceptions import *

def quickSearch(query: str) -> str:
    """
    Quick Search tool based on tavily, helping agent getting summarized answer
    @param
        - query: str
            - question to be answerd
    @return
        - answer: str
            - answer to the query
    """
    try:    
        tavily_api_key = getenv("TAVILY_API_KEY")
        client = tavily.TavilyClient(tavily_api_key)
        response = client.search(query=query,
                                 search_depth='basic',
                                 include_answer=True)
        
        answer = response["answer"]
        return answer
    except (MissingAPIKeyError, InvalidAPIKeyError) as e:
        print(f"---{e}, please rewrite the api_key in config---")
        return "Missing/Invalid tavily api key, could not use quickSearch."
    except (BadRequestError, ForbiddenError) as e:
        print(f"---{e}Failed to get quick research results from tavily---")
        return "Failed to get results from quickSearch, please call again on the tool"

def search(query: str, serpapi_api_key: str = None) -> str:
    """
    Detailed search that will return more detailed answers
    @param
        - query: str
            - query to search on google
    @return
        - answer: str
            - structured results of the search results
    """
    try:
        serpapi_api_key = getenv("SERPAPI_API_KEY")
        client = serpapi.Client(api_key=serpapi_api_key)
        params = {
                "engine": "google",
                "q": query,
                "gl": "cn",
                "hl": "zh-cn"
            }
        
        results = client.serach(params)
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except APIKeyNotProvided as e:
        print(f"---{e} Missing serpapi api key---")
        return "Missing serpapi api key, cannot use search tool."
    except (HTTPConnectionError, HTTPError, TimeoutError) as e:
        print(f"---{e} Failed to get search results from serpapi---")
        return "Failed to get results from search, please call again on the tool"