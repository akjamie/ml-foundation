import logging
import os

import dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain.globals import set_debug
from langchain_community.llms.tongyi import Tongyi
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

set_debug(True)


if __name__ == "__main__":
    os.environ["https"] = "http://127.0.0.1:7890"
    search = SerpAPIWrapper()
    tools = [
        Tool(name="Intermediate Answer", func=search.run, description="useful for when you need to ask with search", )]

    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        temperature=0.2,
        top_p=0.8,
        max_tokens=1000,
        stream=True,
        model_name="qwen-turbo"
    )

    prompt = hub.pull("hwchase17/self-ask-with-search")
    agent = create_self_ask_with_search_agent(
        llm=llm,
        tools=tools,
        prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    agent_executor.invoke({"input": "使用玫瑰作为国花的国家的首都是哪里?"})
