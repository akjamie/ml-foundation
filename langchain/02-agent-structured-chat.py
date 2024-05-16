import logging
import os

from langchain_community.llms.tongyi import Tongyi

from common.annotations import proxy

import dotenv
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain.globals import set_debug
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_core.runnables import Runnable
from langchain_google_genai import GoogleGenerativeAI

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

set_debug(True)

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

if __name__ == "__main__":
    os.environ["https"] = "http://127.0.0.1:7890"
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"],
                             temperature=0.2,
                             top_p=0.8,
                             max_tokens=1000,
                             stream=True
                             )
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        temperature=0.2,
        top_p=0.8,
        max_tokens=1000,
        stream=True,
        model_name="qwen-turbo"
    )

    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)


    # @proxy("http://127.0.0.1:7890")
    async def main():
        response = await agent_executor.ainvoke({"input": "What are the headers on "
                                                          "python.langchain.com/docs/modules?"})
        print(response)


    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
