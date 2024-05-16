import logging
import os

import dotenv
from langchain.chains.llm_math.base import LLMMathChain
from langchain.globals import set_debug
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import load_chat_planner, load_agent_executor, PlanAndExecute
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

set_debug(True)

if __name__ == "__main__":
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    search = SerpAPIWrapper()
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"],
                             temperature=0.2,
                             top_p=0.8,
                             max_tokens=1000,
                             stream=True
                             )
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [Tool(name="Search", func=search.run,
                  description="useful for when you need to answer questions about current events"),
             Tool(name="Calculator", func=llm_math_chain.run,
                  description="useful for when you need to answer questions about math"), ]

    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"],
                                   temperature=0.2,
                                   top_p=0.8,
                                   max_tokens=1000,
                                   stream=True)
    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
    agent.invoke({"input": "在纽约，100美元能买几束玫瑰?"})
