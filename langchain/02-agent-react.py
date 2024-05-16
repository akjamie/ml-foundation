import logging
import os
import warnings
from common.annotations import proxy

import dotenv
from langchain.globals import set_debug
from langchain.agents import load_tools, create_structured_chat_agent, create_react_agent, initialize_agent, AgentType, \
    AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_vertexai import VertexAI

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

set_debug(True)


# warnings.filterwarnings("ignore", category=DeprecationWarning)

def react_with_tongyi(template: str) -> AgentExecutor:
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        temperature=0.2,
        top_p=0.8,
        max_tokens=1000,
        stream=True,
        model_name="qwen-turbo",
    )

    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # agent.invoke({"input": "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"})

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    return agent_executor


@proxy(value="http://127.0.0.1:7890")
def react_with_vertexai(template: str) -> AgentExecutor:
    llm = VertexAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"],
                   temperature=0.2,
                   top_p=0.8,
                   max_tokens=1000,
                   stream=True
                   )

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_exec = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    return agent_exec


@proxy(value="http://127.0.0.1:7890")
def react_with_gemini(template: str) -> AgentExecutor:
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"],
                             temperature=0.2,
                             top_p=0.8,
                             max_tokens=1000,
                             stream=True
                             )

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_exec = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    return agent_exec


if __name__ == "__main__":
    template = '''Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}'''

    input = "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
    # agent_executor = react_with_tongyi(template)
    # agent_executor.invoke({"input": input})

    agent_executor = react_with_gemini(template)
    agent_executor.invoke({"input": "Currently, what is the average price of roses in the market? If I want to markup "
                                    "the price by 15% for selling, how should I determine the selling price?"})
