# 设置OpenAI API的密钥
import os

import dotenv
from langchain import hub
from langchain_google_genai import GoogleGenerativeAI
from langchain.globals import set_debug

dotenv.load_dotenv()
set_debug(True)

# 导入库
from langchain.agents import load_tools, create_react_agent, AgentExecutor

os.environ["https_proxy"] = "http://127.0.0.1:7890"
# 初始化模型和工具
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"],
                         temperature=0.2,
                         top_p=0.8,
                         max_tokens=1000,
                         stream=True
                         )
tools = load_tools(
    ["arxiv"],
)

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# 运行链
response = agent_executor.invoke({"input": "介绍一下2005.14165这篇论文的创新点?"})
print(response)