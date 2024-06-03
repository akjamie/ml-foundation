# 导入一个搜索UID的工具
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_core.language_models import BaseLLM
from tools.search_tool import get_UID
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool


# 通过LangChain代理找到UID的函数
def lookup_v(flower_type: str, llm: BaseLLM):
    # 初始化大模型
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # 寻找UID的模板
    template = """given the {flower} I want you to get a related Weibo UID.
                  Your answer should contain only a UID.
                  The URL always starts with https://weibo.com/u/
                  for example, if https://weibo.com/u/1669879400 is her Weibo, then 1669879400 is her UID
                  This is only the example don't give me this, but the actual UID
                  If you cannot find a UID, please output as I-DONOT-KNOW.
                  """

    # 完整的提示模板
    prompt_template = PromptTemplate(
        input_variables=["flower"], template=template
    )

    # 代理的工具
    tools = [
        Tool(
            name="Crawl Google for Weibo page",
            func=get_UID,
            description="useful for when you need get the Weibo UID",
        )
    ]

    # 初始化代理
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # 返回找到的UID
    ID = agent.invoke(prompt_template.format_prompt(flower=flower_type))

    return ID
