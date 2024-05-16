import logging
import os
from typing import List

from langchain_google_genai import GoogleGenerativeAI

logging.getLogger().setLevel(logging.DEBUG)

import dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain_community.llms.tongyi import Tongyi
from langchain_core.language_models import BaseLLM

dotenv.load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RouterTemplate


def create_routing_chain(model: BaseLLM, destinations: List[str]):
    router_template = RouterTemplate.format(destinations="\n".join(destinations))
    print("路由模板:\n", router_template)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(), )
    print("路由提示:\n", router_prompt)
    router_chain = LLMRouterChain.from_llm(model, router_prompt, verbose=True)
    return router_chain


if __name__ == "__main__":
    # 构建两个场景的模板
    flower_care_template = """你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
                            下面是需要你来回答的问题:
                            {input}"""
    flower_deco_template = """你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
                            下面是需要你来回答的问题:
                            {input}"""

    # 构建提示信息
    prompt_infos = [
        {
            "key": "flower_care",
            "description": "适合回答关于鲜花护理的问题",
            "template": flower_care_template,
        },
        {
            "key": "flower_decoration",
            "description": "适合回答关于鲜花装饰的问题",
            "template": flower_deco_template,
        }
    ]

    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.0,
        max_tokens=1024,
    )

    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    llm = GoogleGenerativeAI(temperature=0.9, model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

    chain_map = {}
    for info in prompt_infos:
        prompt = PromptTemplate(template=info['template'], input_variables=["input"])
        print("目标提示:\n", prompt)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        chain_map[info["key"]] = chain

    # 构建路由链
    destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
    router_chain = create_routing_chain(llm, destinations)

    default_chain = ConversationChain(llm=llm, output_key="text", verbose=True)
    template = """
    Prompt after formatting:
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    
    Human: {input}
    AI:
    """
    default_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template), verbose=True)

    # chain = MultiPromptChain(router_chain=router_chain, destination_chains=chain_map, default_chain=default_chain,
    #                          verbose=False)
    prompt_infos = [
        {
            "name": "flower_care",
            "description": "适合回答关于鲜花护理的问题",
            "prompt_template": flower_care_template,
        },
        {
            "name": "flower_decoration",
            "description": "适合回答关于鲜花装饰的问题",
            "prompt_template": flower_deco_template,
        }
    ]
    chain = MultiPromptChain.from_prompts(llm=llm, prompt_infos=prompt_infos, default_chain=default_chain,
                                          verbose=True)

    # testing
    print("测试:\n", chain.invoke({'input': '我应该把新买的百合花放在哪儿呢，卧室，客厅还是阳台？'}))
    print("测试:\n", chain.invoke({'input': '如何为婚礼场地装饰花朵？'}))
    print("测试:\n", chain.invoke({'input': '男宝宝，5个月大，小名小橙子，请给他起一个阳光的English name？'}))
