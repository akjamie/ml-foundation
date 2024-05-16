import logging
import os
from typing import List, Union, Any, Dict

import dotenv
from pydantic import BaseModel, Field

dotenv.load_dotenv()
from langchain.chains.sequential import SequentialChain
from langchain_core.language_models import BaseLLM

from langchain.chains.llm import LLMChain
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate


def simple_llmchain(template: str, query: str) -> str:
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.0,
        max_tokens=1024,
    )
    prompt = PromptTemplate(template=template, input_variables=["flower"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke(query)


def simple_llmchain_apply(template: str, input_variables: list[str], parameters: list[object]) -> str:
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.0,
        max_tokens=1024,
    )
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.apply(parameters)


def simple_llmchain_generate(template: str, input_variables: list[str], parameters: list[object]) -> str:
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.0,
        max_tokens=1024,
    )
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.generate(parameters)


# define method to create LLMChain with model, prompt, and input_variables
def create_llmchain(model: BaseLLM, template: str, input_variables: list[str], output_key: str) -> LLMChain:
    prompt_template = PromptTemplate(template=template, input_variables=input_variables)
    chain = LLMChain(llm=model, prompt=prompt_template, output_key=output_key)
    return chain


# define method to create SequentialChain with model, prompts, and input_variables
def prompt_with_sequentialchain(model: BaseLLM, prompts: List[Union[Dict[str, str], Any]], input: Any) -> LLMChain:
    if len(prompts) == 0:
        return None

    class Slogan(BaseModel):
        slogan: str = Field(..., description="The slogan of the flower")

    chains = []
    for item in prompts:
        if isinstance(item, dict):
            input_dict = item  # Avoid using the built-in name 'map'
            prompt = input_dict.get("prompt")  # Safely access the item
            input_variables = input_dict.get('input_variables')
            output_key = input_dict.get('output_key')
            if prompt is None or input_variables is None or output_key is None:
                continue  # Skip dictionary items that don't meet the requirements
            try:
                chain = create_llmchain(model, prompt, input_variables, output_key)
                chains.append(chain)
            except Exception as e:
                print(f"An error occurred while creating an LLMChain: {e}")

    chain = SequentialChain(chains=chains, input_variables=prompts[0]['input_variables'],
                            output_variables=[obj["output_key"] for obj in prompts], verbose=True)
    return chain.invoke(input)


def prompt_with_sequentialchain_with_output(model: BaseLLM, prompts: List[Union[Dict[str, str], Any]],
                                            input: Any) -> LLMChain:
    if len(prompts) == 0:
        return None

    chains = []
    for item in prompts:
        if isinstance(item, dict):
            input_dict = item  # Avoid using the built-in name 'map'
            prompt = input_dict.get("prompt")  # Safely access the item
            input_variables = input_dict.get('input_variables')
            output_key = input_dict.get('output_key')
            if prompt is None or input_variables is None or output_key is None:
                continue  # Skip dictionary items that don't meet the requirements
            try:
                chain = create_llmchain(model, prompt, input_variables, output_key)
                chains.append(chain)
            except Exception as e:
                print(f"An error occurred while creating an LLMChain: {e}")

    chain = SequentialChain(chains=chains, input_variables=prompts[0]['input_variables'],
                            output_variables=[obj["output_key"] for obj in prompts], verbose=True)
    return chain.invoke(input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    template = "{flower}的花语是?"
    result = simple_llmchain(template, query="玫瑰")
    print(result)

    template = "{flower}在{season}的花语是?"
    input_list = [{"flower": "玫瑰", 'season': "夏季"}, {"flower": "百合", 'season': "春季"},
                  {"flower": "郁金香", 'season': "秋季"}]
    # print(simple_llmchain_apply(template, ['flower', 'season'], input_list))

    # print(simple_llmchain_generate(template, ['flower', 'season'], input_list))

    model = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.0,
        max_tokens=1024,
    )

    template_1 = """
     你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。 花名: {name} 颜色: {color} 植物学家: 这是关于上述花的介绍:
     """

    template_2 = """ 
    你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。 鲜花介绍: {introduction} 花评人对上述花的评论:
    """

    template_3 = """ 
    你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。 鲜花介绍: {introduction} 花评人对上述花的评论: {review} 社交媒体帖子: 
    """

    inputs = [{
        "prompt": template_1,
        "input_variables": ["name", "color"],
        "output_key": "introduction"
    }, {
        "prompt": template_2,
        "input_variables": ["introduction"],
        "output_key": "review"
    }, {
        "prompt": template_3,
        "input_variables": ["introduction", "review"],
        "output_key": "social_post_text"
    }]
    result = prompt_with_sequentialchain_with_output(model, inputs, {"name": "玫瑰", "color": "红色"})
    print(result["social_post_text"])
