import logging
import os

import dotenv

dotenv.load_dotenv()

import pandas as pd
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser, RetryWithErrorOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.llms.tongyi import Tongyi
from langchain_google_genai import GoogleGenerativeAI

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class FlowerDescription(BaseModel):
    flower_type: str = Field(..., description="The type of flower")
    price: str = Field(..., description="The price of the flower")
    description: str = Field(..., description="The description of the flower")
    reason: str = Field(..., description="The reason for the flower")


def prompt_flower_with_Tongyi(prompt: str, flower_type: str, price: str) -> str:
    parser = PydanticOutputParser(pydantic_object=FlowerDescription)
    format_instructions = parser.get_format_instructions()
    # print(f'format instruction: {format_instructions}')

    prompt_template = PromptTemplate.from_template(prompt,
                                                   partial_variables={"format_instructions": format_instructions})
    # print(prompt_template)
    model = Tongyi(dashscope_api_key=os.environ['DASHSCOPE_API_KEY'], model_name="qwen-turbo")

    result = model.invoke(prompt_template.format(flower_type=flower_type, price=price))
    return parser.parse(result).dict()


def prompt_flower_with_Gemini(prompt: str, flower_type: str, price: str) -> str:
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    parser = PydanticOutputParser(pydantic_object=FlowerDescription)

    format_instructions = parser.get_format_instructions()

    prompt_template = PromptTemplate.from_template(prompt,
                                                   partial_variables={"format_instructions": format_instructions})
    # print(prompt_template)
    model = GoogleGenerativeAI(model='gemini-pro', google_api_key=os.environ['GOOGLE_API_KEY'], max_output_tokens=1024)
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
    result = model.invoke(prompt_template.format(flower_type=flower_type, price=price))

    return new_parser.parse(result).model_dump()


def prompt_retrywitherror_outputparser_gemini(template: str, query: str) -> str:
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    class Action(BaseModel):
        action: str = Field(description="action to take")
        action_input: str = Field(description="input to the action")

    parser = PydanticOutputParser(pydantic_object=Action)
    prompt_template = PromptTemplate(template=template,
                                     input_variables=["query"],
                                     partial_variables={"format_instructions": parser.get_format_instructions()}, )
    prompt_value = prompt_template.format_prompt(query=query)

    # fix the response with OutputFixingParser
    model = GoogleGenerativeAI(model='gemini-pro', google_api_key=os.environ['GOOGLE_API_KEY'], max_output_tokens=1024,
                               temperature=0.0)
    fix_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

    bad_response = '{"action": "search"}'
    # parser.parse(bad_response)
    parsed_response = fix_parser.parse(bad_response)
    print(f'OutputFixingParser, response:{parsed_response}')

    # fix response with RetryWithErrorOutputParser
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)
    parsed_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
    print(f'RetryWithErrorOutputParser, response:{parsed_result}')

def prompt_retrywitherror_outputparser_tongyi(template: str, query: str) -> str:
    class Action(BaseModel):
        action: str = Field(description="action to take")
        action_input: str = Field(description="input to the action")

    parser = PydanticOutputParser(pydantic_object=Action)
    prompt_template = PromptTemplate(template=template,
                                     input_variables=["query"],
                                     partial_variables={"format_instructions": parser.get_format_instructions()}, )
    prompt_value = prompt_template.format_prompt(query=query)

    # fix the response with OutputFixingParser
    model = Tongyi(model_name='qwen-turbo', dashcope_api_key=os.environ['DASHSCOPE_API_KEY'], max_output_tokens=1024,
                               temperature=0.0)
    fix_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

    bad_response = '{"action": "search"}'
    # parser.parse(bad_response)
    parsed_response = fix_parser.parse(bad_response)
    print(f'OutputFixingParser, response:{parsed_response}')

    # fix response with RetryWithErrorOutputParser
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)
    parsed_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
    print(f'RetryWithErrorOutputParser, response:{parsed_result}')

if __name__ == '__main__':
    df = pd.DataFrame(columns=['flower_type', 'price', 'description', 'reason'])

    flowers = ['Rose', 'Lily', 'Tulip', 'Orchid']
    prices = ['10', '20', '30', '40']

    template = """
    You are a flower expert. You are given a flower type and a price.
    The flower type is {flower_type} and the price is {price}.
    Please generate a flower description, a reason for the flower, and a price range for the flower.
    {format_instructions}
    """
    # for flower_type, price in zip(flowers, prices):
    #     # result = prompt_flower_with_Tongyi(template, flower_type, price)
    #     result = prompt_flower_with_Gemini(template, flower_type, price)
    #     df.loc[len(df)] = result
    #
    # print("final output：", df.to_dict(orient='records'))

    template = """
    Answer the user query.\n{format_instructions}\n{query}\n.
    """
    query = "What are the colors of Orchid?"
    # prompt_retrywitherror_outputparser_gemini(template, query=query)
    prompt_retrywitherror_outputparser_tongyi(template, query=query)
