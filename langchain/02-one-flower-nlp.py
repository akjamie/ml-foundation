import os
import logging

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

logging.basicConfig()
logging.getLogger('langchain_community').setLevel(logging.DEBUG)


# define method to invoke the LLM to generate the output based on given template and variables using google gemini
def prompt_flower_description_with_gemini(prompt_template: PromptTemplate, price: str, flower_name: str) -> str:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = GoogleGenerativeAI(temperature=0.9, model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    return llm.invoke(prompt_template.format(price=price, flower_name=flower_name))


def prompt_flower_description_with_tongyi(prompt_template: PromptTemplate, price: str, flower_name: str) -> str:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    llm = Tongyi(dashscope_api_key=DASHSCOPE_API_KEY, model_name='qwen-turbo')
    return llm.invoke(prompt_template.format(price=price, flower_name=flower_name))

if __name__ == "__main__":
    # set env variable for network proxy
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"

    template = """
    You are a professional flower shop copywriter.
    Can you provide a short and engaging description of {flower_name} priced at {price}?
    {format_instructions}
     """

    # prompt = PromptTemplate(
    #     input_variables=["price", "flower_name"],
    #     template=template,
    # )

    response_schemas = [
        ResponseSchema(name="description", description="Description of flower"),
        ResponseSchema(name="reason", description="Reason")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt_template = PromptTemplate.from_template(template,
                                                   partial_variables={"format_instructions": format_instructions})

    flowers = ["Rose", "Lily", "Carnation"]
    prices = ["50", "30", "20"]

    df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

    for flower, price in zip(flowers, prices):
        # print(prompt_flower_description_with_gemini(prompt_template, price, flower))
        result = prompt_flower_description_with_tongyi(prompt_template, price, flower)
        parsed_output = output_parser.parse(result)
        parsed_output['flower'] = flower
        parsed_output['price'] = price

        df.loc[len(df)] = parsed_output

    print(df.to_dict(orient="records"))
    df.to_csv("output/flowers_with_descriptions.csv", index=False, encoding="utf-8", header=["flower", "price", "description", "reason"])
    # time.sleep(30)
