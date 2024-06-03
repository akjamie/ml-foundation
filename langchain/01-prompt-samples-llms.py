import json
import os

# from adodbapi.is64bit import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()


# simple langchain demo with google gemini-pro
def prompt_with_gemini(prompt: str, topic: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import LLMChain
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    promptTemplate = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=llm, prompt=promptTemplate, verbose=True)
    resp = chain.invoke(input=topic)

    return resp


# simple langchain demo with openai gpt-3.5-turbo
def prompt_with_openai(prompt: str, topic: str) -> str:
    try:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
        from langchain_openai import OpenAI
        from langchain.chains import LLMChain

        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=OPENAI_API_KEY,
                     openai_organization=OPENAI_ORGANIZATION)
        promptTemplate = PromptTemplate.from_template(prompt)
        chain = LLMChain(llm=llm, prompt=promptTemplate, verbose=True)

        return chain.invoke(input=topic)
    except Exception as e:
        print(e)

def prompt_with_tongyi(prompt: str, topic: str) -> str:
    from langchain_community.llms import Tongyi
    from langchain.chains import LLMChain

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    llm = Tongyi(dashscope_api_key=DASHSCOPE_API_KEY, model_name="qwen-turbo")
    promptTemplate = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=llm, prompt=promptTemplate, verbose=True)

    return chain.invoke(input=topic)


if __name__ == "__main__":
    # set env variable for network proxy
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    prompt = "You are a content creator. Write me a tweet about {topic}."
    topic = "how ai is really cool"
    # resp = prompt_with_gemini(prompt, topic)
    # print(f'response from gemini: {json.dumps(resp, indent=2)}\n')

    # resp = prompt_with_openai(prompt, topic)
    # print(f'response from openai: {json.dumps(resp, indent=2)}\n')

    #no need proxy for Tongyi
    os.environ.pop("https_proxy")
    resp = prompt_with_tongyi(prompt, topic)
    print(f'response from openai: {json.dumps(resp, indent=2)}\n')
