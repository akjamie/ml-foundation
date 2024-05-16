import os

import dotenv
import torch
import transformers
from langchain import HuggingFacePipeline
from langchain.chains.llm import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()


def prompt_langchain_huggingface(template: str, question: str) -> str:
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.7, "max_length": 256},
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.invoke(input={"question": question})


def prompt_langchain_huggingface_pipeline(template: str, flower_details: str) -> str:
    pipeline = transformers.pipeline("text-generation",
                                     model="meta-llama/Llama-2-7b-chat-hf",
                                     torch_dtype=torch.float16,
                                     device_map="auto",
                                     max_length=1000)

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
    prompt = PromptTemplate(template=template, input_variables=["flower_details"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.invoke(input={"flower_details": flower_details})


if __name__ == "__main__":
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    template = """
    Question: {question}
    Answer:
    """
    question = "Rose is which type of flower?"
    print(prompt_langchain_huggingface(template, question))

    template = """ 
    为以下的花束生成一个详细且吸引人的描述：
    花束的详细信息： 
    ```{flower_details}``` 
    """
    flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"
    print(prompt_langchain_huggingface_pipeline(template, flower_details))
