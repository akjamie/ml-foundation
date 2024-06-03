import os
from dotenv import load_dotenv
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import BaseTool

load_dotenv()

# set env variable for network proxy
os.environ["https_proxy"] = "http://127.0.0.1:7890"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm_model = "gemini-pro"

hf_model = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(hf_model)
model = BlipForConditionalGeneration.from_pretrained(hf_model)

def prompt_rose_slogan(input: str) -> str:
    llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=GOOGLE_API_KEY)
    for i in range(5):
        resp = llm.invoke(input)
        print(resp.content)

class ImageCapTool(BaseTool):
    name = "Image captioner"
    description = "Please write me a slogan about red roses for Valentine’s Day."

    def _run(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs,max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption

    def _arun(self, query:str):
        raise NotImplementedError("This tool does not support async")

def prompt_rose_slogan(image_url: str, topic: str) -> str:
    llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=GOOGLE_API_KEY)
    tools = [ImageCapTool()]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.invoke(input=f"{image_url}\n,{topic}")


if __name__ == "__main__":
    input = "Please write me a slogan about red roses for Valentine’s Day."
    image_url = "https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg"
    #prompt_rose_slogan(input)

    prompt_rose_slogan(image_url, input)
