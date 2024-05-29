from loguru import logger
import os
import dotenv

dotenv.load_dotenv()

from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms.tongyi import Tongyi
from langchain.prompts import PromptTemplate

logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

llm = Tongyi(dashscope_api_key=os.environ["DASHSCOPE_API_KEY"])
prompt = PromptTemplate.from_template("1 + {number} = ")

# this chain will both print to stdout (because verbose=True) and write to 'output.log'
# if verbose=False, the FileCallbackHandler will still write to 'output.log'
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler], verbose=True)
answer = chain.invoke({"number": 2})
logger.info(answer)
