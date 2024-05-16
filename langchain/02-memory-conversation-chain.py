import os

import dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain_community.llms.tongyi import Tongyi

dotenv.load_dotenv()
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.0,
        max_tokens=1024,
    )

    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # llm = GoogleGenerativeAI(temperature=0.9, model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

    # conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    # conversation = ConversationChain(llm=llm, memory=ConversationBufferWindowMemory(k=1))
    # conversation = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))
    conversation = ConversationChain(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=300))
    result = conversation.run("我姐姐明天要过生日，我需要一束生日花束。")
    # print("第一次对话后的记忆:\n", conversation.memory.buffer)
    print("第一次对话结果:\n", result)

    result = conversation.run("她喜欢粉色玫瑰，颜色是粉色的。")
    # print("第二次对话后的记忆:\n", conversation.memory.buffer)
    print("第二次对话结果:\n", result)

    result = conversation.run("我又来了，还记得我昨天为什么要来买花吗？")
    print("第三次对话结果:\n", result)
    result = conversation.run("我姐姐喜欢的花是什么，颜色呢？")
    print("第四次对话结果:\n", result)
