# 设置OpenAI API密钥
import os
import dotenv
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

dotenv.load_dotenv()


# 定义一个命令行聊天机器人的类
class ChatbotWithMemory:
    # 在初始化时，设置花卉行家的角色并初始化聊天模型
    def __init__(self):
        self.llm = ChatTongyi(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template("你是一个花卉行家。你通常的回答不超过30字。"),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = LLMChain(prompt=self.prompt, llm=self.llm, memory=self.memory, verbose=True)

    # 定义一个循环来持续与用户交互
    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            # 如果用户输入“exit”，则退出循环
            if user_input.lower() == 'exit':
                print("再见!")
                break
            response = self.conversation.invoke({"question": user_input})
            print(f"Chatbot: {response['text']}")


# 如果直接运行这个脚本，启动聊天机器人
if __name__ == "__main__":
    bot = ChatbotWithMemory()
    bot.chat_loop()
