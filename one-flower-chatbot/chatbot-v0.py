# 设置OpenAI API密钥
import os
import dotenv
from langchain_community.chat_models import ChatTongyi

dotenv.load_dotenv()
# 导入所需的库和模块
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

# 创建一个聊天模型的实例
chat = ChatTongyi(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))

# 创建一个消息列表
messages = [
    SystemMessage(content="你是一个花卉行家。"),
    HumanMessage(content="朋友喜欢淡雅的颜色，她的婚礼我选择什么花？")
]

# 使用聊天模型获取响应
response = chat(messages)
print(response)
