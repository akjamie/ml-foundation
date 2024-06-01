# 设置OpenAI API密钥
import os

import dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.globals import set_debug
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()
set_debug(True)


# 导入所需的库和模块


class ChatbotWithRetrieval:
    def __init__(self, dir: str):
        base_dir = dir
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            if file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=5)
        all_splits = text_splitter.split_documents(documents)
        self.vectorstore = Qdrant.from_documents(documents=all_splits,
                                                 collection_name="langchain-collection",
                                                 location=":memory:",
                                                 embedding=DashScopeEmbeddings(model="text-embedding-v1",
                                                                               dashscope_api_key=os.getenv(
                                                                                   "DASHSCOPE_API_KEY"))
                                                 )
        self.llm = ChatTongyi(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))

        self.memory = ConversationSummaryMemory(llm=self.llm, memory_key="chat_history", return_messages=True)
        self.retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.retriever, memory=self.memory)

    # 定义一个循环来持续与用户交互
    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            # 如果用户输入“exit”，则退出循环
            if user_input.lower() == 'exit':
                print("再见!")
                break
            print(f'Retriever search result:{self.retriever.invoke(user_input)}')
            response = self.qa.invoke({"question": user_input})
            print(f"Chatbot: {response['answer']}")


# 如果直接运行这个脚本，启动聊天机器人
if __name__ == "__main__":
    bot = ChatbotWithRetrieval("./docs")
    bot.chat_loop()
