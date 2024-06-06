# 设置OpenAI API密钥
import logging
import os
import re
from typing import List

import dotenv
import gradio as gr
from langchain import hub
from langchain.globals import set_debug
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

dotenv.load_dotenv()
set_debug(True)


# 导入所需的库和模块
class ChatbotWithRetrieval:
    def __init__(self, dir: str):
        self.base_dir = dir
        self.documents = self.load_documents_from_dir(self.base_dir)
        if not self.documents:
            logging.warning("No compatible documents found in the directory.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=5)
        all_splits = text_splitter.split_documents(self.documents)
        embeddings = DashScopeEmbeddings(model="text-embedding-v1",
                                         dashscope_api_key=self.get_api_key())

        self.vectorstore = FAISS.from_documents(all_splits, embeddings)
        self.llm = ChatTongyi(dashscope_api_key=self.get_api_key())

        self.retriever = self.vectorstore.as_retriever()
        self.conversation_history = ""
        template = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
        """
        # self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt = ChatPromptTemplate.from_template(template)

        self.qa = ({
                       "context": self.retriever.with_config(run_name="Docs"),
                       "question": RunnablePassthrough()
                   }
                   | self.prompt
                   | self.llm
                   | StrOutputParser()
                   )

    @staticmethod
    def get_api_key() -> str:
        return os.getenv("DASHSCOPE_API_KEY")

    def load_documents_from_dir(self, dir: str) -> List:
        documents = []
        try:
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                if self.is_valid_file(file_path):
                    loader = self.get_loader(file_path)
                    documents.extend(loader.load())
        except Exception as e:
            logging.error(f"Error loading documents from {dir}: {e}")
        return documents

    @staticmethod
    def is_valid_file(file_path: str) -> bool:
        valid_extensions = [".txt", ".pdf", ".docx"]
        return any(file_path.endswith(ext) for ext in valid_extensions)

    @staticmethod
    def get_loader(file_path: str) -> 'Loader':
        if file_path.endswith(".txt"):
            return TextLoader(file_path)
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path)
        if file_path.endswith(".docx"):
            return Docx2txtLoader(file_path)
        raise ValueError(f"Unsupported file type: {file_path}")

    def chat_loop(self):
        print("Chatbot is started!  Enter 'exit' to exit program.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("bye!")
                break
            print(f'Retriever search result:{self.retriever.invoke(user_input)}')
            response = self.qa.invoke({"question": user_input})
            print(f"Chatbot: {response}")

    # @proxy("http://127.0.0.1:7890")
    # @traceable()
    def get_response(self, user_input):
        response = self.qa.invoke(user_input)

        self.conversation_history += f"You: {user_input}\nChatbot: {response}\n"

        if (None != self.conversation_history and len(self.conversation_history) > 300):
            conversation_pattern = r"(You: .*\nChatbot: .*)+"
            # Find all conversation blocks
            conversations = re.findall(conversation_pattern, self.conversation_history, re.MULTILINE)
            # Get the last conversation block
            if(None != conversations and len(conversations) > 3):
                last_3_conversation = conversations[-3:]
                self.conversation_history = "\n".join(last_3_conversation)

        return self.conversation_history


if __name__ == "__main__":
    bot = ChatbotWithRetrieval("docs")
    # bot.get_response({"question":"易速鲜花"})

    interface = gr.Interface(
        fn=bot.get_response,
        inputs="text",
        outputs="text",
        live=False,
        title="One flow chatbot",
        description="Input your question, and the bot will give you the answer."
    )
    interface.launch()
