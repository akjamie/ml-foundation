import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import logging
from flask import Flask, request, render_template

app = Flask(__name__)

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)


# load documents from path
def load_documents(base_dir: str) -> list:
    documents = []
    for file in os.listdir(base_dir):
        # 构建完整的文件路径
        file_path = os.path.join(base_dir, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    return documents


# split documents
def split_documents(documents: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    return chunked_documents


# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
def store_documents(chunked_documents: list) -> Qdrant:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    vectorstore = Qdrant.from_documents(
        documents=chunked_documents,  # 以分块的文档
        embedding=embeddings,  # 用OpenAI的Embedding Model做嵌入
        location=":memory:",  # in-memory 存储
        collection_name="my_documents", )

    return vectorstore


# define QA retrieval
def qa_retrieval(vectorstore) -> RetrievalQA:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

    # Use MultiQueryRetriever with your existing retriever and Gemini LLM
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

    # Create RetrievalQA chain with Gemini LLM and retriever
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)

    return qa_chain


def facade(base_dir: str):
    documents = load_documents(base_dir)
    chunked_documents = split_documents(documents)
    vectorstore = store_documents(chunked_documents)
    qa_chain = qa_retrieval(vectorstore)

    return qa_chain


@app.route('/', methods=['GET', 'POST'])
def home(qa_chain=None):
    if qa_chain is None:
        base_dir = './data'
        qa_chain = facade(base_dir)

    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')

        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})
        print(f'question:[{question}]')

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    # set env variable for network proxy
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    # 加载Documents
    base_dir = './data'

    document = load_documents(base_dir)
    chunked_documents = split_documents(document)
    vectorstore = store_documents(chunked_documents)
    qa_chain = qa_retrieval(vectorstore)

    # pass query to qa_chain and get result from qa_chain
    result = qa_chain({"query": "董事长致辞中提到的企业精神指的是什么?"})
    print(result)

    #
    # home(qa_chain)
    # app.run(host='0.0.0.0', debug=True, port=5000)
