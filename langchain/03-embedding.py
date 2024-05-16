import os
import dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

dotenv.load_dotenv()
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.storage import InMemoryStore


# define a method to return embedder
def get_embedder():
    store = InMemoryStore()
    embeddings_model = HuggingFaceEmbeddings()
    embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_model,  # 实际生成嵌入的工具
        store,  # 嵌入的缓存位置
        namespace=embeddings_model)
    return embedder


# define a method to return FAISS embedder
def get_faiss_embedder():
    embeddings_model = HuggingFaceEmbeddings()
    embedder = FAISS.from_documents(docs, embeddings_model)
    return embedder


if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    docs = [
        "您好，有什么需要帮忙的吗？",
        "哦，你好！昨天我订的花几天送达",
        "请您提供一些订单号？",
        "12345678",
    ]
    embeddings = embeddings_model.embed_documents(
        docs
    )
    print(f'{len(embeddings)}, {len(embeddings[0])}')

    embedded_query = embeddings_model.embed_query("刚才对话中的订单号是多少?")
    print(embedded_query[:3])

    # embedder = get_embedder()
    # embedder.embed_documents(docs)

    loader = TextLoader("./data/花语大全.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings_model)
    print(db.index.ntotal)

    query = "What are the words of the Hydrangea"
    docs = db.similarity_search(query)
    print(docs[0].page_content)

    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    print(docs[0].page_content)

    index = VectorstoreIndexCreator(vectorstore_cls=FAISS,
                                    embedding=HuggingFaceEmbeddings(),
                                    text_splitter=text_splitter)
    index = index.from_loaders([loader])
    # .from_loaders([loader])
    result = index.query(query)
    print(f'final result: {result}')
