import dotenv
import os

from langchain_community.embeddings import IpexLLMBgeEmbeddings

dotenv.load_dotenv()

os.environ["SYCL_CACHE_PERSISTENT"] = "1"
os.environ["BIGDL_LLM_XMX_DISABLED"] = "1"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"


embedding_model = IpexLLMBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "xpu"},
    encode_kwargs={"normalize_embeddings": True},
)

sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
query = "What is IPEX-LLM?"

text_embeddings = embedding_model.embed_documents([sentence, query])
print(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
print(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

query_embedding = embedding_model.embed_query(query)
print(f"query_embedding[:10]: {query_embedding[:10]}")