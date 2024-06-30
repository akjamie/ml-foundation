# 设置OpenAI API密钥
import logging
import os
from pprint import pprint
from typing import List, TypedDict

import dotenv
from langchain.globals import set_debug
from langchain_community.chat_models import ChatOllama, ChatTongyi
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import IpexLLMBgeEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import END
from langgraph.graph import StateGraph

from langchain_google_genai import ChatGoogleGenerativeAI

# from common.annotations import proxy

dotenv.load_dotenv()
set_debug(True)


# 导入所需的库和模块
class ChatbotWithRetrieval:
    def __init__(self, dir: str):
        self.grader = None
        self.base_dir = dir
        self.documents = self.load_documents_from_dir(self.base_dir)
        if not self.documents:
            logging.warning("No compatible documents found in the directory.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(self.documents)

        embeddings = IpexLLMBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "xpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = FAISS.from_documents(all_splits, embeddings)
        # self.llm = ChatTongyi(dashscope_api_key=self.get_api_key())
        self.llm = ChatOllama(model="qwen:14b", format="json", temperature=0)
        # self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

        self.retriever = self.vectorstore.as_retriever()
        self.conversation_history = ""
        template = """
                   You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                   Question: {question}
                   Context: {context}
                   Answer:
               """
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        self.qa = ({
                       "context": self.retriever.with_config(run_name="Docs"),
                       "question": RunnablePassthrough()
                   }
                   | self.prompt
                   | self.llm
                   | StrOutputParser()
                   )

        self.chain = (self.prompt
                      | self.llm
                      | StrOutputParser())

        # template = """<|begin_of_text|><|start_header_id|>system<end_header_id|>
        #            You are a grader assessing relevance
        #            of a retrieved document to a user question. If the document contains keywords related to the user question,
        #            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous
        #            retrievals. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the
        #            question. \n
        #            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        #            <|eot_id|><|start_header_id|>user<|end_header_id>
        #            Here is the retrieved document: \n\n {document} \n\n
        #            Here is the user question: {question} \n
        #            <|eot_id|><|start_header_id|>assistant<|end_header_id>
        #            """
        # template = """
        #                You are a grader assessing relevance
        #                of a retrieved document to a user question. If the document contains keywords related to the user question,
        #                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous
        #                retrievals. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the
        #                question. \n
        #                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        #
        #                Here is the retrieved document: \n\n {document} \n\n
        #                Here is the user question: {question} \n
        #            """
        template = """ 
                    You are a grader assessing the relevance of a retrieved document to a user question. Consider not only the presence of
                    keywords but also their context, frequency, and overall semantic connection to the question. A document should be graded
                    as 'relevant' only if it provides meaningful information that directly or indirectly addresses the question. 
                     
                    Use a scale from 1 (not relevant) to 5 (highly relevant), and provide your judgment as a JSON object with a single key 'score'. 
                    Avoid assigning high scores unless the document is clearly pertinent.
                   
                   Here is the retrieved document: \n\n {document} \n\n
                   Here is the user question: {question} \n 
           """

        prompt = PromptTemplate(template=template, input_variables=["document", "question"], )

        self.grader = prompt | self.llm | JsonOutputParser()

        self.web_search_tool = TavilySearchResults(k=3)

    def grade(self, question: str, document: str) -> str:
        return self.grader.invoke({"document": document, "question": question})

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

    def retrieve(self, state):
        """
        Retrieve documents from vectorstore.
        Args:
            state (dict): the current graph state
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents.
        """

        question = state["question"]
        documents = self.retriever.get_relevant_documents(question)

        return {
            "documents": documents,
            "question": question
        }

    # @proxy("127.0.0.1:7890")
    def generate(self, state):
        """
        Generate response using RAG on retrieved documents.
        Args:
            state (dict): the current graph state
        Returns:
            state (dict): New key added to state, generation, that contains LLM generation.
        """
        question = state["question"]
        documents = state["documents"]

        generated_result = self.chain.invoke({"question": question, "context": documents[0].page_content})
        return {
            "generation": generated_result,
            "question": question,
            "documents": documents
        }

    def grade_documents(self, state):
        """
        Grade retrieved documents, determine whether the retrieved documents are relevant to the question if any
        document is not relevant, we will set a flat to run web search.

        Args:
            state (dict): the current graph state
        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state.
        """
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"
        for doc in documents:
            score = self.grade(question=question, document=doc.page_content)
            grade = score["score"]
            # if grade.lower() == "yes":
            #     print(f"-- Grade document, result is relevant, question:{question}, document:{doc}")
            #     filtered_docs.append(doc)
            # else:
            #     print(f"-- Grade document, result is not relevant, question:{question}, document:{doc}")
            #     web_search = "Yes"
            #     continue
            if grade >= 3:
                print(f"-- Grade document, result is relevant, question:{question}, document:{doc}")
                filtered_docs.append(doc)
            else:
                print(f"-- Grade document, result is not relevant, question:{question}, document:{doc}")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state):
        """
        Determines whether to generate an answer, or add web search.

        Args:
            state (dict): the current graph state
        Returns:
            state (dict): Binary decision for next node.
        """
        question = state["question"]
        documents = state["documents"]

        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d['content'] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]

        return {"documents": documents, "question": question}

    def decide_to_generate(self, state):
        """
        Determine whether to generate an answer, or add web search.

        Args:
            state (dict): the current graph state
        Returns:
            state (dict): Binary decision for next node.
        """
        question = state["question"]
        web_search = state["web_search"]
        filtered_docs = state["documents"]

        if web_search == "Yes":
            return "websearch"
        else:
            return "generate"

    def orchestrate(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {
            "websearch": "websearch",
            "generate": "generate"
        }, )

        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)

        return workflow


class GraphState(TypedDict):
    """
    Present the state of graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]


if __name__ == "__main__":
    os.environ["SYCL_CACHE_PERSISTENT"] = "1"
    os.environ["BIGDL_LLM_XMX_DISABLED"] = "1"
    os.environ["https_proxy"] = "127.0.0.1:7890"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    bot = ChatbotWithRetrieval("docs")

    app = bot.orchestrate().compile()
    question = '易速鲜花的企业精神是什么?'

    print(f'{bot.llm.invoke("Where is the capital of China?")}')

    for output in app.stream({"question": question}):
        for key, value in output.items():
            pprint(f'Finished running: {key}:')
    pprint(value["generation"])
