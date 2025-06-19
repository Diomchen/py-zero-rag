import os
from dotenv import load_dotenv
# 加载环境
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 加载 pdf 文件
pdf_loader = PyPDFLoader(os.getenv("PDF_PATH"))
documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)
print(f"Chunks: {len(chunks)}")

# 使用本地的 bge embedding模型
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU设备名称: {torch.cuda.get_device_name()}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-zh-v1.5",
    model_kwargs = {"device": device},  # 使用GPU如果可用
    encode_kwargs = {"normalize_embeddings": True},
    show_progress=True,
)
print(f"Embeddings: {embeddings}")

# 创建向量存储
vector_store = FAISS.from_documents(
    documents=chunks,  # 使用所有文档
    embedding=embeddings
)
print(f"Vector store: {vector_store}")

# 检索器
retriver = vector_store.as_retriever(search_kwargs={"k":3})
print(f"Retriver: {retriver}")

# 创建提示模板
template = """你是一个问答助手。
请使用下面提供的上下文信息来回答问题。
如果你不知道答案，就直接说不知道，不要尝试编造答案。
最多使用三个句子，保持回答简洁。

问题: {question}
上下文: {context}
回答:
"""

prompt = ChatPromptTemplate.from_template(template)
print(prompt)

# 配置 llm
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),  # 使用base_url而不是openai_base
    temperature=0.7,
)

# 构建 RAG 链
rag_chain = (
    {"context":retriver, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 回答问题
query = "我想了解 ERC-721 协议"
resp = rag_chain.invoke(query)
print(resp)