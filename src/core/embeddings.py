"""
Docstring for rag_agent.src.core.embeddings

1,本版块目标：使用本地ollama的embedding模型接入parse_pdf.py文件parse的结果，进行embedding成一个数值向量，

2,这里面主要关注的是embedding功能，嵌入后的向量需要储存在chromedb之中
"""


from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from utils.parse_pdf import optimize_pdf_extraction
import os

def embed_pdf_chunks(pdf_path, embedding_model='qwen3-emb:4b', db_type='memory'):
    """
    读取PDF，分块后嵌入向量数据库，返回向量库对象。
    Args:
        pdf_path (str): PDF文件路径
        embedding_model (str): Ollama embedding模型名
        db_type (str): 'memory' 或 'chromadb'（如需切换）
    Returns:
        向量库对象
    """
    # 1. 解析PDF并获取分块
    result = optimize_pdf_extraction(pdf_path)
    char_chunks = result['char_chunks']
    # 2. 构造带元数据的Document对象
    docs = []
    for idx, chunk in enumerate(char_chunks):
        metadata = {
            'chunk_id': idx,
            'source': os.path.basename(pdf_path)
        }
        docs.append(Document(page_content=chunk, metadata=metadata))
    # 3. 嵌入
    embeddings = OllamaEmbeddings(model=embedding_model)
    if db_type == 'memory':
        vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    else:
        raise NotImplementedError('仅支持内存向量库，如需 chromadb 可扩展')
    return vectorstore

# ========== 示例用法 ==========
if __name__ == "__main__":
    pdf_path = "D:/浏览器下载/书/经济学/MWG拆分/MWG中文_287.pdf"
    vectorstore = embed_pdf_chunks(pdf_path)
    retriever = vectorstore.as_retriever() # 默认返回top-4
    query = "纳什均衡的定义"
    retrieved_documents = retriever.invoke(query)
    print(retrieved_documents[2].page_content)

