from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OllamaEmbeddings(model='qwen3-emb:4b')

text = "LangChain is the framework for building context-aware reasoning applications"
vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()
retrieved_documents = retriever.invoke("what is langchain")

print(retrieved_documents[0].page_content)

