import ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="qwen3:8b")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
res = chain.invoke({'question':'如何在langchain里面调用ollama的模型？'})
print(res)