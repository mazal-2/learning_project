"""
Docstring for rag_agent.src.utils.parse_pdf

1, 功能： 读取获取教材，分本拆分
2，所调用的api

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecersiveCharacterTextSplitter
    # 加载PDF文件
    file_path = "你的PDF文件路径.pdf"
    loader = PyPDFLoader(file_path)
    
    整理loader处理的pdf文件成列表，并在里每一页补上metadata加上page_content两个参数
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=, chunk_overlap=)
    texts = text_splitter.split_text(document)


3，初步目标：
    能够对读取的pdf课本教材进行读取并且按照一点长度进行chunk以及设定一个合理的chunk_overlap
    这里面的chunksize大小的后面需要存入chromedb进行检索，检索出来的内容会被作为一个上下文文本注入到ai的系统提示中
    chunksize以及里面overlap需要控制到既有精度又能够包含足够的信息让agent能拿到足够的信息来回答问题

高级微观经济学教材节选路径
    D:\浏览器下载\书\经济学\MWG拆分\MWG中文_287.pdf

"""


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def parse_pdf(
    file_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    add_metadata: bool = True
) -> list:
    """
    读取 PDF 文件，按 chunk 拆分文本，并补充 metadata。
    Args:
        file_path (str): PDF 文件路径
        chunk_size (int): 每个 chunk 的最大长度
        chunk_overlap (int): chunk 间重叠长度
        add_metadata (bool): 是否为每个 chunk 添加 metadata
    Returns:
        List[dict]: 每个 chunk 一个 dict，包含 page_content 和 metadata
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    results = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    for page in pages:
        # page.page_content: str, page.metadata: dict
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            item = {
                "page_content": chunk
            }
            if add_metadata:
                meta = dict(page.metadata) if hasattr(page, 'metadata') else {}
                item["metadata"] = meta
            results.append(item)
    return results

mwg = r"D:/浏览器下载/书/经济学/MWG拆分/MWG中文_287.pdf"
results = parse_pdf(file_path=mwg,chunk_size=300,chunk_overlap=50,add_metadata=True)

for r in results:
    print("运行成功，现输出结果：")
    print(r)


