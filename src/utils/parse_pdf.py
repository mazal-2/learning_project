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
    D:\\浏览器下载\\书\\经济学\\MWG拆分\\MWG中文_287.pdf

    
需要对parse后的内容进行更加精确的清洗：
    读取后要明确分词规则以及进行内容进行清洗
    优化分隔符优先级：优先按「章节标题、段落、题号、句号」分割，避免拆分完整题目 / 知识点。
"""
"""

如何优化？ 
一边是读取，另一边是分块；读取后可以尝试格式转化？
分块方式可以尝试递归分块，即先用段落或者句子等标号进行分块（两个换行符或者其他的）

在分块过程如何做好原数据的索引？
元数据应该包括 章节的页码 章节名称 内容类型 以及自身的id号

1,langchain的Document对象：

    from langchain.schema import Document
    docs = []
    for page_num, page_content in pdf_reader.pages:
        # 创建带元数据的 Document 对象
        metadata = {
            "page": page_num + 1,
            "chapter": "当前检测到的章节名",
            "source": "economics_textbook.pdf"
        }
        new_doc = Document(page_content=page_content, metadata=metadata)
        docs.append(new_doc)

2,
"""



import re
import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
def optimize_pdf_extraction(pdf_path, output_txt_path="extracted_optimized.txt",chunk_size:int=1000,chunk_overlap:int=150):
    """
    优化PDF文本提取过程，解决重复字符、换行混乱、文本断裂等问题
    :param pdf_path: PDF文件路径
    :param output_txt_path: 优化后的文本输出路径
    :return: 优化后的完整文本
    """
    # 1. 读取PDF原始文本
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")
    
    # 可选：用 pdfplumber 替代 pypdf
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            raw_text += page_text + "\n"

    # 2. 核心优化：文本清洗规则（针对你的文本特征定制）
    optimized_text = raw_text
    
    # 2.1 去除重复字符（连续3次及以上的相同字符，保留1个）
    # 匹配"相同的相同的相同的" "纯化定理纯化定理"等模式
    optimized_text = re.sub(r"([^\n\s])\1{2,}", r"\1", optimized_text)
    # 匹配带标点的重复（如"。。。" "：：：："）
    optimized_text = re.sub(r"([\u3002\uff1a\uff1b\uff0c\uff1f\uff01])\1{1,}", r"\1", optimized_text)
    
    # 2.2 修复跨行断句（如"完全信 息博弈"→"完全信息博弈"）
    optimized_text = re.sub(r"(\S)\s*\n\s*(\S)", r"\1\2", optimized_text)
    
    # 2.3 清理无意义换行和空行
    # 保留段落级换行（连续2个\n），合并单行换行
    optimized_text = re.sub(r"(?<!\n)\n(?!\n)", " ", optimized_text)
    # 合并连续空行为单个空行
    optimized_text = re.sub(r"\n{3,}", "\n\n", optimized_text)
    
    # 2.4 过滤冗余信息
    # 过滤页码（如 297  、300  ）
    optimized_text = re.sub(r"\n\d{3}\s+", "\n", optimized_text)
    # 过滤译者标注（曹乾（东南大学...））
    optimized_text = re.sub(r"曹乾（东南大学.*?@163\.com\s*）", "", optimized_text)
    # 过滤无关标注（（九）（十）（十一））
    optimized_text = re.sub(r"\n\s*\（[十十一九]{1,2}\）\s*\n", "\n", optimized_text)
    # 过滤重复的章标题（如"第第第第 9 章章章章"）
    optimized_text = re.sub(r"([第章])\1{3,}", r"\1", optimized_text)
    
    # 2.5 修复公式符号显示（基础修复，复杂公式需结合LaTeX）
    # 修复下标（如 iS → S_i）
    optimized_text = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", optimized_text)
    # 修复博弈符号（如 [ ,{ ( )},{ ( )}] → [Γ_N, {Δ(S_i)}, {u_i(·)}]）
    optimized_text = re.sub(r"\[ ,\{ <span data-type=\"inline-math\" data-value=\"\"></span> \},\{<span data-type=\"inline-math\" data-value=\"\"></span>\} \]", r"[Γ_N, \{Δ(S_i)\}, \{u_i(·)\}]", optimized_text)
    
    # 2.6 去除首尾空格和多余换行
    optimized_text = optimized_text.strip()

    # 3. 保存优化后的文本
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(optimized_text)
    
    # 4. 文本分块优化（解决chunk_size过小问题）
    # 按"8.A.1""8.B.2"等题号分块，保证每个知识点完整
    chunks = re.split(r"(?=\n8\.[A-Z]\.\d+)", optimized_text)
    # 过滤空块
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # 5. 对优化后的全文直接按字符分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=["\\n\\n", "\\n", "。", "！",'!',"?", "？", " ", ""])
    char_chunks = text_splitter.split_text(optimized_text) # 使用langchain里面recursive，在这里面带有元数据么？

    return {
        "full_optimized_text": optimized_text,
        "structured_chunks": chunks,  # 按题号结构化的文本块
        "char_chunks": char_chunks    # 按字符分块的结果
    }

"""
# ===================== 运行示例 =====================
if __name__ == "__main__":
    # 替换为你的PDF路径
    pdf_path = "D:/浏览器下载/书/经济学/MWG拆分/MWG中文_287.pdf"
    
    # 执行优化提取
    result = optimize_pdf_extraction(pdf_path)
    
    # 打印结果预览
    print("=== 优化后的完整文本预览（前500字符）===")
    print(result["full_optimized_text"][:500])
    print(f"results的数量为： {len(result["char_chunks"])}") # 分成55个板块
    print("\\n=== 结构化分块结果（前3个块）===")
    for i, chunk in enumerate(result["char_chunks"][:3]):
        print(f"\\n【块 {i+1}】")
        print(chunk[:500] + "..." if len(chunk) > 200 else chunk)
        """