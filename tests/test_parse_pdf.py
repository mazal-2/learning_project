import os
from rag_agent.src.utils.parse_pdf import parse_pdf

def test_parse_pdf():
    # 使用一个存在的PDF路径进行测试，或用mock替代
    test_pdf = r"D:\浏览器下载\书\经济学\MWG拆分\MWG中文_287.pdf"
    if not os.path.exists(test_pdf):
        print("测试PDF文件不存在，跳过测试。")
        return
    chunks = parse_pdf(test_pdf, chunk_size=500, chunk_overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "page_content" in chunk
        assert "metadata" in chunk
    print(f"成功分割 {len(chunks)} 个chunk。示例: {chunks[0]}")

if __name__ == "__main__":
    test_parse_pdf()
