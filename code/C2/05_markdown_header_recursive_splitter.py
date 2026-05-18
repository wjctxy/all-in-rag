from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

markdown_text = """
# RAG 系统设计

## 检索模块

检索模块负责从向量数据库中召回候选片段。为了提升召回效果，需要统一 embedding 模型、分块策略和索引参数。
当候选片段较多时，还会使用重排序模型做二次筛选。

## 生成模块

生成模块负责将问题与检索结果组合成提示词，并调用大语言模型生成答案。
提示词模板通常包含角色设定、约束条件、引用来源和输出格式。
"""

# 第一步：按标题做结构化分块，得到带标题元数据的大块
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)
structured_docs = markdown_splitter.split_text(markdown_text)

# 第二步：对第一步的大块继续做递归字符分块，控制 chunk_size
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
final_chunks = recursive_splitter.split_documents(structured_docs)

print(f"第一阶段（按标题）块数: {len(structured_docs)}")
print(f"第二阶段（递归字符）块数: {len(final_chunks)}")
print("--- 前5个最终块（含继承元数据） ---")
for i, chunk in enumerate(final_chunks[:5], 1):
    print("=" * 60)
    print(f"块{i} 长度: {len(chunk.page_content)}")
    print(f"块{i} 元数据: {chunk.metadata}")
    print(f"块{i} 内容: {chunk.page_content}")
