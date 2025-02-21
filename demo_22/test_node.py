from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

# 初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
    # 指定了一个预训练的sentence-transformer模型的路径
    model_name="D:/Workspace/llm/model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 将创建的嵌入模型赋值给全局设置的embed_model属性，这样在后续的索引构建过程中，就会使用这个模型
Settings.embed_model = embed_model

# 使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
                     tokenizer_name="D:/Workspace/llm/model/Qwen/Qwen1___5-1___8B-Chat",
                     model_kwargs={"trust_remote_code": True},
                     tokenizer_kwargs={"trust_remote_code": True})
# 设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.llm = llm

# 读取CSV文件
# df = pd.read_csv("data_node/node_test.csv")
# 读取 Excel 文件
df = pd.read_excel("data_node/node_test.xlsx", sheet_name="Sheet1", engine="openpyxl")

# 转换为节点
nodes = []
for _, row in df.iterrows():
    text = f"Title: {row['title']}\nContent: {row['content']}"
    metadata = {"id": row['id'], "title": row['title']}
    node = TextNode(text=text, metadata=metadata)
    nodes.append(node)
    # print(row)
    # id                     1
    # title       只剩一个心脏了还能活吗？
    # content    能，人本来就只有一个心脏。
    # Name: 0, dtype: object
    # print(text)
    # Title: 只剩一个心脏了还能活吗？
    # Content: 能，人本来就只有一个心脏。
    # print(metadata)
    # {'id': 1, 'title': '只剩一个心脏了还能活吗？'}
    # print(node)
    # Node ID: 9a3418d7-de0c-4cc6-884c-8a1b17294764
    # Text: Title: 只剩一个心脏了还能活吗？ Content: 能，人本来就只有一个心脏。
# print(nodes)
# 构建索引
index = VectorStoreIndex(nodes)

# 将索引持久化存储到本地的向量数据库
index.storage_context.persist()

# 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。
query_engine = index.as_query_engine()
rsp = query_engine.query("爸爸再婚，我是不是就有了个新娘？")
print(rsp)
