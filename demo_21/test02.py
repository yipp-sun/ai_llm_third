from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM

# 初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
    # 指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/autodl-tmp/llm/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 将创建的嵌入模型赋值给全局设置的embed_model属性，这样在后续的索引构建过程中，就会使用这个模型
Settings.embed_model = embed_model

# 使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="/root/autodl-tmp/llm/Qwen/Qwen1.5-1.8B-Chat",
                     tokenizer_name="/root/autodl-tmp/llm/Qwen/Qwen1.5-1.8B-Chat",
                     model_kwargs={"trust_remote_code": True},
                     tokenizer_kwargs={"trust_remote_code": True})
# 设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.llm = llm

# 从指定目录读取文档，将数据加载到内存
documents = SimpleDirectoryReader("/root/autodl-tmp/demo_21/data").load_data()
print(documents)
# 创建一个VectorStoreIndex,并使用之前加载的文档来构建向量索引，这里调用了embed_model
# 此索引将文档转换为向量，并存储这些向量（内存）以便于快速检索
index = VectorStoreIndex.from_documents(documents)
# 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。该方法调用上记的llm
query_engine = index.as_query_engine()
rsp = query_engine.query("xtuner是什么？")
# rsp = query_engine.query("xtuner是哪家公司的？")

print(rsp)
