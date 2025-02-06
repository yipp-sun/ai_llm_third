from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM

# 使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="/root/autodl-tmp/llm/Qwen/Qwen1.5-1.8B-Chat",
                     tokenizer_name="/root/autodl-tmp/llm/Qwen/Qwen1.5-1.8B-Chat",
                     model_kwargs={"trust_remote_code": True},
                     tokenizer_kwargs={"trust_remote_code": True})
# 调用模型chat引擎得到回复
rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
# rsp = llm.chat(messages=[ChatMessage(content="xtuner是哪家公司的？")])

print(rsp)
