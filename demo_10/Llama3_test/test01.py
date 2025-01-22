# 模型下载
from modelscope import snapshot_download

# model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct',cache_dir="/teacher_data/zhangyang/llm/")
# model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', cache_dir="/root/autodl-tmp/llm/")
# model_dir = snapshot_download('UnicomAI/Unichat-llama3.2-Chinese-1B', cache_dir="/root/autodl-tmp/llm/")
# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen1.5-1.8B-Chat', cache_dir="/root/autodl-tmp/llm/")
