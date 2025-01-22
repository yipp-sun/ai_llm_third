from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(
    r"D:\Workspace\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(
    r"D:\Workspace\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# 加载我们自己训练的权重（中文诗词）
model.load_state_dict(torch.load("net.pt"))

# 使用系统自带的pipeline工具生成内容
pipline = TextGenerationPipeline(model, tokenizer, device=0)

print(pipline("天高", max_length=24))
