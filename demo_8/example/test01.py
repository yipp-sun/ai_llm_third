# 中文白话文文章生成
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline

# 加载模型和分词器，该模型调用情况下AutoTokenizer, AutoModel不可用
model = GPT2LMHeadModel.from_pretrained(
    r"D:\Workspace\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
tokenizer = BertTokenizer.from_pretrained(
    r"D:\Workspace\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
print(model)

# 使用Pipeline调用模型
text_generator = TextGenerationPipeline(model, tokenizer, device="cuda")

# 使用text_generator生成文本
# do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时(概率最大的)，每次生成的结果都是相同的。
# print(text_generator("这是很久之前的事情了,", max_length=100, do_sample=True))
for i in range(3):
    print(text_generator("这是很久之前的事情了,", max_length=100, do_sample=True))
