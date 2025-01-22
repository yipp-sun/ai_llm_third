# 字典自定义操作
from transformers import BertTokenizer

token = BertTokenizer.from_pretrained(
    r"D:\Workspace\llm\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

# 编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=["阳光洒在大地上"],
    add_special_tokens=True,
    truncation=True,
    padding="max_length",
    max_length=20,
    return_length=None
)

print(token.decode(out["input_ids"][0]))

# 获取字典
vocab = token.vocab
print(vocab)
print(len(vocab))
print('阳' in vocab)
print('光' in vocab)
print('阳光' in vocab)

# 自定义内容添加到vocab中
token.add_tokens(new_tokens=["阳光"])
# 重新获取vocab对象
vocab = token.get_vocab()
# print(vocab)
print(len(vocab))
print('阳光' in vocab)

# 编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=["阳光洒在大地上"],
    add_special_tokens=True,
    truncation=True,
    padding="max_length",
    max_length=20,
    return_length=None
)

print(token.decode(out["input_ids"][0]))
print(out["input_ids"])
