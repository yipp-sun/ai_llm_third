from transformers import BertModel, BertConfig
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
# pretrained = BertModel.from_pretrained(
#     r"D:\Workspace\llm\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(
#     DEVICE)
#
# pretrained.embeddings.position_embeddings = torch.nn.Embedding(1024, 768).to(DEVICE)
config = BertConfig.from_pretrained(
    r"D:\Workspace\llm\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
config.max_position_embeddings = 1024
print(config)

# 使用配置文件初始化模型
pretrained = BertModel(config).to(DEVICE)
print(pretrained)


# 定义下游任务
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 10)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结预训练模型权重
        # with torch.no_grad():
        #     out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 模型发生了改动，前面的先验可能无效了，需要模型参与训练，全量微调（因为有先验，所以还是微调）
        out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 增量模型参与训练
        out = self.fc(out.last_hidden_state[:, 0])
        # 多分类，对out激活，可以不加
        # out = out.softmax(dim=1)
        return out
