# 模型训练
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer, AdamW

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练的轮次
EPOCH = 30000

token = BertTokenizer.from_pretrained(
    r"D:\Workspace\AIProject\demo_5\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")


def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    # Model的输入，参照net.py
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    # 根据显存设定，显存大，就设定大些
    batch_size=100,
    # 把数据集打乱，洗牌
    shuffle=True,
    # 舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    # 对加载进来的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    # 开始训练（炼丹开始）
    print(DEVICE)
    model = Model().to(DEVICE)
    # 定义优化器
    optimizer = AdamW(model.parameters())
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将数据存放到DEVICE上
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            # 前向计算（将数据输入模型，得到输出）
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # 根据输出，计算损失
            loss = loss_func(out, labels)
            # 根据损失，优化参数，梯度是变量，不能覆盖，只能更新，删除和追加
            # 固定模式：三步走；清空梯度（梯度归0）
            optimizer.zero_grad()
            # 根据loss.backward()，计算新的梯度
            loss.backward()
            # 更新梯度
            optimizer.step()

            # 每隔5个批次输出训练信息
            if i % 5 == 0:
                # 模型输出的标签，转化为整数
                out = out.argmax(dim=1)
                # 精度，pytorch，true为1，False为0
                acc = (out == labels).sum().item() / len(labels)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        # 每训练完一轮，保存一次参数
        torch.save(model.state_dict(), f"params/{epoch}_bert.pth")
        # save后print()必须有，防止训练中断
        print(epoch, "参数保存成功！")
