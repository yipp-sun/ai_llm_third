from transformers import AdamW
import torch
from data import MyDataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from tensorboardX import SummaryWriter

# 加载数据集
dataset = MyDataset()

# 加载编码器
tokenizer = AutoTokenizer.from_pretrained(
    'D:\Workspace\llm\model\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3')
# 加载模型
# /root/app/huggingface/LLM/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3/
model = AutoModelForCausalLM.from_pretrained(
    'D:\Workspace\llm\model\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3')


# 数据预处理函数
def collate_fn(data):
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,
                                       truncation=True,
                                       max_length=512,
                                       return_tensors='pt')

    data['labels'] = data['input_ids'].clone()

    return data


# 数据加载器
loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=13,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)

# 创建TensorBoard的SummaryWriter实例
# writer = SummaryWriter("/root/app/projects/day17_demo/logdir/")
writer = SummaryWriter("logdir/")


# 训练函数
def train():
    global model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    torch.cuda.empty_cache()  # 在训练开始前清理缓存内存
    for epoch in range(1000):
        for i, data in enumerate(loader):
            for k in data.keys():
                data[k] = data[k].to(device)
            out = model(**data)
            loss = out['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                with torch.no_grad():
                    labels = data['labels'][:, 1:]
                    out = out['logits'].argmax(dim=2)[:, :-1]
                    select = labels != 0
                    labels = labels[select]
                    out = out[select]
                    accuracy = (labels == out).sum().item() / labels.numel()
                    del labels, out  # 删除变量以释放内存

                lr = optimizer.state_dict()['param_groups'][0]['lr']

                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}, LR: {lr}, Accuracy: {accuracy}")

                # 将指标写入TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(loader) + i)
                writer.add_scalar('Accuracy/train', accuracy, epoch * len(loader) + i)
                writer.add_scalar('Learning Rate', lr, epoch * len(loader) + i)

        # 保存模型参数，未保存模型结构
        torch.save(model.state_dict(), 'net.pt')
        print(f"Epoch {epoch} - 权重保存成功！")

        torch.cuda.empty_cache()  # 清除未使用的缓存内存


# 主程序入口
if __name__ == '__main__':
    train()

# 关闭SummaryWriter
writer.close()
