from torch import optim
from config import *
from transformer import *
import os

import matplotlib.pyplot as plt
import json

# 创建保存结果的目录
os.makedirs(results_dir, exist_ok=True)
# 记录训练过程的变量
epoch_losses = []  # 每个epoch的平均损失

# 初始化模型
model = Transformer().cuda()
model.train()
# 损失函数,忽略为0的类别不对其计算loss（因为是padding无意义）
criterion = nn.CrossEntropyLoss(ignore_index=0)
# 优化器设置
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 训练开始
for epoch in range(n_epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        # 放到GPU里
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        outputs = model(enc_inputs, dec_inputs)
        # 计算损失函数
        loss = criterion(outputs, dec_outputs.view(-1))  # 将dec_outputs展平成一维张量

        # 记录损失
        epoch_loss = 0 #记录当前epoch的所有batch损失
        current_loss = loss.item()
        epoch_loss += current_loss

        # 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 取平均的epoch损失
    avg_epoch_loss = epoch_loss / len(loader)
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.10f}")


# 训练结束后保存模型
torch.save(model, os.path.join(results_dir, 'final_model.pth'))
print("训练完成！所有结果保存在:", results_dir)
# 保存损失数据到文件
with open(os.path.join(results_dir, 'training_losses.json'), 'w') as f:
        json.dump({
            'epoch_losses': epoch_losses
        }, f)
# 读取每个epoch损失
with open(os.path.join(results_dir, 'training_losses.json'), 'r') as f:
        loss_data = json.load(f)
    
    
# 绘制Epoch损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_data['epoch_losses'], 'o-')
plt.xlabel('Epoch')
plt.ylabel('Avg Loss')
plt.title('Epoch Average Loss')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'epoch_loss_curve.png'))
plt.close()