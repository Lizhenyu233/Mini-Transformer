# 手写Mini-Transformer练习

---
### 项目简介
学习python和pytorch基本用法后，开始学习Transformer架构，参考教程进行从零手搓Transformer的练习，并且寻找中英文数据集进行训练和测试，最后附上训练损失变化图以及部分测试结果。
### 环境配置
- python（建议3.10以上）
- pytorch（建议最新）
- os
### 项目结构
```
/mini-transformer/
    ├── /train_result/              #训练结果
        ├── epoch_loss_curve.png    #损失曲线
        ├── training_losses.json    #损失数值保存
        ├── final_model.pth         #模型保存文件
        ├── part_of_test.png        #部分测试结果
    ├── config.py                   #参数配置
    ├── dataset.py                  #数据集构建
    ├── transformer.py              #Transformer模型
    ├── train.py                    #训练
    ├── test.py                     #测试 
    ├── eng-cmn.txt                 #中英文数据集
    ├── test_data.txt               #测试数据
    ├── test_result.txt             #测试结果
    ├── readme.md
```
### 使用说明
- 按照环境配置安装相应包
- 在config.py里设置训练参数
- 运行train.py文件进行训练，模型结果保存在train_result文件夹
- 运行test.py文件进行测试，测试对比结果在test_result.txt文件
---
#### Transformer模型（transformer.py）
- PositionalEncoding：使用交错sin方式实现位置编码
- get_attn_pad_mask & get_attn_subsequence_mask：实现对pad占位符的mask，以及decoder中防止self-attention看到后续答案的mask
- ScaledDotProductionAttention：实现attention分数的计算
- PositionwiseFeedForward ：前馈层和残差层的实现
- MultiHeadAttention ：多头注意力机制层，输入有Q,K,V以及对应的attn_mask，根据输入的不同实现self或cross注意力块
- EncoderLayer ：单个encoder块，包括一个self-attention块和一个前馈 & 残差层
- Encoder ：Encoder模块，包括原输入、位置编码和若干encoder块
- DecoderLayer：单个decoder块，包括一个self-attention块、一个cross-attention块和一个前馈 & 残差层
- Decoder：Decoder模块，包括目标输入、位置编码和若干decoder块
- Transformer ：整体框架，包括Encoder、Decoder和一个线性层
#### 数据集设置（dataset.py）
- 特殊符号定义：PAD占位符，SOS序列开始符，EOS序列结束符，UNK未知符（即词表里没有的字符）
- load_dataset ：读取从eng-cmn.txt读取数据集
- build_vocab ：根据读取的数据集建立词表 & word->index映射，并添加上述四个特殊符号
- sentence_to_index ：将数据集转化为index序列，并添加SOS符
- make_data ：将数据集转化为Tensor张量，并在每条数据空白处填充PAD符
- TranslationDataset ：数据集类，定义源输入、目标输入和目标输出，并且定义len以及getitem方法
#### 训练（train.py）
- 训练模型，记录loss损失并作图
- 保存训练后模型、每个epoch的loss值、损失曲线图
#### 参数配置（config.py）
- 训练中超参数配置，包括批大小、token维度、多头注意力head数等
- 文件读取和保存位置，包括训练集读取、模型保存、测试结果保存
#### 测试（test.py）
- greedy_decoder ：贪心策略生成预测，每一步选择概率最大的作为输出

---
### 实验
- 训练参数设置：
  - batch_size = 64
  - shuffle = True
  - epoch = 500
- 数据集构成：
  - 训练集：eng-cmn.txt前10000个中英文短句子组
  - 测试集：eng-cmn.txt第10001-10100条英文句子（包含相应中文句子）
  - [数据来源](https://github.com/clgm2015/TransformerProject.git)
- loss函数曲线

![loss函数曲线](/training_results/epoch_loss_curve.png)

  模型损失值：6.5807438497046e-06
- 部分测试结果  

![部分测试结果](/training_results/part_of_test.png)