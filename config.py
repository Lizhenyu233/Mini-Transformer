from datetime import datetime
import os
# dataset
data_file_path = "eng-cmn.txt"  #训练数据位置
num_lines = 10000    #训练集数量
batch_size = 64 #批大小
shuffle = True  #打乱
# train
# 模型保存位置
results_dir = f"training_results"
n_epochs = 500    #训练轮数

# transformer
# 用来表示一个词的向量长度
d_model = 512
# FFN的隐藏层神经元个数
d_ff = 2048
# 分头后的q、k、v词向量长度，q和k相等
d_k = d_v = 64
# Encoder Layer 和 Decoder Layer的个数
n_layers = 6
# 多头注意力中head的个数
n_heads = 8

# test
load_model_file_path = os.path.join('training_results', 'final_model.pth')
# 测试保存位置
output_file = 'test_result.txt'
test_data_file = "test_data.txt"
test_data_nums = 100 #测试数据读取量