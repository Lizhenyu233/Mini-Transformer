import torch
from config import *
from transformer import *
import json

# greedy贪心策略生成预测，每一步选择概率最大的作为输出
def greedy_decoder(model, enc_input, start_symbol):
    # 编码器输出
    enc_outputs = model.encoder(enc_input)
    # 生成一个1行0列的，和enc_inputs.data类型相同的空张量，待后续填充
    dec_input = torch.zeros(1, 0).type_as(enc_input.data) # .data避免影响梯度信息
    next_symbol = start_symbol #初始化第一个dec_input
    flag = True
    while flag:
        # 将next_symbol拼接到dec_input中，作为新一轮decoder的输入
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], -1)
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        # 返回概率最大所对应的索引
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # prob包含目前为止依次生成的词的索引，最后一个是新生成的
        next_symbol = prob.data[-1]
        # 如果是结束符EOS，则停止预测
        if next_symbol == 3 :
            flag = False
        # print(next_symbol)
    return dec_input 

# 测试
model = torch.load(load_model_file_path, weights_only=False) # weights_only=False保证能读取
model.eval()

predictions = []    #保存预测字符串
test_data = []
real_data = []
#读取测试集
with open(test_data_file, 'r', encoding='utf-8') as f:
        # 读取每行数据
        for i, line in enumerate(f):
            if test_data_nums and i >= test_data_nums:
                break
            parts = line.strip().split('\t') #中英文之间用'\t'隔开
            if len(parts) >= 2:
                test_data.append(parts[0].lower())  # 英文转为小写
                real_data.append(parts[1])
#转化为Tensor张量
enc_inputs = []
for i in range(len(test_data)):
    enc_input = sentence_to_index(test_data[i], en_word2idx)
    enc_input = pad_sequence(enc_input, src_max_len, en_word2idx[PAD])
    enc_inputs.append(enc_input)
enc_inputs = torch.LongTensor(enc_inputs)



with torch.no_grad():   #在评估模型性能时，不需要计算梯度。使用 torch.no_grad() 可以提高评估速度和减少内存消耗
    enc_inputs = enc_inputs.cuda()
    for i in range(len(enc_inputs)):
        # 使用贪心算法，开始字符SOS对应的索引为2
        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=2)
        predict = greedy_dec_input.view(-1) #要展平成一维向量，不然报错！
        print([src_idx2word[n.item()] for n in enc_inputs[i]], '->', [tgt_idx2word[n.item()] for n in predict])
        # 预测字符
        prediction = [tgt_idx2word[n.item()] for n in predict]
        # 移除特殊符号
        pre_filtered_tokens = [token for token in prediction if token not in [SOS, EOS, PAD]]
        prediction_str = ''.join(pre_filtered_tokens)  # 中文直接连接
        # 真实字符
        real_str = real_data[i]
        # 输入字符
        enc_str = test_data[i]
        # 保存输入-预测-目标三元组
        predictions.append({
                    "input": enc_str,
                    "prediction": prediction_str,
                    "target": real_str
                })

#保存文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("预测结果\t真实结果\n")
    f.write("=" * 50 + "\n")
        
    for i, pred in enumerate(predictions, 1):
        f.write(f"样本 {i}:{pred['input']}\n")
        f.write(f"预测: {pred['prediction']}\n")
        f.write(f"真实: {pred['target']}\n")
        f.write("-" * 50 + "\n")
    
print(f"成功保存 {len(predictions)} 条预测结果到 {output_file}")
