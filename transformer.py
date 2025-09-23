from dataset import *
import torch
import numpy as np
import torch.nn as nn
from config import *

class PositionalEncoding(nn.Module):
    '''位置编码'''
    def __init__(self, d_model, dropout=0.1, max_len=5000): 
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        # 先把括号内的分式求出来再取正余弦
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 增加一维来用和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0)
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # 添加位置编码
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''将输入序列中的占位符P的token给mask掉'''
    # 获取q，k的序列长度
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 返回一个和seq_k等大的布尔张量，seq_k元素等于0的位置为True,否则为False
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) #扩维以保证维度一致
    # 为每一个q提供一份k，把第二维度扩展了q次
    res = pad_attn_mask.expand(batch_size, len_q, len_k)
    return res

def get_attn_subsequence_mask(seq):
    """用于获取对后续位置的掩码，防止在预测过程中看到未来时刻的输入"""
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 生成一个上三角矩阵
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte() # 因为只有0、1所以用byte节省内存
    return subsequence_mask

class ScaledDotProductionAttention(nn.Module):
    '''自注意力机制的计算'''
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #计算注意力分数QK^T/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        #进行mask和softmax
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        #乘V得到最终的加权和
        context = torch.matmul(attn, V)
        return context 

class PositionwiseFeedForward(nn.Module):
    '''Feed Forward和 Add & Norm'''
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)


class MultiHeadAttention(nn.Module):
    '''多头注意力机制'''
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        #提取输入用于残差链接、提取batch_size
        residual, batch_size = input_Q, input_Q.size(0)

        #进行多头处理，并且经过线性层
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        #计算注意力
        #自我复制n_heads次，为每个头准备一份mask
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context = ScaledDotProductionAttention()(Q, K, V, attn_mask)

        #concat部分，拼接起来并经过线性层
        context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
        output = self.concat(context)
        return nn.LayerNorm(d_model).cuda()(output + residual)

class EncoderLayer(nn.Module):
    '''单个encoder模块'''
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # Q、K、V均为 enc_inputs
        enc_ouputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_ouputs = self.pos_ffn(enc_ouputs)
        return enc_ouputs

class Encoder(nn.Module):
    '''编码模块'''
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)    #词嵌入
        self.pos_emb = PositionalEncoding(d_model)  #添加位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  #n个encoder模块

    def forward(self, enc_inputs):
        # 词嵌入和位置编码
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        # mask占位符
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) 
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs  

class DecoderLayer(nn.Module):
    '''单个decoder层'''
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        #带掩码的自注意力
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        #混合注意力
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs

class Decoder(nn.Module):
    '''decoder模块'''
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])


    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs).cuda()
        # mask占位符以及mask未来词防止提前看答案
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        # 将两个mask叠加，和大于0的位置是需要被mask掉的，赋为True，和为0的位置是有意义的为False
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).cuda()
        # mask掉占位符，防止enc中的占位符产生影响
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

        return dec_outputs

class Transformer(nn.Module):
    '''整体框架'''
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size).cuda()

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)

        # 解散batch，让他们按行依次排布
        return dec_logits.view(-1, dec_logits.size(-1))
  


