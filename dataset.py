import torch
import torch.utils.data as Data
from collections import Counter
from config import *


# 特殊符号定义
PAD = '<pad>'  # 填充符
SOS = '<sos>'  # 序列开始符
EOS = '<eos>'  # 序列结束符
UNK = '<unk>'  # 未知词

# 读取中英文对照数据集
def load_dataset(file_path, num_lines = None):
    en_sentences = []
    cn_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取每行数据
        for i, line in enumerate(f):
            if num_lines and i >= num_lines:
                break
            parts = line.strip().split('\t') #中英文之间用'\t'隔开
            if len(parts) >= 2:
                en_sentences.append(parts[0].lower())  # 英文转为小写
                cn_sentences.append(parts[1])
    return en_sentences, cn_sentences

# 构建词汇表
def build_vocab(sentences, max_vocab_size=50000, is_eng = True):
    word_counts = Counter()
    for sentence in sentences:
        # 英文按空格分词，中文按字符分词
        if is_eng:
            words = sentence.split()
        else:
            words = list(sentence)
        word_counts.update(words)
    
    # 按频率排序，添加特殊符号
    vocab = [PAD, UNK, SOS, EOS] + \
            [word for word, count in word_counts.most_common(max_vocab_size - 4)]
    # 建立映射字典
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx, vocab

# 句子转换为索引序列
def sentence_to_index(sentence, word2idx, is_target=False, is_eng = True):
    # 中文按字符处理，英文按单词处理
    if is_eng:
        tokens = sentence.split()
    else:
        tokens = list(sentence)

    # 目标序列的输入添加开始符号
    if is_target:
        tokens = [SOS] + tokens
    
    # 转换为索引，未知词用UNK
    return [word2idx.get(token, word2idx[UNK]) for token in tokens]

# 创建训练数据
def make_data(en_sentences, cn_sentences, en_word2idx, cn_word2idx):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    
    # 计算最大长度，用于padding
    src_max_len = max(len(sent.split()) for sent in en_sentences)
    tgt_max_len = max(len(list(sent)) for sent in cn_sentences) + 2  # 包含SOS和EOS符

    for en, cn in zip(en_sentences, cn_sentences):
        # 转换为索引
        enc_input = sentence_to_index(en, en_word2idx)
        dec_input = sentence_to_index(cn, cn_word2idx, is_target=True, is_eng = False)
        
        # 目标序列的输出是解码器输入向右移位并添加结束符
        dec_output = dec_input[1:] + [cn_word2idx[EOS]]
        
        # 填充序列
        enc_input = pad_sequence(enc_input, src_max_len, en_word2idx[PAD])
        dec_input = pad_sequence(dec_input, tgt_max_len, cn_word2idx[PAD])
        dec_output = pad_sequence(dec_output, tgt_max_len, cn_word2idx[PAD])
        
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
    
    return (
        torch.LongTensor(enc_inputs),
        torch.LongTensor(dec_inputs),
        torch.LongTensor(dec_outputs),
        src_max_len, #用于测试
    )

# 序列填充
def pad_sequence(seq, max_len, pad_idx):
    return seq[:max_len] + [pad_idx] * (max_len - len(seq))

# 数据集类
class TranslationDataset(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    
    def __len__(self):
        return len(self.enc_inputs)
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]



# 加载数据集
en_sentences, cn_sentences = load_dataset(data_file_path, num_lines)
    
# 构建词汇表
en_word2idx, src_vocab = build_vocab(en_sentences)
cn_word2idx, tgt_vocab = build_vocab(cn_sentences, is_eng = None)
# 获取词汇表长度
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
# 构建反映射字典
src_idx2word = {idx: word for idx, word in enumerate(src_vocab)}
tgt_idx2word = {idx: word for idx, word in enumerate(tgt_vocab)}

# 创建训练数据
enc_inputs, dec_inputs, dec_outputs, src_max_len= make_data(
    en_sentences, cn_sentences, en_word2idx, cn_word2idx
)

# 创建数据加载器
dataset = TranslationDataset(enc_inputs, dec_inputs, dec_outputs)
loader = Data.DataLoader(dataset, batch_size, shuffle)