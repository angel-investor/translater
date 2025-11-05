# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyPairDataset,MAX_LENGTH, device, \
    english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs
from dataset import MyPairDataset, SOS_token


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, hidden_size):
        super().__init__()
        self.en_vocab_size = en_vocab_size
        self.hidden_size = hidden_size

        # 定义网络层
        self.embedding = nn.Embedding(en_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_x, hidden):
        # input_x 经过Embedding层进行词向量表示 数据形状 [1,6] --> [1,6,256]
        embed_x = self.embedding(input_x)
        # embed_x 进入gru层
        output, hidden = self.gru(embed_x, hidden)
        return output, hidden

    def init_hidden(self):
        # num_layers：RNN的层数，默认就是1
        # batch_size：批次数，也就是样本数，也就是几句话，1句话就是1个批次
        # hidden_size：隐藏层数据维度，也就是特征数
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 测试编码器
def test_encoder():
    # 准备数据
    my_dataset = MyPairDataset(my_pairs)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

    # 准备模型
    encoder = Encoder(english_word_n, 256).to(device)
    for x, y in my_dataloader:
        hidden = encoder.init_hidden()
        output, hidden = encoder(x, hidden)
        print("output-->", output)
        print("hidden-->", hidden)
        break


class Decoder(nn.Module):
    def __init__(self, fr_vocab_size, hidden_size):
        super().__init__()
        self.fr_vocab_size = fr_vocab_size
        self.hidden_size = hidden_size

        # 定义网络层
        self.embedding = nn.Embedding(fr_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, fr_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_y, hidden):
        # input_y 经过Embedding层进行词向量表示 数据形状 [1,1] --> [1,1,256]
        embed_y = self.embedding(input_y)
        # 将embed_y进行Relu激活函数操作，防止过拟合
        embed_y = F.relu(embed_y)
        # 将embed_y及hidden送入GRU模型： output-->[1,1,256]; hn-->[1,1,256]
        output, hn = self.gru(embed_y, hidden)
        # 将上述结果送入输入层
        result = self.softmax(self.out(output[0]))
        return result, hn

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 测试解码器
def test_decoder():
    # 准备数据
    my_dataset = MyPairDataset(my_pairs)
    # batch_size=1 --> 这个值为2时，如果每个样本的 长度 不匹配，就会直接报错
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

    # 准备模型
    encoder = Encoder(english_word_n, 256).to(device)
    decoder = Decoder(french_word_n, 256).to(device)
    for x, y in my_dataloader:
        hidden = encoder.init_hidden()
        encode_output_c, hn = encoder(x, hidden)
        # print("encode_output_c-->", encode_output_c.shape)
        # print("hidden-->", hn.shape)
        for i in range(y.shape[1]):
            tmp = y[0][i].view(1, -1)
            output, hn = decoder(tmp, hn)
            print("output-->", output.shape)
            print("hidden-->", hn.shape)
            break
        break


class DecoderAttention(nn.Module):
    def __init__(self, fr_vocab_size, hidden_size, max_len):
        super().__init__()
        self.fr_vocab_size = fr_vocab_size
        self.hidden_size = hidden_size
        self.max_len = max_len

        # 定义网络层
        # 定义Embedding层
        self.embedding = nn.Embedding(fr_vocab_size, hidden_size)
        # 按照注意力计算规则的第一规则（Q和K进行拼接，然后经过Linear层线性变换，Softmax之后，再和V相乘）
        # 定义Linear层线性变换
        self.attn = nn.Linear(hidden_size * 2, max_len)
        # 按照注意力计算步骤，如果第一步时拼接操作，第二步需要将q和第一步的结果进行拼接，然后再经过第三步linear层指定维度输出
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 定义输出层
        self.out = nn.Linear(hidden_size, fr_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, q, k, v):
        # q-->代表上一个时间步预测出来的结果--》shape-->[1, 1]
        # k-->代表上一个时间步隐藏层输出结果--》shape->[1,1, 256]
        # v-->代表编码器对英文句子编码之后的结果：shape:[1, ?, 256]-->[1, max_len, 256]-->[1, 10, 256]
        # 1. 对q输入进行embedding的表示:embed_x-->shape-->[1,1,256]
        embed_x = self.embedding(q)
        # 2. 对上述的embed_x进行随机的失活:dropout_x-->[1, 1, 256]
        dropout_x = F.dropout(embed_x, p=0.1)
        # 3.按照注意力计算规则计算：
        # 将query(dropout_x-->[1,1,256])和key(k-->[1,1,256])进行拼接;然后经过Linear线性变换;
        # atten_weights-->shape-->[1,1,?]-->[1,1,max_len]-->[1,1,10]
        attn_weights = self.attn(torch.cat([dropout_x, k], dim=-1))
        # 将atten_weights经过softmax之后 要和V(v-->[1, 10, 256])进行相乘-->attn_applied-->[1,1,256]
        # F.softmax(atten_weights, dim=-1)-->[1, 1, 10]
        attn_applied = torch.bmm(F.softmax(attn_weights, dim=-1), v)
        # 按照注意力步骤的第二步：将上述的结果attn_applied([1,1,256]和原来query(dropout_x([1,1,256]))进行拼接
        # 还要按照第三步，将拼接后的结果按照指定尺寸输出：Linear线性变换;attention-->[1,1,256]
        attention = self.attn_combine(torch.cat([dropout_x, attn_applied], dim=-1))
        attention = F.relu(attention)
        # 4. 将attention([1,1, 256])和k(hidden([1,1,256]))送入GRU模型;output-->[1, 1,256]
        output, hn = self.gru(attention, k)# 5. 基于上述的output结果送入输出层预测结果result-->[1, 4345]
        result = self.softmax(self.out(output))
        return result[0], hn, attn_weights


# 测试（带注意力）解码器
def test_decoderAttention():
    # 准备数据
    my_dataset = MyPairDataset(my_pairs)
    # batch_size=1 --> 这个值为2时，如果每个样本的 长度 不匹配，就会直接报错
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

    # 准备编码器解码器
    encoder = Encoder(english_word_n, 256).to(device)
    attn_decoder = DecoderAttention(french_word_n, 256, max_len=MAX_LENGTH).to(device)
    # 将数据送入编码器和解码器
    for x, y in tqdm(my_dataloader):
        hidden = encoder.init_hidden()
        encoder_output, encoder_hidden = encoder(x, hidden)
        # 1、获得注意力计算的V：需要用MAX_LENGTH约束：encoder_output_c-->[10, 256]
        # 这里的目的是，对V做长度约束，从而固定模型中间的数据维度，先全0初始化填充，再将encoder_output赋给encoder_output_c
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for i in range(x.shape[1]):
            encoder_output_c[i] = encoder_output[0, i]
        # 升维操作，匹配数据的维度
        encoder_output_c = encoder_output_c.unsqueeze(0)
        # 2、获得注意力计算的Q：最开始的时候，SOS开始字符
        # 这里的input_y的维度需要调整成[1,1],匹配模型输入
        input_y = torch.tensor([[SOS_token]], device=device)
        # 3、获得注意力计算的K：最开始的时候，用的是编码器的最后一个时间步的输出结果
        hn = encoder_hidden
        # 将上述的Q、K、V一个时间步一个时间步的送入模型中
        for j in range(y.shape[1]):
            result, hn, attn_weights = attn_decoder(q=input_y, k=hn, v=encoder_output_c)
            # print("result-->", result.shape)
            # print("hn-->", hn.shape)
            # print("attn_weights-->", attn_weights.shape)


if __name__ == '__main__':
    # test_encoder()
    # test_decoder()
    test_decoderAttention()
