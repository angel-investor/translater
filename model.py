import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_preprocess import device, read_data
from dataset import MyPairDataset

english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs = read_data()

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, hidden_size):
        super().__init__()
        self.en_vocab_size = en_vocab_size
        self.hidden_size = hidden_size

        # 定义网络层
        self.embedding = nn.Embedding(en_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size, batch_first=True)


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
        return torch.zeros(1, 1,self.hidden_size, device=device)


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
        for i in range(y.shape[1]):
            tmp = y[0][i].view(1, -1)
            output, hn = decoder(tmp, hn)
            print("output-->", output.shape)
            print("hidden-->", hn.shape)
            break

        break

if __name__ == '__main__':
    # test_encoder()
    test_decoder()