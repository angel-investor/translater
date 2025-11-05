import time
import random

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MyPairDataset, MAX_LENGTH, device, SOS_token, EOS_token, \
    english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs
from model import Encoder, DecoderAttention

# 模型训练参数
mylr = 1e-4
epochs = 2
# 设置teacher_forcing比率为0.5
teacher_forcing_ratio = 0.5
print_interval_num = 1000
plot_interval_num = 100


def train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion):
    # 1、将数据x送入编码器得到编码器结果
    h0 = encoder.init_hidden()
    encoder_output, encoder_hidden = encoder(x, h0)
    # print("encoder_output-->", encoder_output.shape)
    # print("encoder_hidden-->", encoder_hidden.shape)
    # 2、准备解码器的三个参数：Q、K、V
    # 参数1：V --> encoder_output_c-->[10, 256]
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for i in range(x.shape[1]):
        encoder_output_c[i] = encoder_output[0, i]
    # 升维操作，匹配数据的维度
    encoder_output_c = encoder_output_c.unsqueeze(0)
    # 参数2：K，解码器上一个时间步隐藏层输出结果，但是初始化的隐藏层输入为编码器最后一个单词的隐藏层输出结果
    decoder_hidden = encoder_hidden
    # 参数3：Q，解码器上一个时间步预测出的结果，初始化时（最开始时），用的是SOS开始字符
    input_y = torch.tensor([[SOS_token]], device=device)
    # 3、定义部分变量
    my_loss = 0.0
    y_len = y.shape[1]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for idx in range(y_len):
            # 1.1、将Q、K、V送入解码器
            output_y, decoder_hidden, attn_weight = decoder(q=input_y, k=decoder_hidden, v=encoder_output_c)
            # print("output_y-->", output_y.shape)
            # 获取当前时间步真实结果
            target_y = y[0][idx].view(1)
            # print("target_y-->", target_y)
            my_loss += criterion(output_y, target_y)
            # print("my_loss-->", my_loss)
            input_y = y[0][idx].view(1, -1)
    else:
        for idx in range(y_len):
            # 1.1、将Q、K、V送入解码器
            output_y, decoder_hidden, attn_weight = decoder(q=input_y, k=decoder_hidden, v=encoder_output_c)
            # print("output_y-->", output_y.shape)
            # 获取当前时间步真实结果
            target_y = y[0][idx].view(1)
            # print("target_y-->", target_y)
            my_loss += criterion(output_y, target_y)
            # print("my_loss-->", my_loss)
            topk, topi = torch.topk(output_y, k=1)
            # print("topk-->", topk)
            # print("topi-->", topi)
            input_y = topi.detach()

    # 梯度清零
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    # 反向传播
    my_loss.backward()
    # 梯度更新
    encoder_optim.step()
    decoder_optim.step()

    return my_loss.item() / y_len


def train():
    # 实例化dataset对象
    my_dataset = MyPairDataset(my_pairs)
    # 实例化dataloader对象
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化编码器对象
    encoder = Encoder(english_word_n, 256).to(device)
    # 实例化解码器对象
    decoder = DecoderAttention(french_word_n, 256, max_len=MAX_LENGTH).to(device)
    # 实例化优化器对象
    encoder_optim = optim.Adam(encoder.parameters(), lr=mylr)
    decoder_optim = optim.Adam(decoder.parameters(), lr=mylr)
    # 实例化损失函数对象
    criterion = nn.NLLLoss()
    # 定义训练日志的参数（保存损失）
    plot_loss_list = []
    # 开始外层迭代训练
    for epoch in range(1, 1 + epochs):
        # 定义其他的训练日志的参数
        print_loss_total, plot_loss_total = 0, 0
        start_time = time.time()
        for i, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            # print("x-->", x)
            # print("y-->", y)
            my_loss = train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion)
            # print("my_loss-->", my_loss)
            # break
            print_loss_total += my_loss
            plot_loss_total += my_loss
            # 每隔1000步打印日志信息
            if i % print_interval_num == 0:
                avg_loss = print_loss_total / print_interval_num
                print_loss_total = 0.0
                user_time = time.time() - start_time
                print("当前轮次：%d，平均损失：%.2f，耗时：%d" % (epoch, avg_loss, user_time))
            # 每隔100步保存画图的数据
            if i % plot_interval_num == 0:
                avg_loss = plot_loss_total / plot_interval_num
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0.0

        # 保存模型
        torch.save(encoder.state_dict(), "./save_model/encoder_%d.pth" % epoch)
        torch.save(decoder.state_dict(), "./save_model/decoder_%d.pth" % epoch)

    # 绘图展示损失
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/loss.png')
    plt.show()


if __name__ == '__main__':
    train()
