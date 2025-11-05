import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MyPairDataset, MAX_LENGTH, device, SOS_token, EOS_token, \
    english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs
from model import Encoder, DecoderAttention


# 内部预测函数
def predict_iter(x, encoder, decoder):
    with torch.no_grad():
        # 编码，一次性的送数据，得到编码器的结果
        h0 = encoder.init_hidden()
        encoder_output, encoder_hidden = encoder(x, h0)
        # print("encoder_output-->", encoder_output.shape)      # [1, 10, 256]
        # print("encoder_hidden-->", encoder_hidden.shape)
        # 2.准备解码器模型的三个参数：Q、K、V
        # 2.1 参数1:V-->encoder_output_c  # [10, 256]
        encoder_hidden_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for i in range(x.shape[1]):
            encoder_hidden_c[i] = encoder_output[0, i]
        encoder_hidden_c = encoder_hidden_c.unsqueeze(dim=0)  # encoder_output_c->[1,10, 256]
        # 参数2: K，解码器上一时间步隐藏层输出结果，但是初始化的隐藏层输入为编码器最后一个单词的隐藏层输出结果
        decoder_hidden = encoder_hidden
        # 参数3: Q, 解码器上一个时间步预测出的结果，但是初始化是第一步要用到特殊的开始字符
        input_y = torch.tensor([[SOS_token]], device=device)
        # 准备：保存预测结果的参数
        decode_words = []
        # 初始化一个全零张量的矩阵，方便后续记录每个时间步解码出来的权重
        decode_weights = torch.zeros(MAX_LENGTH, MAX_LENGTH, device=device)
        # 开始预测
        for idx in range(MAX_LENGTH):
            output_y, decoder_hidden, attn_weight = decoder(q=input_y, k=decoder_hidden, v=encoder_hidden_c)
            # 基于模型的预测结果output_y[1, 4345],找出最大概率值对应的索引就是真实的法文
            topv, topi = torch.topk(output_y, k=1)
            # decode_weights-->[10 ,10]     attn_weight-->[1, 1, 10]
            decode_weights[idx] = attn_weight
            # print(f'topi--> {topi}')
            # print(f'topi--> {topi.item()}')
            # 如果输出值是终止符，则循环停止
            if topi.item() == EOS_token:
                decode_words.append('<EOS>')
                break
            else:
                decode_words.append(french_index2word[topi.item()])

            # 更新input_y
            input_y = topi.detach()

        return decode_words, decode_weights[:idx + 1]


def predict():
    # 实例化编码器模型对象
    encoder = Encoder(english_word_n, 256)
    # 加载训练好的编码器模型的参数
    encoder.load_state_dict(torch.load("./save_model/encoder_2.pth"))
    encoder.to(device)
    # print(encoder)
    # 实例化解码器模型对象
    decoder = DecoderAttention(french_word_n, 256, max_len=MAX_LENGTH)
    # 加载训练好的解码器模型的参数
    decoder.load_state_dict(torch.load("./save_model/decoder_2.pth"))
    decoder.to(device)
    # print(decoder)
    my_sample_pairs = [
        ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
        ['i m more than a friend .', 'je suis plus qu une amie .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
    # print("my_sample_pairs-->", my_sample_pairs)
    # 遍历每个样本去实现模型的预测
    for idx, pair in enumerate(my_sample_pairs):
        # 获取样本的英文句子
        x = pair[0]
        # 获得样本的法文句子
        y = pair[1]
        # 将原始的英文句子进行张量的表示
        list_x = [english_word2index[word] for word in x.split(" ")]
        list_x.append(EOS_token)
        tensor_x = torch.tensor([list_x], dtype=torch.long, device=device)
        # print("tensor_x-->", tensor_x.shape)
        decode_words, decode_weights = predict_iter(tensor_x, encoder, decoder)
        predict_text = ' '.join(decode_words)

        print('x-->', x)
        print('y-->', y)
        print('predict_text-->', predict_text)
        print("**" * 20)



if __name__ == '__main__':
    predict()
