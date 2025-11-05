import torch
from matplotlib import pyplot as plt

from predict import predict_iter, predict

from dataset import MyPairDataset, MAX_LENGTH, device, SOS_token, EOS_token, \
    english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs
from model import Encoder, DecoderAttention

def attention_show():
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
    sentence = "we are both teachers ."
    list_x = [english_word2index[word] for word in sentence.split(" ")]
    list_x.append(EOS_token)
    tensor_x = torch.tensor([list_x], dtype=torch.long, device=device)
    decode_words, decode_weights = predict_iter(tensor_x, encoder, decoder)
    predict_text = ' '.join(decode_words)
    print('sentence-->', sentence)
    print('predict_text-->', predict_text)
    plt.matshow(decode_weights.cpu())
    plt.savefig('./imgs/attention_show.png')
    plt.show()

if __name__ == '__main__':
    attention_show()