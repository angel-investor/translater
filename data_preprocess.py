import re
import torch

"""
# TODO 1.数据预处理
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device-->", device)
SOS_token = 0
EOS_token = 1
max_length = 10
file_path = "./data/eng-fra-v2.txt"


def normalize_string(s):
    s = s.lower().strip()
    # 将  .!? 替换成 ' .',' !',' ?'
    s = re.sub(r"([.!?])", r" \1", s)
    # 将所有的一个或多个非a-zA-Z.!?字符转换成空格
    s = re.sub(r"([^a-zA-Z.!?]+)", r" ", s)
    # print("s-->", s)
    return s


def read_data():
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split("\n")

    # 将文件中的语料数据转换成列表
    my_pairs = [[normalize_string(s) for s in line.split('\t')] for line in data]
    # print("my_pairs -->", my_pairs)

    # 构建英文和法文词表，为下面数值化做铺垫
    # 添加开始结束符号
    english_word2index = {"SOS_token": 0, "EOS_token": 1}
    english_word_n = 2
    french_word2index = {"SOS_token": 0, "EOS_token": 1}
    french_word_n = 2
    # 遍历  英-法对预料对  列表
    for pair in my_pairs:
        # 遍历英文，构建英文word2index词表
        for word in pair[0].split(" "):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        # 遍历法文，构建法文word2index词表
        for word in pair[1].split(" "):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # print(len(english_word2index))
    # print(len(french_word2index))

    # 构建index2word词表
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}

    return (english_word2index,
            english_index2word,
            english_word_n,
            french_word2index,
            french_index2word,
            french_word_n,
            my_pairs
            )


if __name__ == '__main__':
    # normalize_string("A iow,  中文!?ni VB. 你好")
    read_data()
