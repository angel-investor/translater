from torch.utils.data import Dataset
import torch

from data_preprocess import read_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device-->", device)
SOS_token = 0
EOS_token = 1
max_length = 10
file_path = "./data/eng-fra-v2.txt"

# 全局函数调用加载数据的函数
english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs = read_data()


class MyPairDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 迭代器索引越界处理
        item = min(max(0, item),self.sample_len)
        x = self.my_pairs[item][0]
        y = self.my_pairs[item][1]

        # 对x,y进行数值化
        x = [english_word2index[word] for word in x.split(" ")]
        y = [french_word2index[word] for word in y.split(" ")]
        # 在数值化列表末尾添加结束符号
        x.append(EOS_token)
        y.append(EOS_token)
        # 张量化
        tensor_x = torch.tensor(x, device=device)
        tensor_y = torch.tensor(y, device=device)

        return tensor_x, tensor_y


if __name__ == '__main__':
    my_dataset = MyPairDataset(my_pairs=my_pairs)
    # 取第11条数据出来
    print('dataset-->', my_dataset[10])