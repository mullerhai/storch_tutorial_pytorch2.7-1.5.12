import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 加载数据集
train_data, test_data = AG_NEWS(root='.data')


# 定义一个函数来提取文本字段
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# 初始化tokenizer
tokenizer = get_tokenizer('basic_english')

# 构建词汇表
vocab = build_vocab_from_iterator(map(yield_tokens, train_data), specials=["<pad>"])
vocab.set_default_index(vocab["<unk>"])  # 未知词的处理

# 查看一些数据点
print(vars(train_data[0]))  # 查看第一个数据点