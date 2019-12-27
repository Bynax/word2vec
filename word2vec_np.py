# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/12/27

from collections import defaultdict
import numpy as np


def data_preparation():
    text = 'natural language processing and machine learning is fun and exciting'
    corpus = list(map(lambda x: x.lower(), text.split()))
    print(corpus)


class OneHotConvert:
    """
    将语料转换成OneHot
    """

    def __init__(self, windows=3):
        self.windows = windows

    def generate_training_data(self, corpus):
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        # 语料库中词的数目
        self.word_num = (word_counts.keys())
        self.words_list = list(word_counts.keys())
        # dict word -> index
        self.word2index = dict((word, index) for index, word in enumerate(self.words_list))
        # dict index -> word
        self.index2word = dict((index, word) for index, word in enumerate(self.words_list))
        training_data = []
        # Cycle through each sentence in corpus
        for sentence in corpus:
            sent_len = len(sentence)
            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = self.word2onehot(sentence[i])
                # Cycle through context window
                w_context = []
                # Note: window_size 2 will have range of 5 values
                for j in range(i - self.window, i + self.window + 1):
                    # Criteria for context word
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range
                    if j != i and j <= sent_len - 1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j])
                        # training_data contains a one-hot representation of the target word and context words
                training_data.append([w_target, w_context])
        return np.array(training_data)

    def word2onehot(self, word):
        """
        将给定的word转成one hot形式
        :param word:
        :return:
        """
        word_vec = [0 for i in range(self.word_num)]
        word_vec[self.word2index[word]] = 1
        return word_vec


class SkipGram:
    def __init__(self, training_data, window_size=2, dim=10, epoch=50, learning_rate=0.01):
        """
        初始化超参数
        :param window_size: context window +- center word
        :param dim: context window +- center word
        :param epoch: context window +- center word
        :param learning_rate: context window +- center word
        """
        self.window_size = window_size
        self.dim = dim
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.loss = 0

    def train(self, training_data):
        # w1 w2 -> [V,N]
        self.w1 = np.random.uniform(-1, 1, (len(training_data), self.dim))
        # w1和w2为了方便形状设置成相同的，运算的时候只需要简单的转置即可
        self.w2 = np.random.uniform(-1, 1, (len(training_data), self.dim))

        for i in range(self.epochs):
            self.loss = 0  # 每个epoch首先将loss初始化
            for w_c, w_t in training_data:
                y_predict, hidden, output = self.forward(w_t)
                pass


    def forward(self, x):
        # x [1,V] dot [V,N] = [1, N]
        hidden = np.dot(x, self.w1).reshape(1, self.dim)
        # [V,N] dot [N,1] = [V, 1]
        output = np.dot(self.w2, hidden.T)
        y_c = self.softmax(output)
        return y_c, hidden, output

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    # data_preparation()
    pass
