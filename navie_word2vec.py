# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/12/22
import tensorflow as tf
import nltk
import random
import numpy as np
from collections import Counter
from nltk.corpus import gutenberg as bg
from tensorflow.keras import Model, layers, Sequential

flatten = lambda l: [item for sublist in l for item in sublist]


def get_batch(batch_size, train_data):
    '''
    获取Batch
    :param batch_size: 每次的batch大小
    :param train_data: 源数据
    :return:
    '''
    random.shuffle(train_data)
    start = 0
    end = batch_size
    length = len(train_data)
    while end < length:
        batch = train_data[start:end]
        tmp = end
        end = end + batch_size
        start = tmp
        yield batch
    if end >= length:
        batch = train_data[start:]
        yield batch


def prepare_sequence(seq, word2index):
    '''
    输入一串序列返回对应的id序列
    :param seq: 源文字序列
    :param word2index: 构建的由word到编号的dict
    :return:
    '''
    idxs = list(map(lambda x: word2index[x] if word2index.get(x) is not None else word2index['<UNK>'], seq))
    return idxs


def prepare_word(word, word2index):
    '''
    根据词语返回对应的编号
    :param word:
    :param word2index:
    :return:
    '''
    return word2index[word] if word2index.get(word) is not None else word2index['<UNK>']


class SkipGram(Model):

    def __init__(self, vocab_size, projection_dim):
        super(SkipGram, self).__init__()
        self.I_H = layers.Embedding(vocab_size, projection_dim)  # input_hidden matrix
        self.H_U = layers.Embedding(vocab_size, projection_dim)  # hidden_out matrix

    def call(self, inputs, predict, normal):
        inputs_embed = self.I_H(inputs)
        predict_embed = self.H_U(predict)
        normal_embed = self.H_U(normal)

        scores = tf.matmul(predict_embed, tf.transpose(inputs_embed, [0, 2, 1]))  # Bx1xD * BxDx1 => Bx1
        #         print("predict shape:{} input shape:{} result shape{}".format(
        #                 predict_embed.shape,tf.transpose(inputs_embed,[0,2,1]).shape,scores.shape))
        scores = tf.squeeze(scores, 2)
        norm_scores = tf.squeeze(tf.matmul(normal_embed, tf.transpose(inputs_embed, [0, 2, 1])),
                                 2)  # BxVxD * BxDx1 => BxV

        nll = tf.expand_dims(
            -tf.math.reduce_mean(tf.math.log(tf.math.exp(scores) / tf.math.reduce_sum(tf.math.exp(norm_scores), 1), 1)),
            axis=0)  # log-softmax

        return nll  # negative log likelihood


if __name__ == '__main__':
    corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100]  # sampling sentences for test

    corpus = [[word.lower() for word in sent] for sent in corpus]
    word_count = Counter(flatten(corpus))
    border = int(len(word_count) * 0.01)
    stopwords = word_count.most_common()[:border] + list(reversed(word_count.most_common()))[:border]
    stopwords = [s[0] for s in stopwords]
    vocab = list(set(flatten(corpus)) - set(stopwords))
    vocab.append('<UNK>')
    word2index = {'<UNK>': 0}
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)

    index2word = {v: k for k, v in word2index.items()}

    WINDOW_SIZE = 3
    windows = flatten(
        [list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in
         corpus])

    train_data = []

    for window in windows:
        for i in range(WINDOW_SIZE * 2 + 1):
            if i == WINDOW_SIZE or window[i] == '<DUMMY>':
                continue
            train_data.append((window[WINDOW_SIZE], window[i]))
    print(train_data[:WINDOW_SIZE * 2])

    X_train = []
    Y_train = []

    for tr in train_data:
        X_train.append(prepare_word(tr[0], word2index))
        Y_train.append(prepare_word(tr[1], word2index))

    train_data = list(zip(X_train, Y_train))

    EMBEDDING_SIZE = 30
    BATCH_SIZE = 256
    EPOCH = 100
    losses = []
    model = SkipGram(len(word2index), EMBEDDING_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(EPOCH):
        for i, batch in enumerate(get_batch(BATCH_SIZE, train_data)):
            inputs, targets = zip(*batch)
            vocabs = tf.expand_dims(tf.convert_to_tensor(prepare_sequence(list(vocab), word2index)), 0)
            inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 1)
            targets = tf.expand_dims(tf.convert_to_tensor(targets), 1)
            with tf.GradientTape() as tape:
                loss = model(inputs, targets, vocabs)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                losses.append(loss.numpy().tolist()[0])

        if epoch % 10 == 0:
            print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
