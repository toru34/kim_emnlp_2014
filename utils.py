from collections import defaultdict

import numpy as np

def init_V(w2v, w2i):
    V_init = np.random.normal(size=(len(w2i), w2v.vector_size))
    for w, i in w2i.items():
        if w in w2v:
            V_init[w2i[w]] = w2v[w]
    return V_init

def make_emb_zero(V, word_ids):
    emb_dim = V.shape()[1]
    for word_id in word_ids:
        V.init_row(word_id, np.zeros(emb_dim))

def binary_pred(x):
    return np.piecewise(x, [x < 0.5, x >= 0.5], [0, 1]).astype('int32')

def forwards(layers, x, test=False):
    for layer in layers:
        x = layer.forward(x, test)
    return x

def associate_parameters(layers):
    for layer in layers:
        layer.associate_parameters()

def build_batch(data, w2i, max_win):
    max_len = max(len(datum) for datum in data)

    data = [[w2i['<s>']]*(max_win - 1) + datum + [w2i['</s>']]*(max_len - len(datum)) for datum in data]
    return np.array(data)

def sort_data_by_length(data_X, data_y):
    data_X_lens = [len(com) for com in data_X]
    sorted_data_indexes = sorted(range(len(data_X_lens)), key=lambda x: -data_X_lens[x])

    data_X = [data_X[ind] for ind in sorted_data_indexes]
    data_y = [data_y[ind] for ind in sorted_data_indexes]
    return data_X, data_y

def build_w2c(data_path, n_data=1e+10, len_lim=1e+10, vocab=None):
    w2c = defaultdict(lambda: 0)

    count = 0
    for line in open(data_path):
        sentence = line.strip().split()
        if len(sentence) > len_lim:
            continue

        for word in sentence:
            if vocab:
                if word in vocab:
                    w2c[word] += 1
            else:
                w2c[word] += 1

        count += 1
        if count >= n_data:
            break

    return w2c

def build_w2i(data_path, w2c, unk='<unk>', vocab_size=int(1e+10)):
    sorted_w2c = sorted(w2c.items(), key=lambda x: -x[1])
    sorted_w = [w for w, c in sorted_w2c if w != unk]

    w2i = {w: np.int32(i+3) for i, w in enumerate(sorted_w[:vocab_size-3])}
    w2i['<s>'], w2i['</s>'], w2i[unk] = np.int32(0), np.int32(1), np.int32(2)
    i2w = {i: w for w, i in w2i.items()}

    return w2i, i2w

def encode(sentence, w2i, unk='<unk>'):
    encoded_sentence = []
    for word in sentence:
        if word in w2i:
            encoded_sentence.append(w2i[word])
        else:
            encoded_sentence.append(w2i[unk])
    return encoded_sentence

def build_dataset(x_path, y_path, w2i, unk='<unk>', n_data=1e+10, len_lim=1e+10):
    data_x, data_y = [], []

    count = 0
    for line_x, line_y in zip(open(x_path), open(y_path)):
        sentence_x = line_x.strip().split()
        id_y       = int(line_y)

        if len(sentence_x) > len_lim:
            continue

        encoded_sentence_x = encode(sentence_x, w2i, unk=unk)

        data_x.append(encoded_sentence_x)
        data_y.append(id_y)

        count += 1
        if count >= n_data:
            break

    return data_x, data_y
