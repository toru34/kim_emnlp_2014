from collections import defaultdict

import numpy as np

def binary_pred(x):
    if x > 0.5:
        return 1
    else:
        return 0

def associate_parameters(layers):
    for layer in layers:
        layer.associate_parameters()

def f_props(layers, x, train=True):
    for layer in layers:
        x = layer.f_prop(x, train)
    return x

def build_word2count(file_path, w2c=None, vocab=None, min_len=5):
    if w2c is None:
        w2c = defaultdict(lambda: 0)
    for line in open(file_path, encoding='utf-8', errors='ignore'):
        sentence = line.strip().split()
        if len(sentence) < min_len:
            continue
        for word in sentence:
            if vocab:
                if word in vocab:
                    w2c[word] += 1
            else:
                w2c[word] += 1
    return w2c

def encode(sentence, w2i):
    encoded_sentence = []
    for word in sentence:
        if word in w2i:
            encoded_sentence.append(w2i[word])
        else:
            encoded_sentence.append(w2i['unk'])
    return encoded_sentence

def build_dataset(file_path, vocab_size=10000, w2c=None, w2i=None, min_len=5):
    if w2i is None:
        sorted_w2c = sorted(w2c.items(), key=lambda x: -x[1])
        w2i = {w: np.int32(i+1) for i, (w, c) in enumerate(sorted_w2c[:vocab_size-1])}
        w2i['unk'] = np.int32(0)

    data = []
    for line in open(file_path, encoding='utf-8', errors='ignore'):
        sentence = line.strip().split()
        if len(sentence) < min_len:
            continue
        encoded_sentence = encode(sentence, w2i)
        data.append(encoded_sentence)
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w
