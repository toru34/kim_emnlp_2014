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

def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x

def build_word2count(file_path, w2c=None, min_len=5):
    if w2c is None:
        w2c = defaultdict(lambda: 0)
    for line in open(file_path, encoding='utf-8', errors='ignore'):
        sentence = line.strip().split()
        if len(sentence) < min_len:
            continue
        for word in sentence:
            w2c[word] += 1
    return w2c

def encode(sentence, w2i):
    encoded_sentence = []
    for word in sentence:
        if word in w2i:
            encoded_sentence.append(w2i[word])
        else:
            encoded_sentence.append(w2i['<unk>'])
    return encoded_sentence

def build_dataset(file_path, vocab_size=10000, w2c=None, w2i=None, min_len=5, target=False):
    if w2i is None:
        sorted_w2c = sorted(w2c.items(), key=lambda x: -x[1])
        if target:
            w2i = {w: np.int32(i+2) for i, (w, c) in enumerate(sorted_w2c[:vocab_size-3])}
            w2i['<s>'], w['</s>'] = np.int32(0), np.int32(1)
            w2i['<unk>'] = np.int32(2)
        else:
            w2i = {w: np.int32(i+1) for i, (w, c) in enumerate(sorted_w2c[:vocab_size-1])}
            w2i['<unk>'] = np.int32(0)

    data = []
    for line in open(file_path, encoding='utf-8', errors='ignore'):
        sentence = line.strip().split()
        if len(sentence) < min_len:
            continue
        if target:
            sentence = ['<s>'] + sentence + ['</s>']
        encoded_sentence = encode(sentence, w2i)
        data.append(encoded_sentence)
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w
