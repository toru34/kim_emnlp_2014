import _dynet as dy
import numpy as np

class CNN:
    def __init__(self, model, vocab_size, emb_dim, num_fil, function, dropout_prob=0.5):
        self.V = model.add_lookup_parameters((vocab_size, emb_dim))
        self._W_w3 = model.add_parameters((3, emb_dim, 1, num_fil))
        self._W_w4 = model.add_parameters((4, emb_dim, 1, num_fil))
        self._W_w5 = model.add_parameters((5, emb_dim, 1, num_fil))
        self._b_w3 = model.add_parameters((num_fil))
        self._b_w4 = model.add_parameters((num_fil))
        self._b_w5 = model.add_parameters((num_fil))
        self.function = function
        self.emb_dim = emb_dim
        self.num_fil = num_fil
        self.dropout_prob = dropout_prob

    def associate_parameters(self):
        self.W_w3 = dy.parameter(self._W_w3)
        self.W_w4 = dy.parameter(self._W_w4)
        self.W_w5 = dy.parameter(self._W_w5)
        self.b_w3 = dy.parameter(self._b_w3)
        self.b_w4 = dy.parameter(self._b_w4)
        self.b_w5 = dy.parameter(self._b_w5)

    def f_prop(self, sentence, train):
        sen_len = len(sentence)

        word_embs = dy.concatenate([dy.lookup(self.V, word) for word in sentence], d=1)
        word_embs = dy.transpose(word_embs)
        word_embs = dy.reshape(word_embs, (sen_len, self.emb_dim, 1))

        convd_w3 = dy.conv2d_bias(word_embs, self.W_w3, self.b_w3, stride=(1, 1))
        convd_w4 = dy.conv2d_bias(word_embs, self.W_w4, self.b_w4, stride=(1, 1))
        convd_w5 = dy.conv2d_bias(word_embs, self.W_w5, self.b_w5, stride=(1, 1))

        actd_w3 = self.function(convd_w3)
        actd_w4 = self.function(convd_w4)
        actd_w5 = self.function(convd_w5)

        poold_w3 = dy.maxpooling2d(convd_w3, ksize=(sen_len-3+1, 1), stride=(sen_len-3+1, 1))
        poold_w4 = dy.maxpooling2d(convd_w4, ksize=(sen_len-4+1, 1), stride=(sen_len-4+1, 1))
        poold_w5 = dy.maxpooling2d(convd_w5, ksize=(sen_len-5+1, 1), stride=(sen_len-5+1, 1))

        poold_w3 = dy.reshape(poold_w3, (self.num_fil,))
        poold_w4 = dy.reshape(poold_w4, (self.num_fil,))
        poold_w5 = dy.reshape(poold_w5, (self.num_fil,))

        z = dy.concatenate([poold_w3, poold_w4, poold_w5])

        if train:
            # Apply dropout
            p = dy.random_bernoulli(z.dim()[0], self.dropout_prob)
            z = dy.cmult(z, p)
        else:
            z = z*self.dropout_prob

        return z

class Dense:
    def __init__(self, model, in_dim, out_dim, function=lambda x: x):
        self._W = model.add_parameters((out_dim, in_dim))
        self._b = model.add_parameters((out_dim))
        self.function = function

    def associate_parameters(self):
        self.W = dy.parameter(self._W)
        self.b = dy.parameter(self._b)

    def f_prop(self, x, train):
        return self.function(self.W*x + self.b)
