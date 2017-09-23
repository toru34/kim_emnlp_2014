import _dynet as dy

class CNNText:
    def __init__(self, model, emb_dim, win_sizes, num_fil, function, dropout_prob=0.5, multichannel=False):
        pc = model.add_subcollection()

        if multichannel:
            in_fil = 2
        else:
            in_fil = 1

        # CNN
        self._Ws = [pc.add_parameters((win_size, emb_dim, in_fil, num_fil)) for win_size in win_sizes]
        self._bs = [pc.add_parameters((num_fil)) for _ in win_sizes]
        self.function = function
        self.dropout_prob = dropout_prob

        self.emb_dim = emb_dim
        self.num_fil = num_fil
        self.win_sizes = win_sizes
        self.multichannel = multichannel

        self.pc = pc
        self.spec = (emb_dim, win_sizes, num_fil, function, dropout_prob, multichannel)

    def __call__(self, word_embs, train=True):
        if self.multichannel:
            sen_len = len(word_embs[0])

            word_embs1 = dy.concatenate(word_embs[0], d=1)
            word_embs2 = dy.concatenate(word_embs[1], d=1)

            word_embs1 = dy.transpose(word_embs1)
            word_embs2 = dy.transpose(word_embs2)

            word_embs = dy.concatenate([word_embs1, word_embs2], d=2)
        else:
            sen_len = len(word_embs)

            word_embs = dy.concatenate(word_embs, d=1)
            word_embs = dy.transpose(word_embs)
            word_embs = dy.reshape(word_embs, (sen_len, self.emb_dim, 1))

        convds = [dy.conv2d_bias(word_embs, W, b, stride=(1, 1)) for W, b in zip(self.Ws, self.bs)]
        actds = [self.function(convd) for convd in convds]
        poolds = [dy.maxpooling2d(actd, ksize=(sen_len-win_size+1, 1), stride=(sen_len-win_size+1, 1)) for win_size, actd in zip(self.win_sizes, actds)]
        z = dy.concatenate([dy.reshape(poold, (self.num_fil,)) for poold in poolds])

        if train:
            # Apply dropout
            p = dy.random_bernoulli(len(self.win_sizes)*self.num_fil, self.dropout_prob)
            z = dy.cmult(z, p)
        else:
            z = z*self.dropout_prob

        return z

    def associate_parameters(self):
        self.Ws = [dy.parameter(_W) for _W in self._Ws]
        self.bs = [dy.parameter(_b) for _b in self._bs]

    @staticmethod
    def from_spec(spec, model):
        emb_dim, win_sizes, num_fil, function, dropout_prob, multichannel = spec
        return CNNText(model, emb_dim, win_sizes, num_fil, function, dropout_prob, multichannel)

    def param_collection(self):
        return self.pc

class Dense:
    def __init__(self, model, in_dim, out_dim, function=lambda x: x):
        pc = model.add_subcollection()

        self._W = model.add_parameters((out_dim, in_dim))
        self._b = model.add_parameters((out_dim))
        self.function = function

        self.pc = pc
        self.spec = (in_dim, out_dim, function)

    def __call__(self, x, train):
        return self.function(self.W*x + self.b)

    def associate_parameters(self):
        self.W = dy.parameter(self._W)
        self.b = dy.parameter(self._b)

    @staticmethod
    def from_spec(spec, model):
        in_dim, out_dim, function = spec
        return Dense(model, in_dim, out_dim, function)

    def param_collection(self):
        return self.pc
