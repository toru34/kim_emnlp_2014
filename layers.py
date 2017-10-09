import numpy as np
import _dynet as dy

class CNNText:
    def __init__(self, model, emb_dim, win_sizes, num_cha, num_fil, function, dropout_prob=0.5):
        pc = model.add_subcollection()

        # CNN
        self._Ws = [pc.add_parameters((win_size, emb_dim, num_cha, num_fil)) for win_size in win_sizes]
        self._bs = [pc.add_parameters((num_fil)) for _ in win_sizes]
        self.function = function
        self.dropout_prob = dropout_prob

        self.num_fil = num_fil
        self.win_sizes = win_sizes

        self.pc = pc
        self.spec = (emb_dim, win_sizes, num_cha, num_fil, function, dropout_prob)

    def __call__(self, x, train=True):
        sen_len = x.dim()[0][0]

        convds = [dy.conv2d_bias(x, W, b, stride=(1, 1)) for W, b in zip(self.Ws, self.bs)]
        actds = [self.function(convd) for convd in convds]
        poolds = [dy.maxpooling2d(actd, ksize=(sen_len-win_size+1, 1), stride=(sen_len-win_size+1, 1)) for win_size, actd in zip(self.win_sizes, actds)]
        z = dy.concatenate([dy.reshape(poold, (self.num_fil,)) for poold in poolds])

        if train:
            # Apply dropout
            z = dy.dropout(z, self.dropout_prob)
        return z

    def associate_parameters(self):
        self.Ws = [dy.parameter(_W) for _W in self._Ws]
        self.bs = [dy.parameter(_b) for _b in self._bs]

    @staticmethod
    def from_spec(spec, model):
        emb_dim, win_sizes, num_cha, num_fil, function, dropout_prob = spec
        return CNNText(model, emb_dim, win_sizes, num_cha, num_fil, function, dropout_prob)

    def param_collection(self):
        return self.pc

class Dense:
    def __init__(self, model, in_dim, out_dim, function=lambda x: x):
        pc = model.add_subcollection()

        self._W = pc.add_parameters((out_dim, in_dim))
        self._b = pc.add_parameters((out_dim))
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

    def scale_W(self, l2_norm_lim):
        W_l2norm = np.linalg.norm(self._W.as_array(), ord=2)
        if W_l2norm > l2_norm_lim:
            self._W.scale(l2_norm_lim/W_l2norm)
