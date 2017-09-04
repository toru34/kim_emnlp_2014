import math
import time
import argparse

import _dynet as dy
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from utils import build_word2count, build_dataset, f_props, associate_parameters, binary_pred
from layers import CNN, Dense

RANDOM_STATE = 42

# Activate autobatching
dyparams = dy.DynetParams()
dyparams.set_autobatch(True)
dyparams.set_autobatch(RANDOM_STATE)
dyparams.init()

def main():
    parser = argparse.ArgumentParser(description='CNN sentence classifier in DyNet')

    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs for training [default: 3]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('--num_filters', type=int, default=20, help='number of filters in each window size [default: 20]')
    parser.add_argument('--emb_dim', type=int, default=64, help='embedding size for each word [default: 64]')
    parser.add_argument('--vocab_size', type=int, default=30000, help='vocabulary size [default: 30000]')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='dropout probability [default: 0.5]')
    args = parser.parse_args()

    VOCAB_SIZE = args.vocab_size
    EMB_DIM = args.emb_dim
    OUT_DIM = 1
    NUM_FIL = args.num_filters
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.num_epochs
    DROPOUT_PROB = args.dropout_prob

    # Build dataset
    w2c = build_word2count('./data/rt-polarity.neg', min_len=5)
    w2c = build_word2count('./data/rt-polarity.pos', w2c, min_len=5)

    data_neg, w2i, i2w = build_dataset('./data/rt-polarity.neg', vocab_size=30000, w2c=w2c, min_len=5)
    data_pos, _, _ = build_dataset('./data/rt-polarity.pos', w2i=w2i, min_len=5)

    data_X = data_neg + data_pos
    data_y = [0]*len(data_neg) + [1]*len(data_pos)

    data_X, data_y = shuffle(data_X, data_y, random_state=RANDOM_STATE)

    train_X, valid_X, train_y, valid_y = train_test_split(data_X, data_y, test_size=0.1, random_state=RANDOM_STATE)

    # Build model
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    layers = [
        CNN(model, VOCAB_SIZE, EMB_DIM, NUM_FIL, dy.tanh, DROPOUT_PROB),
        Dense(model, NUM_FIL*3, OUT_DIM, dy.logistic),
    ]

    n_batches = math.ceil(len(train_X)/BATCH_SIZE)
    start_time = time.time()

    for epoch in range(N_EPOCHS):
        train_b_preds = []
        train_b_losses = []
        train_X, train_y = shuffle(train_X, train_y, random_state=RANDOM_STATE)
        for i in range(n_batches):
            # Create a new computation graph
            dy.renew_cg()
            associate_parameters(layers)

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            train_X_mb = train_X[start:end]
            train_y_mb = train_y[start:end]

            train_mb_losses = []
            for instance_x, instance_y in zip(train_X_mb, train_y_mb):
                t = dy.scalarInput(instance_y)
                y = f_props(layers, instance_x, train=True)

                train_b_preds.append(binary_pred(y.value()))
                loss = dy.binary_log_loss(y, t)
                train_mb_losses.append(loss)

            train_mb_loss = dy.average(train_mb_losses)

            # Forward propagation
            train_b_losses.append(train_mb_loss.value())

            # Backward propagation
            train_mb_loss.backward()
            trainer.update()

        train_b_loss_ = np.mean(train_b_losses)

        # Valid
        # Create a new computation graph
        dy.renew_cg()
        associate_parameters(layers)

        valid_b_preds = []
        valid_b_losses = []

        for instance_x, instance_y in zip(valid_X, valid_y):
            t = dy.scalarInput(instance_y)
            y = f_props(layers, instance_x, train=False)

            valid_b_preds.append(binary_pred(y.value()))
            loss = dy.binary_log_loss(y, t)
            valid_b_losses.append(loss)

        valid_b_loss = dy.average(valid_b_losses)

        # Forward
        valid_b_loss_ = valid_b_loss.value()

        end_time = time.time()
        print('EPOCH: %d, Train Loss: %.3f (F1: %.3f, Ac: %.3f), Valid Loss: %.3f (F1: %.3f, Ac: %.3f), Time: %.3f[s]' % (
            epoch+1,
            train_b_loss_,
            f1_score(train_y, train_b_preds),
            accuracy_score(train_y, train_b_preds),
            valid_b_loss_,
            f1_score(valid_y, valid_b_preds),
            accuracy_score(valid_y, valid_b_preds),
            end_time-start_time
        ))

if __name__ == '__main__':
    main()
