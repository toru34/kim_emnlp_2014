import os
import math
import time
import pickle
import argparse

import gensim
import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score

from utils import associate_parameters, binary_pred, build_word2count, build_dataset, f_props, init_V
from layers import Dense, CNNText

RANDOM_STATE = 34
np.random.seed(RANDOM_STATE)

def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--train_x_file', type=str, default='./data/train_x.txt', help='File path of train x data [default: `./data/train_x.txt`]')
    parser.add_argument('--train_y_file', type=str, default='./data/train_y.txt', help='File path of train y data [default: `./data/train_x.txt`]')
    parser.add_argument('--valid_x_file', type=str, default='./data/valid_x.txt', help='File path of valid x data [default: `./data/valid_x.txt`]')
    parser.add_argument('--valid_y_file', type=str, default='./data/valid_y.txt', help='File path of valid y data [default: `./data/valid_y.txt`]')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 10]')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 64]')
    parser.add_argument('--win_sizes', type=int, nargs='*', default=[3, 4, 5], help='Window sizes of filters [default: [3, 4, 5]]')
    parser.add_argument('--num_fil', type=int, default=100, help='Number of filters in each window size [default: 100]')
    parser.add_argument('--s', type=float, default=3.0, help='L2 norm constraint on w [default: 3.0]')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability [default: 0.5]')
    parser.add_argument('--v_strategy', type=str, default='static', help='Embedding strategy. rand: Random  initialization. static: Load pretrained embeddings and do not update during the training. non-static: Load pretrained embeddings and update during the training. [default: static]')
    parser.add_argument('--alloc_mem', type=int, default=4096, help='Amount of memory to allocate [mb] [default: 4096]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    N_EPOCHS = args.n_epochs
    WIN_SIZES = args.win_sizes
    BATCH_SIZE = args.batch_size
    EMB_DIM = 300
    OUT_DIM = 1
    L2_NORM_LIM = args.s
    NUM_FIL = args.num_fil
    DROPOUT_PROB = args.dropout_prob
    V_STRATEGY = args.v_strategy
    ALLOC_MEM = args.alloc_mem
    if V_STRATEGY in ['rand', 'static', 'non-static']:
        NUM_CHA = 1
    else:
        NUM_CHA = 2

    # FILE paths
    W2V_FILE = './GoogleNews-vectors-negative300.bin'
    TRAIN_X_FILE = './data/train_x.txt'
    TRAIN_Y_FILE = './data/train_y.txt'
    VALID_X_FILE = './data/valid_x.txt'
    VALID_Y_FILE = './data/valid_y.txt'

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Load pretrained embeddings
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(W2V_FILE, binary=True)
    vocab = pretrained_model.wv.vocab.keys()
    w2v = pretrained_model.wv

    # Build dataset
    w2c = build_word2count(TRAIN_X_FILE, vocab=vocab)
    train_X, w2i, i2w = build_dataset(TRAIN_X_FILE, w2c=w2c, padid=False, unksym='unk')
    train_y = np.loadtxt(TRAIN_Y_FILE)
    valid_X, _, _ = build_dataset(VALID_X_FILE, w2i=w2i, unksym='unk')
    valid_y = np.loadtxt(VALID_Y_FILE)

    max_win = max(WIN_SIZES)
    train_X = [[0]*max(WIN_SIZES) + instance_x + [0]*max(WIN_SIZES) for instance_x in train_X]
    valid_X = [[0]*max(WIN_SIZES) + instance_x + [0]*max(WIN_SIZES) for instance_x in valid_X]

    vocab_size = len(w2i)

    V_init = init_V(w2v, w2i)

    with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

    # Build model
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    # V1
    V1 = model.add_lookup_parameters((vocab_size, EMB_DIM))
    if V_STRATEGY in ['static', 'non-static', 'multichannel']:
        V1.init_from_array(V_init)
    if V_STRATEGY in ['static', 'multichannel']:
        V1_UPDATE = False
    else: # 'rand', 'non-static'
        V1_UPDATE = True

    # V2
    if V_STRATEGY == 'multichannel':
        V2 = model.add_lookup_parameters((vocab_size, EMB_DIM))
        V2.init_from_array(V_init)
        V2_UPDATE = True

    layers = [
        CNNText(model, EMB_DIM, WIN_SIZES, NUM_CHA, NUM_FIL, dy.tanh, DROPOUT_PROB),
        Dense(model, 3*NUM_FIL, OUT_DIM, dy.logistic)
    ]

    # Train model
    n_batches_train = math.ceil(len(train_X)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_X)/BATCH_SIZE)

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        # Train
        train_X, train_y = shuffle(train_X, train_y)
        loss_all_train = []
        pred_all_train = []
        for i in tqdm(range(n_batches_train)):
            # Create a new computation graph
            dy.renew_cg()
            associate_parameters(layers)

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            train_X_mb = train_X[start:end]
            train_y_mb = train_y[start:end]

            losses = []
            preds = []
            for instance_x, instance_y in zip(train_X_mb, train_y_mb):
                sen_len = len(instance_x)

                if V_STRATEGY in ['rand', 'static', 'non-static']:
                    x_embs = dy.concatenate([dy.lookup(V1, x_t, update=V1_UPDATE) for x_t in instance_x], d=1)
                    x_embs = dy.transpose(x_embs)
                    x_embs = dy.reshape(x_embs, (sen_len, EMB_DIM, 1))
                else: # 'multichannel'
                    x_embs1 = dy.concatenate([dy.lookup(V1, x_t, update=V2_UPDATE) for x_t in instance_x], d=1)
                    x_embs2 = dy.concatenate([dy.lookup(V2, x_t, update=V2_UPDATE) for x_t in instance_x], d=1)
                    x_embs1 = dy.transpose(x_embs1)
                    x_embs2 = dy.transpose(x_embs2)
                    x_embs = dy.concatenate([x_embs1, x_embs2], d=2)

                t = dy.scalarInput(instance_y)
                y = f_props(layers, x_embs, train=True)

                loss = dy.binary_log_loss(y, t)
                losses.append(loss)
                preds.append(y)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all_train.append(mb_loss.value())
            pred_all_train.extend(binary_pred(dy.concatenate_to_batch(preds).npvalue()).flatten().tolist())

            # Backward propagation
            mb_loss.backward()
            trainer.update()

            # L2 norm constraint
            layers[1].scale_W(L2_NORM_LIM)

        # Valid
        loss_all_valid = []
        pred_all_valid = []
        for i in range(n_batches_valid):
            # Create a new computation graph
            dy.renew_cg()
            associate_parameters(layers)

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            valid_X_mb = valid_X[start:end]
            valid_y_mb = valid_y[start:end]

            losses = []
            preds = []
            for instance_x, instance_y in zip(valid_X_mb, valid_y_mb):
                sen_len = len(instance_x)

                if V_STRATEGY in ['rand', 'static', 'non-static']:
                    x_embs = dy.concatenate([dy.lookup(V1, x_t, update=False) for x_t in instance_x], d=1)
                    x_embs = dy.transpose(x_embs)
                    x_embs = dy.reshape(x_embs, (sen_len, EMB_DIM, 1))
                else: # 'multichannel'
                    x_embs1 = dy.concatenate([dy.lookup(V2, x_t, update=False) for x_t in instance_x], d=1)
                    x_embs2 = dy.concatenate([dy.lookup(V2, x_t, update=False) for x_t in instance_x], d=1)
                    x_embs1 = dy.transpose(x_embs1)
                    x_embs2 = dy.transpose(x_embs2)
                    x_embs = dy.concatenate([x_embs1, x_embs2], d=2)

                t = dy.scalarInput(instance_y)
                y = f_props(layers, x_embs, train=False)

                loss = dy.binary_log_loss(y, t)
                losses.append(loss)
                preds.append(y)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all_valid.append(mb_loss.value())
            pred_all_valid.extend(binary_pred(dy.concatenate_to_batch(preds).npvalue()).flatten().tolist())

        print('EPOCH: %d, Train Loss: %.3f (F1: %.3f, Acc: %.3f), Valid Loss: %.3f (F1: %.3f, Acc: %.3f), Time: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all_train),
            f1_score(train_y, pred_all_train),
            accuracy_score(train_y, pred_all_train),
            np.mean(loss_all_valid),
            f1_score(valid_y, pred_all_valid),
            accuracy_score(valid_y, pred_all_valid),
            time.time()-start_time,
        ))

    # Save model
    if V_STRATEGY in ['rand', 'static', 'non-static']:
        dy.save('./model', [V1] + layers)
    else:
        dy.save('./model', [V1, V2] + layers)

if __name__ == '__main__':
    main()
