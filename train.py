import os
import math
import time
import pickle
import argparse

import gensim
import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from utils import associate_parameters, binary_pred, build_word2count, build_dataset, sort_data_by_length, f_props, init_V
from layers import CNNText, Dense

RANDOM_STATE = 34
rng = np.random.RandomState(RANDOM_STATE)

def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs [default: 25]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--win_sizes', type=list, default=[2,3,4], help='Window sizes of filters [default: [2, 3, 4]]')
    parser.add_argument('--num_fil', type=int, default=100, help='Number of filters in each window size [default: 100]')
    parser.add_argument('--vocab_size', type=int, default=60000, help='Vocabulary size [default: 60000]')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability [default: 0.5]')
    parser.add_argument('--v_strategy', type=str, default='rand', help='Embedding strategy. rand: Random  initialization. static: Load pretrained embeddings and do not update during the training. non-static: Load pretrained embeddings and update during the training.')
    parser.add_argument('--alloc_mem', type=int, default=4096, help='Amount of memory to allocate [mb] [default: 4096]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    vocab_size = args.vocab_size
    N_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    WIN_SIZES = args.win_sizes
    EMB_DIM = 300
    OUT_DIM = 1
    NUM_FIL = args.num_fil
    DROPOUT_PROB = args.dropout_prob
    V_STRATEGY = args.v_strategy
    ALLOC_MEM = args.alloc_mem
    if V_STRATEGY is 'multichannel':
        MULTICHANNEL = True
    else:
        MULTICHANNEL = False

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

    # Build dataset =======================================================================================
    w2c = build_word2count(TRAIN_X_FILE)

    train_X, w2i, i2w = build_dataset(TRAIN_X_FILE, vocab_size=vocab_size, w2c=w2c, padid=False, unksym='unk')
    valid_X, _, _ = build_dataset(VALID_X_FILE, w2i=w2i, unksym='unk')

    with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

    train_X = [[0]*max(WIN_SIZES) + instance_x + [0]*max(WIN_SIZES) for instance_x in train_X]
    valid_X = [[0]*max(WIN_SIZES) + instance_x + [0]*max(WIN_SIZES) for instance_x in valid_X]

    with open(TRAIN_Y_FILE, 'r') as f_t, open(VALID_Y_FILE, 'r') as f_v:
        train_y = np.array(f_t.read().split('\n')).astype('int32')
        valid_y = np.array(f_v.read().split('\n')).astype('int32')

    train_X, train_y = sort_data_by_length(train_X, train_y)
    valid_X, valid_y = sort_data_by_length(valid_X, valid_y)

    vocab_size = len(w2i)

    # Load pretrained embeddings ==========================================================================
    if V_STRATEGY == 'rand':
        V_init = rng.normal(size=(vocab_size, EMB_DIM))
    else:
        pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(W2V_FILE, binary=True)
        w2v = pretrained_model.wv
        V_init = init_V(w2v, w2i, rng)

        import gc
        del pretrained_model
        gc.collect()

    # Build model =========================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    if V_STRATEGY in ['static', 'multichannel']:
        V_sta = model.add_lookup_parameters((vocab_size, EMB_DIM))
        V_sta.init_from_array(V_init)
    if V_STRATEGY in ['rand', 'non-static', 'multichannel']:
        V_non = model.add_lookup_parameters((vocab_size, EMB_DIM))
        V_non.init_from_array(V_init)

    layers = [
        CNNText(model, EMB_DIM, WIN_SIZES, NUM_FIL, dy.rectify, DROPOUT_PROB, multichannel=MULTICHANNEL),
        Dense(model, 3*NUM_FIL, OUT_DIM, dy.logistic)
    ]

    # Train model
    n_batches_train = math.ceil(len(train_X)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_X)/BATCH_SIZE)

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        # Train
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
                x_embss = []
                if V_STRATEGY in ['static', 'multichannel']:
                    x_embss.append([dy.lookup(V_sta, x_t, update=False) for x_t in instance_x])
                if V_STRATEGY in ['rand', 'non-static', 'multichannel']:
                    x_embss.append([dy.lookup(V_non, x_t, update=True) for x_t in instance_x])

                t = dy.scalarInput(instance_y)
                if MULTICHANNEL:
                    y = f_props(layers, x_embss, train=True)
                else:
                    y = f_props(layers, x_embss[0], train=True)

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
                x_embss = []
                if V_STRATEGY in ['static', 'multichannel']:
                    x_embss.append([dy.lookup(V_sta, x_t, update=False) for x_t in instance_x])
                if V_STRATEGY in ['rand', 'non-static', 'multichannel']:
                    x_embss.append([dy.lookup(V_non, x_t, update=True) for x_t in instance_x])

                t = dy.scalarInput(instance_y)
                if MULTICHANNEL:
                    y = f_props(layers, x_embss, train=True)
                else:
                    y = f_props(layers, x_embss[0], train=True)

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

        # Save model ==========================================================================================
        Vs = []
        if V_STRATEGY in ['static', 'multichannel']:
            Vs.append(V_sta)
        if V_STRATEGY in ['rand', 'non-static', 'multichannel']:
            Vs.append(V_non)
        dy.save('./model_epoch'+str(epoch), Vs + layers)

if __name__ == '__main__':
    main()
