import os
# import sys
import math
import time
import pickle
import argparse
from datetime import datetime

import gensim
import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from utils import build_w2c, build_w2i, build_dataset, build_batch, associate_parameters, forwards, sort_data_by_length, binary_pred, make_emb_zero, init_V
from layers import CNNText, Dense

RANDOM_SEED = 34
np.random.seed(RANDOM_SEED)

RESULTS_DIR = './results/' + datetime.now().strftime('%Y%m%d%H%M')
try:
    os.mkdir('results')
except:
    pass
os.mkdir(RESULTS_DIR)
# sys.stdout = open(os.path.join(RESULTS_DIR, 'output.txt'), 'w')

def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification in DyNet')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use. For cpu, set -1 [default: 0]')
    parser.add_argument('--train_x_path', type=str, default='./data/train_x.txt', help='File path of train x data [default: `./data/train_x.txt`]')
    parser.add_argument('--train_y_path', type=str, default='./data/train_y.txt', help='File path of train y data [default: `./data/train_x.txt`]')
    parser.add_argument('--valid_x_path', type=str, default='./data/valid_x.txt', help='File path of valid x data [default: `./data/valid_x.txt`]')
    parser.add_argument('--valid_y_path', type=str, default='./data/valid_y.txt', help='File path of valid y data [default: `./data/valid_y.txt`]')
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
    W2V_PATH     = './GoogleNews-vectors-negative300.bin'
    TRAIN_X_PATH = args.train_x_path
    TRAIN_Y_PATH = args.train_y_path
    VALID_X_PATH = args.valid_x_path
    VALID_Y_PATH = args.valid_y_path

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_random_seed(RANDOM_SEED)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Load pretrained embeddings
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
    vocab = pretrained_model.wv.vocab.keys()
    w2v = pretrained_model.wv

    # Build dataset =======================================================================================================
    w2c = build_w2c(TRAIN_X_PATH, vocab=vocab)
    w2i, i2w = build_w2i(TRAIN_X_PATH, w2c, unk='unk')
    train_x, train_y = build_dataset(TRAIN_X_PATH, TRAIN_Y_PATH, w2i, unk='unk')
    valid_x, valid_y = build_dataset(VALID_X_PATH, VALID_Y_PATH, w2i, unk='unk')

    train_x, train_y = sort_data_by_length(train_x, train_y)
    valid_x, valid_y = sort_data_by_length(valid_x, valid_y)

    VOCAB_SIZE = len(w2i)
    print('VOCAB_SIZE:', VOCAB_SIZE)

    V_init = init_V(w2v, w2i)

    with open(os.path.join(RESULTS_DIR, './w2i.dump'), 'wb') as f_w2i, open(os.path.join(RESULTS_DIR, './i2w.dump'), 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

    # Build model =================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    # V1
    V1 = model.add_lookup_parameters((VOCAB_SIZE, EMB_DIM))
    if V_STRATEGY in ['static', 'non-static', 'multichannel']:
        V1.init_from_array(V_init)
    if V_STRATEGY in ['static', 'multichannel']:
        V1_UPDATE = False
    else: # 'rand', 'non-static'
        V1_UPDATE = True
    make_emb_zero(V1, [w2i['<s>'], w2i['</s>']])

    # V2
    if V_STRATEGY == 'multichannel':
        V2 = model.add_lookup_parameters((VOCAB_SIZE, EMB_DIM))
        V2.init_from_array(V_init)
        V2_UPDATE = True
        make_emb_zero(V2, [w2i['<s>'], w2i['</s>']])

    layers = [
        CNNText(model, EMB_DIM, WIN_SIZES, NUM_CHA, NUM_FIL, dy.tanh, DROPOUT_PROB),
        Dense(model, 3*NUM_FIL, OUT_DIM, dy.logistic)
    ]

    # Train model ================================================================================
    n_batches_train = math.ceil(len(train_x)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_x)/BATCH_SIZE)

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
            x = build_batch(train_x[start:end], w2i, max(WIN_SIZES)).T
            t = np.array(train_y[start:end])

            sen_len = x.shape[0]

            if V_STRATEGY in ['rand', 'static', 'non-static']:
                x_embs = dy.concatenate_cols([dy.lookup_batch(V1, x_t, update=V1_UPDATE) for x_t in x])
                x_embs = dy.transpose(x_embs)
                x_embs = dy.reshape(x_embs, (sen_len, EMB_DIM, 1))
            else: # multichannel
                x_embs1 = dy.concatenate_cols([dy.lookup_batch(V1, x_t, update=V1_UPDATE) for x_t in x])
                x_embs2 = dy.concatenate_cols([dy.lookup_batch(V2, x_t, update=V2_UPDATE) for x_t in x])
                x_embs1 = dy.transpose(x_embs1)
                x_embs2 = dy.transpose(x_embs2)
                x_embs  = dy.concatenate([x_embs1, x_embs2], d=2)

            t = dy.inputTensor(t, batched=True)
            y = forwards(layers, x_embs, test=False)

            mb_loss = dy.mean_batches(dy.binary_log_loss(y, t))

            # Forward prop
            loss_all_train.append(mb_loss.value())
            pred_all_train.extend(list(binary_pred(y.npvalue().flatten())))

            # Backward prop
            mb_loss.backward()
            trainer.update()

            # L2 norm constraint
            layers[1].scale_W(L2_NORM_LIM)

            # Make padding embs zero
            if V_STRATEGY in ['rand', 'non-static']:
                make_emb_zero(V1, [w2i['<s>'], w2i['</s>']])
            elif V_STRATEGY in ['multichannel']:
                make_emb_zero(V2, [w2i['<s>'], w2i['</s>']])

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
            x = build_batch(valid_x[start:end], w2i, max(WIN_SIZES)).T
            t = np.array(valid_y[start:end])

            sen_len = x.shape[0]

            if V_STRATEGY in ['rand', 'static', 'non-static']:
                x_embs = dy.concatenate_cols([dy.lookup_batch(V1, x_t, update=V1_UPDATE) for x_t in x])
                x_embs = dy.transpose(x_embs)
                x_embs = dy.reshape(x_embs, (sen_len, EMB_DIM, 1))
            else: # multichannel
                x_embs1 = dy.concatenate_cols([dy.lookup_batch(V1, x_t, update=V1_UPDATE) for x_t in x])
                x_embs2 = dy.concatenate_cols([dy.lookup_batch(V2, x_t, update=V2_UPDATE) for x_t in x])
                x_embs1 = dy.transpose(x_embs1)
                x_embs2 = dy.transpose(x_embs2)
                x_embs  = dy.concatenate([x_embs1, x_embs2], d=2)

            t = dy.inputTensor(t, batched=True)
            y = forwards(layers, x_embs, test=True)

            mb_loss = dy.mean_batches(dy.binary_log_loss(y, t))

            # Forward prop
            loss_all_valid.append(mb_loss.value())
            pred_all_valid.extend(list(binary_pred(y.npvalue().flatten())))

        print('EPOCH: %d, Train Loss:: %.3f (F1:: %.3f, Acc:: %.3f), Valid Loss:: %.3f (F1:: %.3f, Acc:: %.3f), Time:: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all_train),
            f1_score(train_y, pred_all_train),
            accuracy_score(train_y, pred_all_train),
            np.mean(loss_all_valid),
            f1_score(valid_y, pred_all_valid),
            accuracy_score(valid_y, pred_all_valid),
            time.time()-start_time,
        ))

        # Save model =========================================================================================================================
        if V_STRATEGY in ['rand', 'static', 'non-static']:
            dy.save(os.path.join(RESULTS_DIR, './model_e'+str(epoch+1)), [V1] + layers)
        else:
            dy.save(os.path.join(RESULTS_DIR, './model_e'+str(epoch+1)), [V1, V2] + layers)

if __name__ == '__main__':
    main()
