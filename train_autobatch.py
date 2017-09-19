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
from sklearn.model_selection import train_test_split

from utils import associate_parameters, binary_pred, build_word2count, build_dataset, sort_data_by_length, f_props

RANDOM_STATE = 34
rng = np.random.RandomState(RANDOM_STATE)

def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs [default: 25]')
    parser.add_argument('--batch_size', type=int, default=50, help='Mini batch size [default: 32]')
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters in each window size [default: 100]')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size [default: 10000]')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability [default: 0.5]')
    parser.add_argument('--embedding_strategy', type=str, default='rand', help='Embedding strategy. rand: Random  initialization. static: Load pretrained embeddings and do not update during the training. non-static: Load pretrained embeddings and update during the training.')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size. (only applied to rand option) [default: 300]')
    parser.add_argument('--alloc_mem', type=int, default=4096, help='Amount of memory to allocate [mb] [default: 4096]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.gpu < 0:
        import _dynet as dy  # Use cpu
    else:
        import _gdynet as dy # Use gpu

    from layers import CNNText, Dense

    vocab_size = args.vocab_size
    N_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    EMB_DIM = args.emb_dim
    OUT_DIM = 1
    NUM_FIL = args.num_filters
    DROPOUT_PROB = args.dropout_prob
    V_STRATEGY = args.embedding_strategy
    ALLOC_MEM = args.alloc_mem

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Build dataset =============================================================================================
    if V_STRATEGY == 'rand':
        V_UPDATE = True
        w2c = build_word2count('./data/rt-polaritydata/rt-polarity.neg')
        w2c = build_word2count('./data/rt-polaritydata/rt-polarity.pos', w2c=w2c)
        data_neg, w2i, i2w = build_dataset('./data/rt-polaritydata/rt-polarity.neg', vocab_size=vocab_size, w2c=w2c, padid=True)
        data_pos, _, _ = build_dataset('./data/rt-polaritydata/rt-polarity.pos', w2i=w2i)

        V_init = None
    elif V_STRATEGY == 'static':
        V_UPDATE = False
        EMB_DIM = 300
        pretrained_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        vocab = pretrained_model.wv.vocab.keys()

        w2c = build_word2count('./data/rt-polaritydata/rt-polarity.neg', vocab=vocab)
        w2c = build_word2count('./data/rt-polaritydata/rt-polarity.pos', w2c=w2c, vocab=vocab)
        data_neg, w2i, i2w = build_dataset('./data/rt-polaritydata/rt-polarity.neg', vocab_size=vocab_size, w2c=w2c, padid=True, unksym='unk')
        data_pos, _, _ = build_dataset('./data/rt-polaritydata/rt-polarity.pos', w2i=w2i, unksym='unk')

        V_init = np.array([np.zeros(EMB_DIM) if w == '<pad>' else pretrained_model[w] for w in w2i.keys()])

        import gc
        del pretrained_model
        gc.collect()
    elif V_STRATEGY == 'non-static':
        V_UPDATE = True
        EMB_DIM = 300
        pretrained_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        vocab = pretrained_model.wv.vocab.keys()

        w2c = build_word2count('./data/rt-polaritydata/rt-polarity.neg')
        w2c = build_word2count('./data/rt-polaritydata/rt-polarity.pos', w2c=w2c)
        data_neg, w2i, i2w = build_dataset('./data/rt-polaritydata/rt-polarity.neg', vocab_size=vocab_size, w2c=w2c, padid=True, unksym='unk')
        data_pos, _, _ = build_dataset('./data/rt-polaritydata/rt-polarity.pos', w2i=w2i, unksym='unk')

        V_init = np.array([pretrained_model[w] if (w in vocab) else rng.normal(size=(EMB_DIM)) for w in w2i.keys()])
        V_init[w2i['<pad>']] = np.zeros(EMB_DIM)

        import gc
        del pretrained_model
        gc.collect()

    vocab_size = len(w2i)

    data_X = data_neg + data_pos
    data_X = [[0,0,0,0] + instance_x + [0,0,0,0] for instance_x in data_X]
    data_y = [0 for i in range(len(data_neg))] + [1 for i in range(len(data_pos))]
    data_X, data_y = shuffle(data_X, data_y, random_state=RANDOM_STATE)

    train_X, valid_X, train_y, valid_y = train_test_split(data_X, data_y, test_size=0.1, random_state=RANDOM_STATE)
    train_X, train_y = sort_data_by_length(train_X, train_y)
    valid_X, valid_y = sort_data_by_length(valid_X, valid_y)

    # Build model
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    V = model.add_lookup_parameters((vocab_size, EMB_DIM))
    if V_init is not None:
        V.init_from_array(V_init)
    else:
        V.init_row(w2i['<pad>'], np.zeros(EMB_DIM))

    layers = [
        CNNText(model, EMB_DIM, NUM_FIL, dy.rectify, DROPOUT_PROB),
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
                x_embs = [dy.lookup(V, x_t, V_UPDATE) for x_t in instance_x]

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
                x_embs = [dy.lookup(V, x_t) for x_t in instance_x]

                t = dy.scalarInput(instance_y)
                y = f_props(layers, x_embs, train=True)

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
    dy.save('./model', [V] + layers)
    with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

if __name__ == '__main__':
    main()
