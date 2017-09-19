import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from utils import build_dataset, associate_parameters, f_props, binary_pred

def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--model_file', type=str, default='./model', help='Model to use for prediction [default: ./model]')
    parser.add_argument('--input_file', type=str, default='./data/rt-polaritydata/rt-polarity.neg', help='Input file path [default: ./data/rt-polaritydata/rt-polarity.neg]')
    parser.add_argument('--output_file', type=str, default='./pred.txt', help='Output file path [default: ./pred.txt]')
    parser.add_argument('--w2i_file', type=str, default='./w2i.dump', help='Word2Index file path [default: ./w2i.dump]')
    parser.add_argument('--i2w_file', type=str, default='./i2w.dump', help='Index2Word file path [default: ./i2w.dump]')
    parser.add_argument('--alloc_mem', type=int, default=1024, help='Amount of memory to allocate [mb] [default: 1024]')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.gpu < 0:
        import _dynet as dy  # Use cpu
    else:
        import _gdynet as dy # Use gpu

    MODEL_FILE = args.model_file
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    W2I_FILE = args.w2i_file
    I2W_FILE = args.i2w_file
    ALLOC_MEM = args.alloc_mem

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Load model
    with open(W2I_FILE, 'rb') as f_w2i, open(I2W_FILE, 'rb') as f_i2w:
        w2i = pickle.load(f_w2i)
        i2w = pickle.load(f_i2w)

    test_X, _, _ = build_dataset(INPUT_FILE, w2i=w2i)
    test_X = [[0,0,0,0] + instance_x + [0,0,0,0] for instance_x in test_X]

    model = dy.Model()
    V_layers = dy.load(MODEL_FILE, model)
    V, layers = V_layers[0], V_layers[1:]

    # Prediction
    pred_txt = ''
    for instance_x in tqdm(test_X):
        dy.renew_cg()
        associate_parameters(layers)

        x_embs = [dy.lookup(V, x_t) for x_t in instance_x]

        y = f_props(layers, x_embs, train=False)

        pred_txt += str(int(binary_pred(y.value()))) + '\n'

    with open(OUTPUT_FILE, 'w') as f:
        f.write(pred_txt)

if __name__ == '__main__':
    main()
