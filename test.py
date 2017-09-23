import os
import pickle
import argparse

import _dynet as dy
from tqdm import tqdm

from utils import build_dataset, associate_parameters, f_props, binary_pred

def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--model_file', type=str, default='./model', help='Model to use for prediction [default: ./model]')
    parser.add_argument('--input_file', type=str, default='./data/valid_x.txt', help='Input file path [default: ./data/valid_x.txt]')
    parser.add_argument('--output_file', type=str, default='./pred_y.txt', help='Output file path [default: ./pred_y.txt]')
    parser.add_argument('--w2i_file', type=str, default='./w2i.dump', help='Word2Index file path [default: ./w2i.dump]')
    parser.add_argument('--i2w_file', type=str, default='./i2w.dump', help='Index2Word file path [default: ./i2w.dump]')
    parser.add_argument('--alloc_mem', type=int, default=1024, help='Amount of memory to allocate [mb] [default: 1024]')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

    # Load model ===============================================================================================================
    model = dy.Model()
    Vs_layers = dy.load(MODEL_FILE, model)
    layers = Vs_layers[-2:]
    MULTICHANNEL = layers[0].multichannel
    if MULTICHANNEL:
        V1, V2 = Vs_layers[0], Vs_layers[1]
    else:
        V = Vs_layers[0]
    WIN_SIZES = layers[0].win_sizes

    # Load test data ===========================================================================================================
    with open(W2I_FILE, 'rb') as f_w2i, open(I2W_FILE, 'rb') as f_i2w:
        w2i = pickle.load(f_w2i)
        i2w = pickle.load(f_i2w)

    test_X, _, _ = build_dataset(INPUT_FILE, w2i=w2i, unksym='unk')

    test_X = [[0]*max(WIN_SIZES) + instance_x + [0]*max(WIN_SIZES) for instance_x in test_X]

    # Pred =====================================================================================================================
    pred_y_txt = ''
    # for instance_x in tqdm(test_X):
    for instance_x in test_X:
        dy.renew_cg()
        associate_parameters(layers)

        if MULTICHANNEL:
            x_embs1 = [dy.lookup(V1, x_t) for x_t in instance_x]
            x_embs2 = [dy.lookup(V2, x_t) for x_t in instance_x]
            y = f_props(layers, [x_embs1, x_embs2], train=False)
        else:
            x_embs = [dy.lookup(V, x_t) for x_t in instance_x]
            y = f_props(layers, x_embs, train=False)

        pred_y_txt += str(int(binary_pred(y.value()))) + '\n'
        print(y.value())

    with open(OUTPUT_FILE, 'w') as f:
        f.write(pred_y_txt)

if __name__ == '__main__':
    main()
