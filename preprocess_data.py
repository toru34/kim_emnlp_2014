import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 34
np.random.seed(RANDOM_STATE)

def main():
    NEG_FILE = './rt-polaritydata/rt-polarity.neg'
    POS_FILE = './rt-polaritydata/rt-polarity.pos'

    TRAIN_X_FILE = './data/train_x.txt'
    TRAIN_Y_FILE = './data/train_y.txt'
    VALID_X_FILE = './data/valid_x.txt'
    VALID_Y_FILE = './data/valid_y.txt'

    with open(NEG_FILE, 'r', encoding='utf-8', errors='ignore') as f_neg, open(POS_FILE, 'r', encoding='utf-8', errors='ignore') as f_pos:
        data_neg = f_neg.read().split('\n')[:-1]
        data_pos = f_pos.read().split('\n')[:-1]

    data_X = data_neg + data_pos
    data_y = ['0']*len(data_neg) + ['1']*len(data_pos)

    train_X, valid_X, train_y, valid_y = train_test_split(data_X, data_y, test_size=0.1)

    train_X_txt = '\n'.join(train_X)
    train_y_txt = '\n'.join(train_y)
    valid_X_txt = '\n'.join(valid_X)
    valid_y_txt = '\n'.join(valid_y)

    with open(TRAIN_X_FILE, 'w') as f_tx, open(TRAIN_Y_FILE, 'w') as f_ty, open(VALID_X_FILE, 'w') as f_vx, open(VALID_Y_FILE, 'w') as f_vy:
        f_tx.write(train_X_txt)
        f_ty.write(train_y_txt)
        f_vx.write(valid_X_txt)
        f_vy.write(valid_y_txt)

if __name__ == '__main__':
    main()
