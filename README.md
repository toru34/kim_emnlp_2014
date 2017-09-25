## Convolutional Neural Networks for Sentence Classification

Unofficial DyNet implementation for the paper Convolutional Neural Networks for Sentence Classification (EMNLP 2014)[1].

### 1. Requirements
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- gensim 2.3.0+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

### 2. Prepare dataset
To get movie review data[2] and pretrained word embeddings[3], run
```
sh data_download.sh
```
and
```
python preprocess.py
```

### 3. Train
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `-1`]
- `--n_epochs`: Number of epochs [default: `25`]
- `--batch_size`: Mini batch size [default: `64`]
- `--win_sizes`: Window sizes of filters [default: `[3, 4, 5]`]
- `--num_fil`: Number of filters in each window size [default: `100`]
- `--s`: L2 norm constraint on w [default: `3.0`]
- `--dropout_prob`: Dropout probability [default: `0.5`]
- `--v_strategy`: Embeding strategy. [default: `non-static`]
    - `rand`: Random initialization.
    - `static`: Load pretrained embeddings and do not update during the training.
    - `non-static`: Load pretrained embeddings and update during the training.
    - `multichannel`: Load pretrained embeddings as two channels and update one of them during the training and do not update the other one.
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `4096`]

#### Command example
```
python train_autobatch.py --num_epochs 20
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `-1`]
- `--model_file`: Model to use for prediction [default: `./model`]
- `--input_file`: Input file path [default: `./data/valid_x.txt`]
- `--output_file`: Output file paht [default: `./pred_y.txt`]
- `--w2i_file`: Word2Index file path [default: `./w2i.dump`]
- `--i2w_file`: Index2Word file path [default: `./i2w.dump`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `1024`]

#### Command example
```
python test.py
```

### 5. Results
All examples below are trained with the default arguments except for `V_STRATEGY`. Adam optimizer is used (Original paper used Adadelta).

##### `V_STRATEGY`: static
```
EPOCH: 1, Train Loss: 0.595 (F1: 0.664, Acc: 0.668), Valid Loss: 0.493 (F1: 0.771, Acc: 0.767), Time: 13.062[s]
EPOCH: 2, Train Loss: 0.442 (F1: 0.793, Acc: 0.795), Valid Loss: 0.466 (F1: 0.788, Acc: 0.781), Time: 27.259[s]
EPOCH: 3, Train Loss: 0.361 (F1: 0.839, Acc: 0.840), Valid Loss: 0.455 (F1: 0.800, Acc: 0.796), Time: 41.446[s]
EPOCH: 4, Train Loss: 0.292 (F1: 0.882, Acc: 0.883), Valid Loss: 0.454 (F1: 0.818, Acc: 0.807), Time: 55.642[s]
EPOCH: 5, Train Loss: 0.223 (F1: 0.920, Acc: 0.921), Valid Loss: 0.457 (F1: 0.814, Acc: 0.807), Time: 69.843[s]
```
##### `V_STRATEGY`: non-static
```
EPOCH: 1, Train Loss: 0.579 (F1: 0.675, Acc: 0.678), Valid Loss: 0.468 (F1: 0.786, Acc: 0.782), Time: 1211.785[s]
EPOCH: 2, Train Loss: 0.350 (F1: 0.846, Acc: 0.847), Valid Loss: 0.456 (F1: 0.812, Acc: 0.803), Time: 2413.420[s]
EPOCH: 3, Train Loss: 0.200 (F1: 0.927, Acc: 0.928), Valid Loss: 0.495 (F1: 0.807, Acc: 0.806), Time: 3631.156[s]
EPOCH: 4, Train Loss: 0.098 (F1: 0.976, Acc: 0.976), Valid Loss: 0.574 (F1: 0.804, Acc: 0.799), Time: 4847.422[s]
EPOCH: 5, Train Loss: 0.043 (F1: 0.993, Acc: 0.993), Valid Loss: 0.689 (F1: 0.777, Acc: 0.784), Time: 6061.401[s]
```
##### `V_STRATEGY`: rand
```
```
##### `V_STRATEGY`: multichannel
```
```

### Notes

### Reference
- [1] Y. Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of EMNLP 2014 \[[pdf\]](https://arxiv.org/pdf/1408.5882.pdf)
- [2] B. Peng and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL \[[pdf\]](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf)
- [3] Google News corpus word vector \[[link\]](https://code.google.com/archive/p/word2vec/)
