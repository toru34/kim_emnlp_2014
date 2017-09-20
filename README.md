## Convolutional Neural Networks for Sentence Classification

Unofficial DyNet implementation for the paper Convolutional Neural Networks for Sentence Classification (EMNLP 2014)[1].

### 1. Requirements
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- Keras 2.0.6+
- gensim 2.3.0+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

### 2. Prepare dataset
To get movie review data[2] and pretrained word embeddings[3], run
```
sh data_download.sh
```
.

### 3. Train
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set -1 [default: -1]
- `--n_epochs`: Number of epochs [default: 25]
- `--batch_size`: Mini batch size [default: 32]
- `--num_filters`: Number of filters in each window size [default: 100]
- `--vocab_size`: Vocabulary size [default: 10000]
- `--dropout_prob`: Dropout probability [default: 0.5]
- `--embedding_strategy`: Embeding strategy. [default: rand]
    - `rand`: Random initialization.
    - `static`: Load pretrained embeddings and do not update during the training.
    - `non-static`: Load pretrained embeddings and update during the training.
- `--emb_dim`: Word embedding size. (Only applied to rand option) [default: 300]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: 4096]

#### Command example
```
python train_manualbatch.py --num_epochs 20
```
Replace `train_manualbatch.py` with `train_autobatch.py` to use autobatching.

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set -1 [default: -1]
- `--model_file`: Model to use for prediction [default: ./model]
- `--input_file`: Input file path [default: ./data/rt-polaritydata/rt-polarity.neg]
- `--output_file`: Output file paht [default: ./pred.txt]
- `--w2i_file`: Word2Index file path [default: ./w2i.dump]
- `--i2w_file`: Index2Word file path [default: ./i2w.dump]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: 1024]

#### Command example
```
python test.py
```

### 5. Results
Work in progress

### Notes

### Reference
- [1] Y. Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of EMNLP 2014 \[[pdf\]](https://arxiv.org/pdf/1408.5882.pdf)
- [2] B. Peng and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL \[[pdf\]](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf)
- [3] Google News corpus word vector \[[link\]](https://code.google.com/archive/p/word2vec/)
