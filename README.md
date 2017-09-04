## Convolutional Neural Networks for Sentence Classification

DyNet implementation for the paper Convolutional Neural Networks for Sentence Classification (EMNLP 2014).

### Requirement
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- scikit-learn 0.19.0+

### Arguments
- `--num_epochs`: Number of epochs for training [default: 3]
- `--batch_size`: Batch size for training [default: 32]
- `--num_filters`: Number of filters in each window size [default: 20]
- `--emb_dim`: Embedding size for each word [default: 64]
- `--vocab_size`: Vocabulary size [default: 30000]

### How to run
```
python main.py --num_epochs 10 --batch_size 64 --num_filters 100 --emb_dim 128 --vocab_size 20000
```

### References
- Y. Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of EMNLP 2014 (original paper)
- B. Pang and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL (dataset used in this implementation)
