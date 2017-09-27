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

If you use your own dataset, please specify the paths of train and valid data files with command-line arguments described below.

### 3. Train
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `-1`]
- `--train_x_file`: File path of train x data [default: `./data/train_x.txt`]
- `--train_y_file`: File path of train y data [default: `./data/train_y.txt`]
- `--valid_x_file`: File path of valid x data [default: `./data/valid_x.txt`]
- `--valid_y_file`: File path of valid y data [default: `./data/valid_y.txt`]
- `--n_epochs`: Number of epochs [default: `10`]
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
python train.py --num_epochs 20
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `-1`]
- `--model_file`: Model to use for prediction [default: `./model`]
- `--input_file`: Input file path [default: `./data/valid_x.txt`]
- `--output_file`: Output file path [default: `./pred_y.txt`]
- `--w2i_file`: Word2Index file path [default: `./w2i.dump`]
- `--i2w_file`: Index2Word file path [default: `./i2w.dump`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `1024`]

#### Command example
```
python test.py
```

### 5. Results
All examples below are trained with the default arguments except for `v_strategy`.
##### `v_strategy`: static
```
EPOCH: 1, Train Loss: 5.329 (F1: 0.625, Acc: 0.619), Valid Loss: 0.524 (F1: 0.765, Acc: 0.738), Time: 22.393[s]
EPOCH: 2, Train Loss: 0.528 (F1: 0.744, Acc: 0.744), Valid Loss: 0.535 (F1: 0.777, Acc: 0.739), Time: 46.321[s]
EPOCH: 3, Train Loss: 0.464 (F1: 0.782, Acc: 0.782), Valid Loss: 0.834 (F1: 0.496, Acc: 0.648), Time: 70.446[s]
EPOCH: 4, Train Loss: 0.432 (F1: 0.803, Acc: 0.805), Valid Loss: 0.514 (F1: 0.796, Acc: 0.766), Time: 94.648[s]
EPOCH: 5, Train Loss: 0.389 (F1: 0.829, Acc: 0.829), Valid Loss: 0.538 (F1: 0.790, Acc: 0.764), Time: 118.901[s]
EPOCH: 6, Train Loss: 0.379 (F1: 0.830, Acc: 0.831), Valid Loss: 0.695 (F1: 0.761, Acc: 0.693), Time: 143.136[s]
EPOCH: 7, Train Loss: 0.324 (F1: 0.864, Acc: 0.865), Valid Loss: 0.515 (F1: 0.780, Acc: 0.792), Time: 167.331[s]
EPOCH: 8, Train Loss: 0.272 (F1: 0.884, Acc: 0.884), Valid Loss: 0.936 (F1: 0.526, Acc: 0.662), Time: 191.525[s]
EPOCH: 9, Train Loss: 0.222 (F1: 0.913, Acc: 0.914), Valid Loss: 0.512 (F1: 0.786, Acc: 0.799), Time: 215.742[s]
EPOCH: 10, Train Loss: 0.191 (F1: 0.923, Acc: 0.923), Valid Loss: 0.879 (F1: 0.753, Acc: 0.675), Time: 239.964[s]
EPOCH: 11, Train Loss: 0.138 (F1: 0.956, Acc: 0.956), Valid Loss: 0.538 (F1: 0.805, Acc: 0.781), Time: 264.237[s]
EPOCH: 12, Train Loss: 0.104 (F1: 0.974, Acc: 0.974), Valid Loss: 0.544 (F1: 0.806, Acc: 0.781), Time: 288.519[s]
EPOCH: 13, Train Loss: 0.089 (F1: 0.981, Acc: 0.981), Valid Loss: 0.501 (F1: 0.809, Acc: 0.807), Time: 312.741[s]
EPOCH: 14, Train Loss: 0.081 (F1: 0.985, Acc: 0.985), Valid Loss: 0.508 (F1: 0.793, Acc: 0.800), Time: 337.279[s]
EPOCH: 15, Train Loss: 0.073 (F1: 0.987, Acc: 0.987), Valid Loss: 0.498 (F1: 0.809, Acc: 0.800), Time: 361.525[s]
EPOCH: 16, Train Loss: 0.065 (F1: 0.991, Acc: 0.991), Valid Loss: 0.499 (F1: 0.808, Acc: 0.809), Time: 385.839[s]
EPOCH: 17, Train Loss: 0.063 (F1: 0.993, Acc: 0.993), Valid Loss: 0.501 (F1: 0.812, Acc: 0.811), Time: 410.181[s]
EPOCH: 18, Train Loss: 0.057 (F1: 0.994, Acc: 0.994), Valid Loss: 0.509 (F1: 0.819, Acc: 0.810), Time: 434.461[s]
EPOCH: 19, Train Loss: 0.053 (F1: 0.995, Acc: 0.995), Valid Loss: 0.508 (F1: 0.808, Acc: 0.810), Time: 458.751[s]
EPOCH: 20, Train Loss: 0.051 (F1: 0.995, Acc: 0.995), Valid Loss: 0.537 (F1: 0.791, Acc: 0.800), Time: 483.087[s]
```
##### `v_strategy`: non-static
```
EPOCH: 1, Train Loss: 3.498 (F1: 0.598, Acc: 0.599), Valid Loss: 0.520 (F1: 0.767, Acc: 0.754), Time: 1748.334[s]
EPOCH: 2, Train Loss: 0.543 (F1: 0.733, Acc: 0.734), Valid Loss: 0.510 (F1: 0.791, Acc: 0.756), Time: 3529.542[s]
EPOCH: 3, Train Loss: 0.435 (F1: 0.803, Acc: 0.804), Valid Loss: 0.698 (F1: 0.608, Acc: 0.696), Time: 5304.123[s]
EPOCH: 4, Train Loss: 0.353 (F1: 0.848, Acc: 0.848), Valid Loss: 0.591 (F1: 0.788, Acc: 0.747), Time: 7084.194[s]
EPOCH: 5, Train Loss: 0.268 (F1: 0.885, Acc: 0.885), Valid Loss: 0.567 (F1: 0.794, Acc: 0.770), Time: 8861.466[s]
EPOCH: 6, Train Loss: 0.144 (F1: 0.948, Acc: 0.948), Valid Loss: 0.755 (F1: 0.776, Acc: 0.724), Time: 10642.765[s]
EPOCH: 7, Train Loss: 0.098 (F1: 0.972, Acc: 0.972), Valid Loss: 0.529 (F1: 0.805, Acc: 0.806), Time: 12421.859[s]
EPOCH: 8, Train Loss: 0.068 (F1: 0.987, Acc: 0.987), Valid Loss: 0.558 (F1: 0.789, Acc: 0.798), Time: 14201.918[s]
EPOCH: 9, Train Loss: 0.052 (F1: 0.992, Acc: 0.992), Valid Loss: 0.559 (F1: 0.804, Acc: 0.794), Time: 15980.530[s]
EPOCH: 10, Train Loss: 0.043 (F1: 0.994, Acc: 0.994), Valid Loss: 0.568 (F1: 0.805, Acc: 0.797), Time: 17761.415[s]
EPOCH: 11, Train Loss: 0.035 (F1: 0.997, Acc: 0.997), Valid Loss: 0.579 (F1: 0.797, Acc: 0.791), Time: 19540.847[s]
EPOCH: 12, Train Loss: 0.030 (F1: 0.997, Acc: 0.997), Valid Loss: 0.592 (F1: 0.801, Acc: 0.796), Time: 21319.278[s]
EPOCH: 13, Train Loss: 0.026 (F1: 0.998, Acc: 0.998), Valid Loss: 0.597 (F1: 0.803, Acc: 0.801), Time: 23100.210[s]
EPOCH: 14, Train Loss: 0.023 (F1: 0.998, Acc: 0.998), Valid Loss: 0.601 (F1: 0.795, Acc: 0.797), Time: 24881.060[s]
EPOCH: 15, Train Loss: 0.021 (F1: 0.999, Acc: 0.999), Valid Loss: 0.613 (F1: 0.805, Acc: 0.799), Time: 26665.062[s]
EPOCH: 16, Train Loss: 0.018 (F1: 0.999, Acc: 0.999), Valid Loss: 0.622 (F1: 0.807, Acc: 0.801), Time: 28449.734[s]
EPOCH: 17, Train Loss: 0.017 (F1: 1.000, Acc: 1.000), Valid Loss: 0.623 (F1: 0.800, Acc: 0.798), Time: 30231.646[s]
EPOCH: 18, Train Loss: 0.016 (F1: 0.999, Acc: 0.999), Valid Loss: 0.631 (F1: 0.798, Acc: 0.795), Time: 32015.808[s]
EPOCH: 19, Train Loss: 0.015 (F1: 0.999, Acc: 0.999), Valid Loss: 0.634 (F1: 0.803, Acc: 0.799), Time: 33801.406[s]
EPOCH: 20, Train Loss: 0.014 (F1: 0.999, Acc: 0.999), Valid Loss: 0.655 (F1: 0.794, Acc: 0.796), Time: 35583.942[s]
```
##### `v_strategy`: rand
```
EPOCH: 1, Train Loss: 0.786 (F1: 0.564, Acc: 0.556), Valid Loss: 0.633 (F1: 0.671, Acc: 0.647), Time: 1745.219[s]
EPOCH: 2, Train Loss: 0.623 (F1: 0.654, Acc: 0.653), Valid Loss: 0.607 (F1: 0.725, Acc: 0.658), Time: 3508.173[s]
EPOCH: 3, Train Loss: 0.532 (F1: 0.746, Acc: 0.745), Valid Loss: 0.719 (F1: 0.455, Acc: 0.625), Time: 5273.342[s]
EPOCH: 4, Train Loss: 0.414 (F1: 0.809, Acc: 0.809), Valid Loss: 0.639 (F1: 0.752, Acc: 0.689), Time: 7037.384[s]
EPOCH: 5, Train Loss: 0.285 (F1: 0.886, Acc: 0.886), Valid Loss: 0.524 (F1: 0.759, Acc: 0.747), Time: 8800.880[s]
EPOCH: 6, Train Loss: 0.184 (F1: 0.947, Acc: 0.947), Valid Loss: 0.546 (F1: 0.756, Acc: 0.745), Time: 10566.495[s]
EPOCH: 7, Train Loss: 0.136 (F1: 0.969, Acc: 0.969), Valid Loss: 0.549 (F1: 0.745, Acc: 0.750), Time: 12334.440[s]
EPOCH: 8, Train Loss: 0.098 (F1: 0.985, Acc: 0.985), Valid Loss: 0.570 (F1: 0.751, Acc: 0.754), Time: 14100.293[s]
EPOCH: 9, Train Loss: 0.074 (F1: 0.993, Acc: 0.993), Valid Loss: 0.578 (F1: 0.766, Acc: 0.751), Time: 15866.979[s]
EPOCH: 10, Train Loss: 0.061 (F1: 0.994, Acc: 0.994), Valid Loss: 0.585 (F1: 0.752, Acc: 0.754), Time: 17630.276[s]
EPOCH: 11, Train Loss: 0.050 (F1: 0.997, Acc: 0.997), Valid Loss: 0.599 (F1: 0.755, Acc: 0.754), Time: 19392.854[s]
EPOCH: 12, Train Loss: 0.042 (F1: 0.998, Acc: 0.998), Valid Loss: 0.609 (F1: 0.757, Acc: 0.753), Time: 21154.998[s]
EPOCH: 13, Train Loss: 0.037 (F1: 0.998, Acc: 0.998), Valid Loss: 0.623 (F1: 0.750, Acc: 0.752), Time: 22916.443[s]
EPOCH: 14, Train Loss: 0.032 (F1: 0.999, Acc: 0.999), Valid Loss: 0.634 (F1: 0.763, Acc: 0.760), Time: 24679.292[s]
EPOCH: 15, Train Loss: 0.029 (F1: 0.999, Acc: 0.999), Valid Loss: 0.631 (F1: 0.761, Acc: 0.753), Time: 26442.437[s]
EPOCH: 16, Train Loss: 0.024 (F1: 0.999, Acc: 0.999), Valid Loss: 0.652 (F1: 0.754, Acc: 0.752), Time: 28202.414[s]
EPOCH: 17, Train Loss: 0.023 (F1: 0.999, Acc: 0.999), Valid Loss: 0.671 (F1: 0.745, Acc: 0.747), Time: 29960.799[s]
EPOCH: 18, Train Loss: 0.021 (F1: 0.999, Acc: 0.999), Valid Loss: 0.666 (F1: 0.765, Acc: 0.755), Time: 31720.556[s]
EPOCH: 19, Train Loss: 0.020 (F1: 0.999, Acc: 0.999), Valid Loss: 0.680 (F1: 0.757, Acc: 0.753), Time: 33481.964[s]
EPOCH: 20, Train Loss: 0.017 (F1: 0.999, Acc: 0.999), Valid Loss: 0.701 (F1: 0.742, Acc: 0.747), Time: 35243.052[s]
```
##### `v_strategy`: multichannel
```
EPOCH: 1, Train Loss: 14.700 (F1: 0.592, Acc: 0.543), Valid Loss: 0.570 (F1: 0.669, Acc: 0.707), Time: 1205.823[s]
EPOCH: 2, Train Loss: 0.986 (F1: 0.653, Acc: 0.656), Valid Loss: 0.547 (F1: 0.712, Acc: 0.747), Time: 2414.326[s]
EPOCH: 3, Train Loss: 0.530 (F1: 0.769, Acc: 0.770), Valid Loss: 0.629 (F1: 0.692, Acc: 0.739), Time: 3619.243[s]
EPOCH: 4, Train Loss: 0.534 (F1: 0.787, Acc: 0.788), Valid Loss: 0.536 (F1: 0.802, Acc: 0.773), Time: 4824.759[s]
EPOCH: 5, Train Loss: 0.392 (F1: 0.842, Acc: 0.843), Valid Loss: 0.508 (F1: 0.780, Acc: 0.784), Time: 6030.836[s]
EPOCH: 6, Train Loss: 0.287 (F1: 0.887, Acc: 0.887), Valid Loss: 0.822 (F1: 0.773, Acc: 0.719), Time: 7236.089[s]
EPOCH: 7, Train Loss: 0.153 (F1: 0.941, Acc: 0.941), Valid Loss: 0.577 (F1: 0.791, Acc: 0.799), Time: 8440.763[s]
EPOCH: 8, Train Loss: 0.092 (F1: 0.967, Acc: 0.967), Valid Loss: 0.604 (F1: 0.781, Acc: 0.792), Time: 9647.744[s]
EPOCH: 9, Train Loss: 0.062 (F1: 0.983, Acc: 0.983), Valid Loss: 0.581 (F1: 0.806, Acc: 0.795), Time: 10853.264[s]
EPOCH: 10, Train Loss: 0.046 (F1: 0.989, Acc: 0.989), Valid Loss: 0.595 (F1: 0.812, Acc: 0.799), Time: 12060.162[s]
EPOCH: 11, Train Loss: 0.035 (F1: 0.992, Acc: 0.992), Valid Loss: 0.601 (F1: 0.810, Acc: 0.803), Time: 13267.643[s]
EPOCH: 12, Train Loss: 0.027 (F1: 0.997, Acc: 0.997), Valid Loss: 0.604 (F1: 0.816, Acc: 0.813), Time: 14475.249[s]
EPOCH: 13, Train Loss: 0.023 (F1: 0.997, Acc: 0.997), Valid Loss: 0.619 (F1: 0.805, Acc: 0.805), Time: 15680.489[s]
EPOCH: 14, Train Loss: 0.019 (F1: 0.997, Acc: 0.997), Valid Loss: 0.621 (F1: 0.811, Acc: 0.809), Time: 16885.096[s]
EPOCH: 15, Train Loss: 0.019 (F1: 0.998, Acc: 0.998), Valid Loss: 0.617 (F1: 0.807, Acc: 0.802), Time: 18089.870[s]
EPOCH: 16, Train Loss: 0.015 (F1: 0.999, Acc: 0.999), Valid Loss: 0.642 (F1: 0.802, Acc: 0.802), Time: 19295.037[s]
EPOCH: 17, Train Loss: 0.013 (F1: 0.999, Acc: 0.999), Valid Loss: 0.632 (F1: 0.809, Acc: 0.806), Time: 20500.711[s]
EPOCH: 18, Train Loss: 0.014 (F1: 0.999, Acc: 0.999), Valid Loss: 0.634 (F1: 0.817, Acc: 0.810), Time: 21704.948[s]
EPOCH: 19, Train Loss: 0.012 (F1: 0.999, Acc: 0.999), Valid Loss: 0.640 (F1: 0.810, Acc: 0.806), Time: 22909.151[s]
EPOCH: 20, Train Loss: 0.012 (F1: 0.999, Acc: 0.999), Valid Loss: 0.648 (F1: 0.808, Acc: 0.803), Time: 24114.725[s]
```

### Notes
- All experiments are done with GeForce GTX 1060 (6GB).
- Adadelta optimizer is used in all experiments above, but you can use Adam for fast convergence.

### Reference
- [1] Y. Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of EMNLP 2014 \[[pdf\]](https://arxiv.org/pdf/1408.5882.pdf)
- [2] B. Peng and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL \[[pdf\]](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf)
- [3] Google News corpus word vector \[[link\]](https://code.google.com/archive/p/word2vec/)
