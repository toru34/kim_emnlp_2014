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
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--train_x_path`: File path of train x data [default: `./data/train_x.txt`]
- `--train_y_path`: File path of train y data [default: `./data/train_y.txt`]
- `--valid_x_path`: File path of valid x data [default: `./data/valid_x.txt`]
- `--valid_y_path`: File path of valid y data [default: `./data/valid_y.txt`]
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
python train_manualbatch.py --num_epochs 20
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
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
All examples below are trained with the default arguments except `v_strategy`.
##### `v_strategy`: static
```
EPOCH: 1, Train Loss: 0.625 (F1: 0.644, Acc: 0.647), Valid Loss: 0.513 (F1: 0.765, Acc: 0.756), Time: 22.341[s]
EPOCH: 2, Train Loss: 0.450 (F1: 0.784, Acc: 0.787), Valid Loss: 0.484 (F1: 0.801, Acc: 0.783), Time: 44.712[s]
EPOCH: 3, Train Loss: 0.372 (F1: 0.835, Acc: 0.836), Valid Loss: 0.463 (F1: 0.799, Acc: 0.791), Time: 67.112[s]
EPOCH: 4, Train Loss: 0.304 (F1: 0.879, Acc: 0.879), Valid Loss: 0.458 (F1: 0.811, Acc: 0.800), Time: 89.552[s]
EPOCH: 5, Train Loss: 0.241 (F1: 0.916, Acc: 0.916), Valid Loss: 0.469 (F1: 0.812, Acc: 0.796), Time: 112.099[s]
EPOCH: 6, Train Loss: 0.185 (F1: 0.945, Acc: 0.945), Valid Loss: 0.472 (F1: 0.806, Acc: 0.804), Time: 134.670[s]
EPOCH: 7, Train Loss: 0.137 (F1: 0.969, Acc: 0.969), Valid Loss: 0.485 (F1: 0.809, Acc: 0.809), Time: 157.273[s]
EPOCH: 8, Train Loss: 0.101 (F1: 0.983, Acc: 0.983), Valid Loss: 0.490 (F1: 0.812, Acc: 0.811), Time: 179.885[s]
EPOCH: 9, Train Loss: 0.082 (F1: 0.989, Acc: 0.989), Valid Loss: 0.493 (F1: 0.813, Acc: 0.807), Time: 202.485[s]
EPOCH: 10, Train Loss: 0.072 (F1: 0.990, Acc: 0.990), Valid Loss: 0.494 (F1: 0.810, Acc: 0.806), Time: 225.067[s]

```
##### `v_strategy`: non-static
```
EPOCH: 1, Train Loss: 0.611 (F1: 0.654, Acc: 0.658), Valid Loss: 0.490 (F1: 0.783, Acc: 0.776), Time: 1763.849[s]
EPOCH: 2, Train Loss: 0.370 (F1: 0.835, Acc: 0.837), Valid Loss: 0.484 (F1: 0.798, Acc: 0.776), Time: 3542.999[s]
EPOCH: 3, Train Loss: 0.227 (F1: 0.919, Acc: 0.920), Valid Loss: 0.487 (F1: 0.796, Acc: 0.794), Time: 5319.272[s]
EPOCH: 4, Train Loss: 0.121 (F1: 0.969, Acc: 0.969), Valid Loss: 0.527 (F1: 0.799, Acc: 0.786), Time: 7095.262[s]
EPOCH: 5, Train Loss: 0.058 (F1: 0.990, Acc: 0.990), Valid Loss: 0.583 (F1: 0.803, Acc: 0.792), Time: 8871.713[s]
EPOCH: 6, Train Loss: 0.029 (F1: 0.997, Acc: 0.997), Valid Loss: 0.634 (F1: 0.798, Acc: 0.794), Time: 10650.794[s]
EPOCH: 7, Train Loss: 0.015 (F1: 0.999, Acc: 0.999), Valid Loss: 0.688 (F1: 0.797, Acc: 0.794), Time: 12426.908[s]
EPOCH: 8, Train Loss: 0.009 (F1: 0.999, Acc: 0.999), Valid Loss: 0.740 (F1: 0.786, Acc: 0.784), Time: 14205.622[s]
EPOCH: 9, Train Loss: 0.006 (F1: 1.000, Acc: 1.000), Valid Loss: 0.781 (F1: 0.802, Acc: 0.794), Time: 15983.344[s]
EPOCH: 10, Train Loss: 0.004 (F1: 1.000, Acc: 1.000), Valid Loss: 0.819 (F1: 0.785, Acc: 0.784), Time: 17760.783[s]
```
##### `v_strategy`: rand
```
EPOCH: 1, Train Loss: 0.682 (F1: 0.578, Acc: 0.570), Valid Loss: 0.604 (F1: 0.704, Acc: 0.689), Time: 1767.448[s]
EPOCH: 2, Train Loss: 0.486 (F1: 0.780, Acc: 0.781), Valid Loss: 0.522 (F1: 0.752, Acc: 0.737), Time: 3548.673[s]
EPOCH: 3, Train Loss: 0.300 (F1: 0.889, Acc: 0.890), Valid Loss: 0.530 (F1: 0.746, Acc: 0.750), Time: 5327.865[s]
EPOCH: 4, Train Loss: 0.168 (F1: 0.949, Acc: 0.949), Valid Loss: 0.549 (F1: 0.771, Acc: 0.758), Time: 7107.400[s]
EPOCH: 5, Train Loss: 0.081 (F1: 0.983, Acc: 0.983), Valid Loss: 0.631 (F1: 0.763, Acc: 0.765), Time: 8886.359[s]
EPOCH: 6, Train Loss: 0.036 (F1: 0.995, Acc: 0.995), Valid Loss: 0.723 (F1: 0.757, Acc: 0.759), Time: 10662.619[s]
EPOCH: 7, Train Loss: 0.019 (F1: 0.998, Acc: 0.998), Valid Loss: 0.769 (F1: 0.761, Acc: 0.757), Time: 12433.836[s]
EPOCH: 8, Train Loss: 0.011 (F1: 0.999, Acc: 0.999), Valid Loss: 0.835 (F1: 0.753, Acc: 0.757), Time: 14207.155[s]
EPOCH: 9, Train Loss: 0.007 (F1: 1.000, Acc: 1.000), Valid Loss: 0.870 (F1: 0.761, Acc: 0.756), Time: 15979.763[s]
EPOCH: 10, Train Loss: 0.005 (F1: 1.000, Acc: 1.000), Valid Loss: 0.911 (F1: 0.760, Acc: 0.753), Time: 17749.891[s]
```
##### `v_strategy`: multichannel
```
EPOCH: 1, Train Loss: 0.626 (F1: 0.659, Acc: 0.661), Valid Loss: 0.480 (F1: 0.776, Acc: 0.773), Time: 1198.847[s]
EPOCH: 2, Train Loss: 0.334 (F1: 0.855, Acc: 0.856), Valid Loss: 0.493 (F1: 0.800, Acc: 0.774), Time: 2410.564[s]
EPOCH: 3, Train Loss: 0.171 (F1: 0.946, Acc: 0.947), Valid Loss: 0.479 (F1: 0.811, Acc: 0.805), Time: 3622.606[s]
EPOCH: 4, Train Loss: 0.075 (F1: 0.986, Acc: 0.987), Valid Loss: 0.506 (F1: 0.815, Acc: 0.810), Time: 4834.480[s]
EPOCH: 5, Train Loss: 0.034 (F1: 0.996, Acc: 0.996), Valid Loss: 0.557 (F1: 0.810, Acc: 0.797), Time: 6047.958[s]
EPOCH: 6, Train Loss: 0.016 (F1: 0.999, Acc: 0.999), Valid Loss: 0.588 (F1: 0.814, Acc: 0.811), Time: 7261.833[s]
EPOCH: 7, Train Loss: 0.010 (F1: 0.999, Acc: 0.999), Valid Loss: 0.615 (F1: 0.808, Acc: 0.803), Time: 8475.354[s]
EPOCH: 8, Train Loss: 0.006 (F1: 1.000, Acc: 1.000), Valid Loss: 0.659 (F1: 0.805, Acc: 0.809), Time: 9687.347[s]
EPOCH: 9, Train Loss: 0.005 (F1: 1.000, Acc: 1.000), Valid Loss: 0.668 (F1: 0.808, Acc: 0.801), Time: 10897.239[s]
EPOCH: 10, Train Loss: 0.004 (F1: 1.000, Acc: 1.000), Valid Loss: 0.693 (F1: 0.802, Acc: 0.794), Time: 12104.786[s]

```

### Notes
- All experiments are done with GeForce GTX 1060 (6GB).
- Adam optimizer is used in all experiments (Original paper used Adadelta).

### Reference
- [1] Y. Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of EMNLP 2014 \[[pdf\]](https://arxiv.org/pdf/1408.5882.pdf)
- [2] B. Peng and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL \[[pdf\]](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf)
- [3] Google News corpus word vector \[[link\]](https://code.google.com/archive/p/word2vec/)
