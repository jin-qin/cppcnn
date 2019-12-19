# CPPCNN
A tiny framework of convolutional nerual networks with C++ implementation. You can add arbitrary number of convolutional layers, max pooling layers and hidden layers to construct your own convolutional neural networks.

This framework contains a sub neural network framework which is multi-layer neural network, you can also use it as CNNs to construct your own multi-layer neural netowrk.

This project also implemented `Momentum SGD` and `Adam Optimizer` in Multi-layer Neural Networks part and can be enabled by functions.

# Build
> `mkdir build && cd build`\
> `cmake ..`\
> `make` on macOS OR `open solution` if you are using windows

Tested on Windows or macOS, but should also work on linux systems.

# Test and verification
This tiny frame work has been tested on MNIST(handwritten digits) dataset by LeCun.
1. make sure there are 4 files named `train-images.idx3-ubyte` `train-labels.idx1-ubyte` `t10k-images.idx3-ubyte` `t10k-labels.idx1-ubyte` in the `data/mnist` path. Dataset can be downloaded from [here](http://yann.lecun.com/exdb/mnist/).
2. after build, if you are on macOS, run `./cppcnn` in `build` folder, then it will load MNIST dataset and after 1-3 hours on CPU, it will give the test results.

## Performance of Multi-layer Neural Networks
Get 95% accuracy by 5 epoch, with hyperparameters:
- learning rate: 1.0
- hidden layers: one hidden layer with 256 units
- mini batch size: 256
- momentum SGD: disabled
- adam optimizer: disabled

## Performance of Convolutional Neural Networks
Get 98% accuracy by 3 epoch, with hyperparameters:
- learning rate: 0.005
- convolution layers: one convolution layer with `8 convolution filters`, `size 3`, `stride 1`, `no padding`
- pooling layers: one max pooling layer with `size 2`, `stride 2`
- hidden layers: two hidden layer with 32 and 16 units
- mini batch size: 1
- momentum SGD: disabled
- adam optimizer: disabled

# TO DO
- [ ] Fix pooling layer bug for odd case.
- [ ] Implement `Momentum SGD` and `Adam Optimizer` in CNN class.
- [ ] Provide average pooling layer.
- [ ] Provide GPU acceleration.
