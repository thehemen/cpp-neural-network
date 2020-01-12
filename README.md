# C++ Neural Network
C++ Neural Network made from scratch. Written for training purposes only. Uses [OpenMP](https://www.openmp.org/) for faster computations.

It gives 98.4% accuracy on MNIST dataset in ~40 minutes training with Intel Core i5-7400.
# Supported features
Layers:
* Embedding,
* Conv1D,
* Conv2D,
* SeparableConv2D (depthwise+pointwise convolutions),
* MaxPooling1D,
* MaxPooling2D,
* GlobalMaxPooling1D,
* Flatten,
* Dense,
* Activation1D / Activation2D / Activation3D.

Activation types:
* Sigmoid,
* Tanh,
* ReLU,
* Softmax (described as layer).

Metrics:
* Binary Crossentropy Loss,
* Categorical Crossentropy Loss,
* Binary Crossentropy Accuracy,
* Categorical Crossentropy Accuracy.

Initializers:
* Xavier weight initializer.

Optimizers:
* Adaptive moment estimation (Adam).
# How to run XOR example
Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target xor
$ ./xor
```
# How to run MNIST example
Download [MNIST](http://yann.lecun.com/exdb/mnist/) archives and unpack it.
Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target mnist
$ ./mnist [train-images-path] [train-labels-path] [test-images-path] [test-labels-path]
```
# How to run SST-2/SST-5 example
[Download](https://github.com/HaebinShin/stanford-sentiment-dataset) refined Stanford Sentiment Treebank dataset.
Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target sst
$ ./sst [is_binary] [train-path] [dev-path] [test-path]
```
