# C++ Neural Network
C++ Neural Network made from scratch. Written for training purposes only. Uses [OpenMP](https://www.openmp.org/) for faster computations.

|Dataset |Duration |Accuracy|
|--------|---------|--------|
|MNIST   |26 min.  |98.4 %  |
|CIFAR-10|39 min.  |56.1 %  |
|SST-2   |33 min.  |74.9 %  |
|SST-5   |1h. 7min.|34.3 %  |

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
# How to run CIFAR-10 example
Download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) archive and unpack it.

Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target cifar10
$ ./cifar10 [data-batch-1] ... [data-batch-5] [test-batch]
```
# How to run SST-2/SST-5 example
Download refined [Stanford Sentiment Treebank](https://github.com/HaebinShin/stanford-sentiment-dataset) dataset.

For binary classification use *binary*-tagged files, for five-class — *fine*-tagged ones.

Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target sst
$ ./sst [is_binary] [train-path] [dev-path] [test-path]
```
