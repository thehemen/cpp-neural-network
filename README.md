# C++ Neural Network
C++ Neural Network made from scratch. Written for training purposes only. Uses [OpenMP](https://www.openmp.org/) for faster computations.
It gives 98,4% accuracy on MNIST dataset after ~40 min training on [Intel Core-i5 7400](https://ark.intel.com/content/www/ru/ru/ark/products/97147/intel-core-i5-7400-processor-6m-cache-up-to-3-50-ghz.html).
# Supported features
Layers:
* Conv2D,
* SeparableConv2D (depthwise+pointwise convolutions),
* MaxPooling2D,
* Flatten,
* Dense,
* Activation{n}D (for activations of both 1d- and 2d- data types).

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
Create /mnist directory, [download](http://yann.lecun.com/exdb/mnist/) MNIST archives and unpack here,
Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target mnist
$ ./mnist
```