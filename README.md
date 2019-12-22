# C++ Neural Network
C++ Neural Network made from scratch. Written for training purposes only. Uses [OpenMP](https://www.openmp.org/) for faster computations.
# How to run XOR
Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target xor
$ ./xor
```
# How to run MNIST
Create /mnist directory, [download](http://yann.lecun.com/exdb/mnist/) MNIST archives and unpack here,
Create /bin directory, run here:
```sh
$ cmake ..
$ cmake --build . --target mnist
$ ./mnist
```
# To-Do
Backpropagation between convolutional layers works not enough perfectly and should be fixed.