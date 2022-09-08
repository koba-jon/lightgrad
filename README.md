# LightGrad
***LightGrad*** is ***Light**est **Grad**ient Calculation Framework* in C++.<br>
This framework is a learning material created for us to understand Define-by-Run such as PyTorch.<br>

## (1) Clone

~~~
$ git clone https://github.com/koba-jon/lightgrad.git
$ cd lightgrad
~~~

## (2) Create Library

~~~
$ cd cmake
$ mkdir build
$ cd build
$ cmake ..
$ make install
$ cd ../..
~~~

## (3) Execute for Example

~~~
$ cd example
~~~

### Build

For release (default):
~~~
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
$ cd ..
~~~

For debug:
~~~
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug ..
$ make
$ cd ..
~~~


### Run

~~~
$ ./Example
~~~
