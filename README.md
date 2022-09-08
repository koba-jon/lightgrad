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

This operation created the directory "<this_repository_name>/lightgrad".

## (3) Execute

~~~
$ cd example
~~~

### Set Path

~~~
$ vi CMakeLists.txt
~~~

Please change the 5th line of "CMakeLists.txt" according to the path of the directory "<this_repository_name>/lightgrad".<br>
The following is the default value for "example".

~~~
4: project(Example CXX)
5: list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lightgrad)
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
