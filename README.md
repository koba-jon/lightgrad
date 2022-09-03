# LightGrad
**LightGrad** is Deep Learning Framework based on Define-by-Run in C++.<br>
This framework is a learning material created for us to understand Define-by-Run such as PyTorch, and it is the simplest Deep Learning framework.<br>

## Build & Execute

~~~
$ git clone https://github.com/koba-jon/lightgrad.git
$ cd lightgrad/src
~~~

For debug:
~~~
$ g++ main.cpp tensor.cpp functional.cpp -O0 -fsanitize=address -fno-omit-frame-pointer -g -Wall -std=c++17 -o Example
~~~

For release:
~~~
$ g++ main.cpp tensor.cpp functional.cpp -O2 -Wall -std=c++17 -o Example
~~~

~~~
$ Example
~~~
