#include <iostream>
#include "lightgrad.hpp"

namespace lg = lightgrad;


int main(void){


    // ----------------
    // Scalar version
    // ----------------

    // (0) Declare tensor
    lg::TensorFloat x1, x2, x3, y, grad;

    // (1) Set data as x1 = 2, x2 = 3 and x3 = 5
    x1.from_scalar(2.0);
    x2.from_scalar(3.0);
    x3.from_scalar(5.0);
    std::cout << "x1 = " << x1.scalar();
    std::cout << ", x2 = " << x2.scalar();
    std::cout << ", x3 = " << x3.scalar();
    std::cout << std::endl;
    
    // (2) y = x1^3 * x2^2 + x1 * x3 = 82
    y = x1 * x1 * x1 * x2 * x2 + x1 * x3;
    std::cout << "y = x1^3 * x2^2 + x1 * x3 = " << y.scalar() << std::endl;

    // (3.1) y' = 3 * x1^2 * x2^2 + x3 = 113
    grad = lg::differential(y, x1, 1);
    std::cout << "y' = 3 * x1^2 * x2^2 + x3 = " << grad.scalar() << std::endl;

    // (3.2) y'' = 6 * x1 * x2^2 = 108
    grad = lg::differential(y, x1, 2);
    std::cout << "y'' = 6 * x1 * x2^2 = " << grad.scalar() << std::endl;

    // (3.3) y''' = 6 * x2^2 = 54
    grad = lg::differential(y, x1, 3);
    std::cout << "y''' = 6 * x2^2 = " << grad.scalar() << std::endl;

    // (3.4) y'''' = 0
    grad = lg::differential(y, x1, 4);
    std::cout << "y'''' = " << grad.scalar() << std::endl << std::endl;


    // ----------------
    // Tensor version
    // ----------------

    // (0) Declare tensor
    lg::TensorFloat a1, a2, a3, a4;
    lg::TensorFloat b1, b2, b3;
    lg::TensorFloat c, d;

    // (1) Set data
    a1.from_array({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    a1.new_grad();
    std::cout << "a1 = " << std::endl;
    std::cout << a1 << std::endl;
    /*****************************************************************/
    a2.from_array({3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4}, {2, 2, 3});
    a2.new_grad();
    std::cout << "a2 = " << std::endl;
    std::cout << a2 << std::endl;
    /*****************************************************************/
    a3.from_array({2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 3});
    a3.new_grad();
    std::cout << "a3 = " << std::endl;
    std::cout << a3 << std::endl;
    /*****************************************************************/
    a4.from_scalar(2);
    a4.new_grad();
    std::cout << "a4 = " << std::endl;
    std::cout << a4 << std::endl;

    // (2) Forward
    b1 = a1 + a2;
    std::cout << "b1 = a1 + a2 = " << std::endl;
    std::cout << b1 << std::endl;
    /*****************************************************************/
    b2 = a1 * a2 + lg::detach(a3);
    std::cout << "b2 = a1 * a2 + a3.detach() = " << std::endl;
    std::cout << b2 << std::endl;
    /*****************************************************************/
    b3 = lg::expand(a4, a1.shape());
    std::cout << "b3 = a4.expand() = " << std::endl;
    std::cout << b3 << std::endl;
    /*****************************************************************/
    c = (b1 + b2) * b3;
    std::cout << "c = (b1 + b2) * b3 = " << std::endl;
    std::cout << c << std::endl;
    /*****************************************************************/
    d = lg::sum(c);
    std::cout << "d = c.sum() = " << std::endl;
    std::cout << d << std::endl;

    // (3) Backward
    d.backward();

    // (4) Show result
    std::cout << "a1.grad = " << std::endl;
    std::cout << a1.grad() << std::endl;
    std::cout << "a2.grad = " << std::endl;
    std::cout << a2.grad() << std::endl;
    std::cout << "a3.grad = " << std::endl;
    std::cout << a3.grad() << std::endl;
    std::cout << "a4.grad = " << std::endl;
    std::cout << a4.grad() << std::endl;

    // (5) Post-processing
    a1.delete_grad();
    a2.delete_grad();
    a3.delete_grad();
    a4.delete_grad();

    return 0;

}

