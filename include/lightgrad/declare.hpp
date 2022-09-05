#ifndef LIGHTGRAD_DECLARE_HPP
#define LIGHTGRAD_DECLARE_HPP


// Forward declaration for class and structure
namespace lightgrad{
    
    // For tensor.hpp
    class TensorFloat;  // Tensor with float type (paired with TensorFloatParam)
    struct TensorFloatParam;  // Parameter of tensor with float type (paired with TensorFloat)

    // For operator.hpp
    class Addition;  // Addition
    class Multiplication;  // Multiplication
    class Subscript;  // Subscript (paired with Unsubscript)
    class Unsubscript;  // Unsubscript (paired with Subscript)

    // For functional.hpp
    class Function;  // Function
    class Identity;  // Identity
    class View;  // View
    class Sum;  // Sum (paired with Expand)
    class Expand;  // Expand (paired with Sum)

    // For optimizer.hpp
    class Optimizer;  // Optimizer
    class SGD;  // Stochastic Gradient Descent

}


#endif