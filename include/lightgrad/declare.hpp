#ifndef LIGHTGRAD_DECLARE_HPP
#define LIGHTGRAD_DECLARE_HPP


// Forward declaration for class and structure
namespace lightgrad{
    
    
    // For tensor.hpp
    class TensorFloat;  // Tensor with float type
    struct TensorFloatStruct;  // Tensor with float type for structure

    // For operator.hpp
    class Addition;  // Addition
    class Multiplication;  // Multiplication

    // For functional.hpp
    class Sum;  // Sum
    class Expand;  // Expand


    // -----------------
    // class{Function}
    // -----------------
    class Function{

    public:

        // Constructor
        Function() = default;

        // Function (pure virtual)
        virtual void backward(TensorFloat grad) = 0;
        virtual std::string type_name() = 0;

        // Destructor (pure virtual)
        virtual ~Function() = default;

    };


}


#endif