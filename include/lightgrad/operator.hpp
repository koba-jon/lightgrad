#ifndef LIGHTGRAD_OPERATOR_HPP
#define LIGHTGRAD_OPERATOR_HPP

#include <string>
#include <lightgrad/declare.hpp>
#include <lightgrad/tensor.hpp>


// ----------------------
// namespace{lightgrad}
// ----------------------
namespace lightgrad{


    // ---------------------------
    // class{Addition}(Function)
    // ---------------------------
    class Addition : public Function{

    private:

        // Member variable
        TensorFloat input1;
        TensorFloat input2;

    public:

        // Constructor
        Addition() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input1_, TensorFloat input2_);
        void backward(TensorFloat grad) override;
        std::string type_name() override;

        // Destructor
        ~Addition() = default;

    };


    // ---------------------------------
    // class{Multiplication}(Function)
    // ---------------------------------
    class Multiplication : public Function{

    private:

        // Member variable
        TensorFloat input1;
        TensorFloat input2;

    public:

        // Constructor
        Multiplication() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input1_, TensorFloat input2_);
        void backward(TensorFloat grad) override;
        std::string type_name() override;

        // Destructor
        ~Multiplication() = default;

    };


}


#endif