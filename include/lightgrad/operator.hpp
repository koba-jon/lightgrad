#ifndef LIGHTGRAD_OPERATOR_HPP
#define LIGHTGRAD_OPERATOR_HPP

#include <string>
#include <lightgrad/declare.hpp>
#include <lightgrad/tensor.hpp>
#include <lightgrad/functional.hpp>


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
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

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
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~Multiplication() = default;

    };


    // ----------------------------
    // class{Subscript}(Function)
    // ----------------------------
    class Subscript : public Function{

    private:

        // Member variable
        TensorFloat input;
        size_t idx, dim;

    public:

        // Constructor
        Subscript() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input_, const size_t idx_);
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~Subscript() = default;

    };


    // ------------------------------
    // class{Unsubscript}(Function)
    // ------------------------------
    class Unsubscript : public Function{

    private:

        // Member variable
        TensorFloat input;
        size_t idx;

    public:

        // Constructor
        Unsubscript() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input_, const size_t idx_, const size_t dim_);
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~Unsubscript() = default;

    };


}


#endif