#ifndef LIGHTGRAD_FUNCTIONAL_HPP
#define LIGHTGRAD_FUNCTIONAL_HPP

#include <string>
#include <vector>
#include <lightgrad/declare.hpp>


// ----------------------
// namespace{lightgrad}
// ----------------------
namespace lightgrad{


    // Forward declaration for function
    TensorFloat sum(TensorFloat tensor);
    TensorFloat expand(TensorFloat tensor, const std::vector<size_t> &shape);
    /****************************/
    TensorFloat detach(TensorFloat tensorI);
    TensorFloat differential(TensorFloat y, TensorFloat x, const unsigned int order = 1);


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


    // ----------------------
    // class{Sum}(Function)
    // ----------------------
    class Sum : public Function{

    private:

        // Member variable
        TensorFloat input;

    public:

        // Constructor
        Sum() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input_);
        void backward(TensorFloat grad) override;
        std::string type_name() override;

        // Destructor
        ~Sum() = default;

    };


    // -------------------------
    // class{Expand}(Function)
    // -------------------------
    class Expand : public Function{

    private:

        // Member variable
        size_t size = 0;
        std::vector<size_t> shape;
        /****************************/
        TensorFloat input;

    public:

        // Constructor
        Expand() = delete;  // Default
        Expand(const std::vector<size_t> &shape_);  // Original

        // Function
        TensorFloat forward(TensorFloat input_);
        void backward(TensorFloat grad) override;
        std::string type_name() override;

        // Destructor
        ~Expand() = default;

    };


}


#endif