#ifndef LIGHTGRAD_FUNCTIONAL_HPP
#define LIGHTGRAD_FUNCTIONAL_HPP

#include <string>
#include <vector>
#include <lightgrad/declare.hpp>
#include <lightgrad/tensor.hpp>


// ----------------------
// namespace{lightgrad}
// ----------------------
namespace lightgrad{


    // Forward declaration for function
    TensorFloat identity(TensorFloat tensor);
    TensorFloat view(TensorFloat tensorI, const std::vector<size_t> &shape);
    TensorFloat sum(TensorFloat tensor);
    TensorFloat expand(TensorFloat tensor, const std::vector<size_t> &shape);
    /****************************/
    TensorFloat differential(TensorFloat y, TensorFloat x, const unsigned int order = 1);


    // -----------------
    // class{Function}
    // -----------------
    class Function{

    public:

        // Constructor
        Function() = default;

        // Function (pure virtual)
        virtual void backward(TensorFloat grad_) = 0;
        virtual std::string type_name() = 0;
        virtual Function *clone_pre() = 0;
        virtual void clone_post() = 0;

        // Destructor (pure virtual)
        virtual ~Function() = default;

    };


    // ---------------------------
    // class{Identity}(Function)
    // ---------------------------
    class Identity : public Function{

    private:

        // Member variable
        TensorFloat input;

    public:

        // Constructor
        Identity() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input_);
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~Identity() = default;

    };


    // -----------------------
    // class{View}(Function)
    // -----------------------
    class View : public Function{

    private:

        // Member variable
        TensorFloat input;

    public:

        // Constructor
        View() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input_, const std::vector<size_t> &shape_);
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~View() = default;

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
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~Sum() = default;

    };


    // -------------------------
    // class{Expand}(Function)
    // -------------------------
    class Expand : public Function{

    private:

        // Member variable
        TensorFloat input;

    public:

        // Constructor
        Expand() = default;  // Default

        // Function
        TensorFloat forward(TensorFloat input_, const std::vector<size_t> &shape_);
        void backward(TensorFloat grad_) override;
        std::string type_name() override;
        Function *clone_pre() override;
        void clone_post() override;

        // Destructor
        ~Expand() = default;

    };


}


#endif