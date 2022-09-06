#ifndef LIGHTGRAD_TENSOR_HPP
#define LIGHTGRAD_TENSOR_HPP

#include <iostream>
#include <vector>
#include <lightgrad/declare.hpp>


// ----------------------
// namespace{lightgrad}
// ----------------------
namespace lightgrad{


    // Forward declaration for function
    std::ostream &operator<<(std::ostream &os, const TensorFloat &tensor);


    // --------------------
    // class{TensorFloat}
    // --------------------
    class TensorFloat{

    private:

        // Function
        void copy(const TensorFloat &tensorI, TensorFloat &tensorO);

    public:

        // Member variable
        bool exist = false;
        TensorFloatParam *param = nullptr;

        // Constructor
        TensorFloat() = default;  // Default
        TensorFloat(const TensorFloat &tensor);  // Copy
        /****************************/
        TensorFloat(const std::vector<size_t> &shape_);  // Original
        TensorFloat(const float scalar);  // Original
        TensorFloat(const float scalar, const std::vector<size_t> &shape_);  // Original
        TensorFloat(const float *array, const std::vector<size_t> &shape_);  // Original
        TensorFloat(const std::vector<float> &array, const std::vector<size_t> &shape_);  // Original
        TensorFloat(Function * const creator_, const std::vector<size_t> &shape_);  // Original

        // Operator
        TensorFloat operator+(const TensorFloat &tensor);  // Addition
        TensorFloat operator*(const TensorFloat &tensor);  // Multiplication
        TensorFloat operator=(const TensorFloat &tensor);  // Assignment
        TensorFloat operator=(const float scalar);  // Assignment
        TensorFloat &operator+=(const TensorFloat &tensor);  // Addition assignment
        TensorFloat &operator*=(const TensorFloat &tensor);  // Multiplication assignment
        TensorFloat operator[](const size_t idx);  // Subscript
        friend std::ostream &operator<<(std::ostream &os, const TensorFloat &tensor);  // Insertion

        // Function
        void connect(TensorFloatParam * const param_);
        void disconnect();
        /****************************/
        void allocate(const std::vector<size_t> &shape_);
        void from_scalar(const float scalar);
        void from_scalar(const float scalar, const std::vector<size_t> &shape_);
        void from_array(const float *array, const std::vector<size_t> &shape_);
        void from_array(const std::vector<float> &array, const std::vector<size_t> &shape_);
        void create(Function * const creator_, const std::vector<size_t> &shape_);
        /****************************/
        void backward(TensorFloat grad_ = 1.0);
        /****************************/
        void new_grad();
        void delete_grad();
        /****************************/
        TensorFloat detach();
        TensorFloat clone();
        TensorFloat clone_pre();
        void clone_post();
        /****************************/
        size_t size();
        std::vector<size_t> shape();
        float *data();
        TensorFloat &grad();
        float scalar();

        // Destructor
        ~TensorFloat();

    };


    // -----------------------------
    // structure{TensorFloatParam}
    // -----------------------------
    struct TensorFloatParam{

    public:

        // Member variable
        size_t count = 0;
        /****************************/
        bool exist = false;
        size_t size = 0;
        std::vector<size_t> shape;
        float *data = nullptr;
        /****************************/
        bool grad_on = false;
        TensorFloat grad;
        /****************************/
        bool created = false;
        Function *creator = nullptr;
        /****************************/
        bool cloned = false;
        TensorFloatParam *destination = nullptr;

        // Constructor
        TensorFloatParam() = default;  // Default
        TensorFloatParam(const TensorFloatParam &tensor) = delete;  // Copy

        // Operator
        TensorFloatParam operator=(const TensorFloatParam &tensor) = delete;  // Assignment

        // Destructor
        ~TensorFloatParam();

    };


}


#endif