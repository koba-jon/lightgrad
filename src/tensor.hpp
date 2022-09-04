#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include "declare.hpp"


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
        TensorFloatStruct *struct_ptr = nullptr;

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
        friend std::ostream &operator<<(std::ostream &os, const TensorFloat &tensor);  // Insertion

        // Function
        void connect(TensorFloatStruct * const struct_ptr_);
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
        size_t size();
        std::vector<size_t> shape();
        float *data();
        TensorFloat &grad();
        float scalar();

        // Destructor
        ~TensorFloat();

    };


    // ------------------------------
    // structure{TensorFloatStruct}
    // ------------------------------
    struct TensorFloatStruct{

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

        // Constructor
        TensorFloatStruct() = default;  // Default
        TensorFloatStruct(const TensorFloatStruct &tensor) = delete;  // Copy

        // Operator
        TensorFloatStruct operator=(const TensorFloatStruct &tensor) = delete;  // Assignment

        // Destructor
        ~TensorFloatStruct();

    };


}


#endif