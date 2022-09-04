#include <iostream>
#include <string>
#include <cstdlib>
#include <lightgrad/tensor.hpp>
#include <lightgrad/operator.hpp>


// --------------------------------------------------------------
// namespace{lightgrad} -> class{Addition} -> function{forward}
// --------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Addition::forward(TensorFloat input1_, TensorFloat input2_){

    if ((input1_.size() == 0) || (input2_.size() == 0)){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::exit(1);
    }
    else if (input1_.shape() != input2_.shape()){
        std::cerr << "Error: The shape between 'input1' and 'input2' is not equal." << std::endl;
        std::exit(1);
    }

    Addition *func = new Addition;
    /****************************************/
    func->input1 = input1_;
    func->input2 = input2_;

    size_t size;
    float *input1_data, *input2_data, *output_data;
    TensorFloat output;
    /****************************************/
    size = func->input1.size();
    input1_data = func->input1.data();
    input2_data = func->input2.data();
    output.create(func, func->input1.shape());
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < size; i++){
        output_data[i] = input1_data[i] + input2_data[i];
    }

    return output;

}


// ---------------------------------------------------------------
// namespace{lightgrad} -> class{Addition} -> function{backward}
// ---------------------------------------------------------------
void lightgrad::Addition::backward(TensorFloat grad){
    this->input1.backward(grad);
    this->input2.backward(grad);
    return;
}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{Addition} -> function{type_name}
// ----------------------------------------------------------------
std::string lightgrad::Addition::type_name(){
    return "Addition";
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication} -> function{forward}
// --------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Multiplication::forward(TensorFloat input1_, TensorFloat input2_){

    if ((input1_.size() == 0) || (input2_.size() == 0)){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::exit(1);
    }
    else if (input1_.shape() != input2_.shape()){
        std::cerr << "Error: The shape between 'input1' and 'input2' is not equal." << std::endl;
        std::exit(1);
    }

    Multiplication *func = new Multiplication;
    /****************************************/
    func->input1 = input1_;
    func->input2 = input2_;

    size_t size;
    float *input1_data, *input2_data, *output_data;
    TensorFloat output;
    /****************************************/
    size = func->input1.size();
    input1_data = func->input1.data();
    input2_data = func->input2.data();
    output.create(func, func->input1.shape());
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < size; i++){
        output_data[i] = input1_data[i] * input2_data[i];
    }

    return output;

}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication} -> function{backward}
// ---------------------------------------------------------------------
void lightgrad::Multiplication::backward(TensorFloat grad){
    this->input1.backward(grad * this->input2);
    this->input2.backward(grad * this->input1);
    return;
}


// ----------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication} -> function{type_name}
// ----------------------------------------------------------------------
std::string lightgrad::Multiplication::type_name(){
    return "Multiplication";
}
