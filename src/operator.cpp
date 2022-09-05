#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <lightgrad/tensor.hpp>
#include <lightgrad/operator.hpp>


// ------------------------------------------------------------------------
// namespace{lightgrad} -> class{Addition}(Function) -> function{forward}
// ------------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Addition::forward(TensorFloat input1_, TensorFloat input2_){

    if ((input1_.size() == 0) || (input2_.size() == 0)){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Addition::forward)" << std::endl;
        std::exit(1);
    }
    else if (input1_.shape() != input2_.shape()){
        std::cerr << "Error: The shape between 'input1' and 'input2' is not equal." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Addition::forward)" << std::endl;
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


// -------------------------------------------------------------------------
// namespace{lightgrad} -> class{Addition}(Function) -> function{backward}
// -------------------------------------------------------------------------
void lightgrad::Addition::backward(TensorFloat grad_){
    this->input1.backward(grad_);
    this->input2.backward(grad_);
    return;
}


// --------------------------------------------------------------------------
// namespace{lightgrad} -> class{Addition}(Function) -> function{type_name}
// --------------------------------------------------------------------------
std::string lightgrad::Addition::type_name(){
    return "Addition";
}


// --------------------------------------------------------------------------
// namespace{lightgrad} -> class{Addition}(Function) -> function{clone_pre}
// --------------------------------------------------------------------------
lightgrad::Function *lightgrad::Addition::clone_pre(){
    Addition *func = new Addition;
    func->input1 = this->input1.clone_pre();
    func->input2 = this->input2.clone_pre();
    return func;
}


// ---------------------------------------------------------------------------
// namespace{lightgrad} -> class{Addition}(Function) -> function{clone_post}
// ---------------------------------------------------------------------------
void lightgrad::Addition::clone_post(){
    this->input1.clone_post();
    this->input2.clone_post();
    return;
}


// ------------------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication}(Function) -> function{forward}
// ------------------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Multiplication::forward(TensorFloat input1_, TensorFloat input2_){

    if ((input1_.size() == 0) || (input2_.size() == 0)){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Multiplication::forward)" << std::endl;
        std::exit(1);
    }
    else if (input1_.shape() != input2_.shape()){
        std::cerr << "Error: The shape between 'input1' and 'input2' is not equal." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Multiplication::forward)" << std::endl;
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


// -------------------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication}(Function) -> function{backward}
// -------------------------------------------------------------------------------
void lightgrad::Multiplication::backward(TensorFloat grad_){
    this->input1.backward(grad_ * this->input2);
    this->input2.backward(grad_ * this->input1);
    return;
}


// --------------------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication}(Function) -> function{type_name}
// --------------------------------------------------------------------------------
std::string lightgrad::Multiplication::type_name(){
    return "Multiplication";
}


// --------------------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication}(Function) -> function{clone_pre}
// --------------------------------------------------------------------------------
lightgrad::Function *lightgrad::Multiplication::clone_pre(){
    Multiplication *func = new Multiplication;
    func->input1 = this->input1.clone_pre();
    func->input2 = this->input2.clone_pre();
    return func;
}


// ---------------------------------------------------------------------------------
// namespace{lightgrad} -> class{Multiplication}(Function) -> function{clone_post}
// ---------------------------------------------------------------------------------
void lightgrad::Multiplication::clone_post(){
    this->input1.clone_post();
    this->input2.clone_post();
    return;
}


// -------------------------------------------------------------------------
// namespace{lightgrad} -> class{Subscript}(Function) -> function{forward}
// -------------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Subscript::forward(TensorFloat input_, const size_t idx_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Subscript::forward)" << std::endl;
        std::exit(1);
    }
    else if (input_.shape().size() == 0){
        if (idx_ > 0){
            std::cerr << "Error: The index of the subscript operator should be 0." << std::endl;
            std::cerr << "       [operator.cpp](lightgrad::Subscript::forward)" << std::endl;
            std::exit(1);
        }
    }
    else if (idx_ >= input_.shape()[0]){
        std::cerr << "Error: The index of the subscript operator should be less than " << input_.shape()[0] << "." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Subscript::forward)" << std::endl;
        std::exit(1);
    }

    Subscript *func = new Subscript;
    /****************************************/
    func->input = input_;
    func->idx = idx_;
    if (input_.shape().size() == 0){
        func->dim = 0;
    }
    else{
        func->dim = input_.shape()[0];
    }

    size_t sizeI, sizeO;
    std::vector<size_t> shapeI, shapeO;
    size_t offset;
    float *input_data, *output_data;
    TensorFloat output;
    /****************************************/
    sizeI = func->input.size();
    shapeI = func->input.shape();
    if (shapeI.size() == 0){
        sizeO = 1;
        shapeO = std::vector<size_t>();
        offset = 0;
    }
    else{
        sizeO = sizeI / shapeI[0];
        shapeO = shapeI;
        shapeO.erase(shapeO.begin());
        offset = func->idx * sizeO;
    }
    /****************************************/
    input_data = func->input.data();
    output.create(func, shapeO);
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < sizeO; i++){
        output_data[i] = input_data[offset + i];
    }

    return output;

}


// --------------------------------------------------------------------------
// namespace{lightgrad} -> class{Subscript}(Function) -> function{backward}
// --------------------------------------------------------------------------
void lightgrad::Subscript::backward(TensorFloat grad_){

    if (this->dim == 0){
        this->input.backward(
            identity(grad_)
        );
    }
    else{
        this->input.backward(
            Unsubscript().forward(grad_, this->idx, this->dim)
        );
    }

    return;

}


// ---------------------------------------------------------------------------
// namespace{lightgrad} -> class{Subscript}(Function) -> function{type_name}
// ---------------------------------------------------------------------------
std::string lightgrad::Subscript::type_name(){
    return "Subscript";
}


// ---------------------------------------------------------------------------
// namespace{lightgrad} -> class{Subscript}(Function) -> function{clone_pre}
// ---------------------------------------------------------------------------
lightgrad::Function *lightgrad::Subscript::clone_pre(){
    Subscript *func = new Subscript;
    func->input = this->input.clone_pre();
    func->idx = this->idx;
    func->dim = this->dim;
    return func;
}


// ----------------------------------------------------------------------------
// namespace{lightgrad} -> class{Subscript}(Function) -> function{clone_post}
// ----------------------------------------------------------------------------
void lightgrad::Subscript::clone_post(){
    this->input.clone_post();
    return;
}


// ---------------------------------------------------------------------------
// namespace{lightgrad} -> class{Unsubscript}(Function) -> function{forward}
// ---------------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Unsubscript::forward(TensorFloat input_, const size_t idx_, const size_t dim_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Unsubscript::forward)" << std::endl;
        std::exit(1);
    }
    else if (idx_ >= dim_){
        std::cerr << "Error: The index must be less than 'dim'." << std::endl;
        std::cerr << "       [operator.cpp](lightgrad::Unsubscript::forward)" << std::endl;
        std::exit(1);
    }

    Unsubscript *func = new Unsubscript;
    /****************************************/
    func->input = input_;
    func->idx = idx_;

    size_t sizeI, sizeO;
    std::vector<size_t> shapeI, shapeO;
    size_t offset;
    float *input_data, *output_data;
    TensorFloat output;
    /****************************************/
    sizeI = func->input.size();
    shapeI = func->input.shape();
    sizeO = sizeI * dim_;
    shapeO = shapeI;
    shapeO.insert(shapeO.begin(), dim_);
    offset = func->idx * sizeI;
    /****************************************/
    input_data = func->input.data();
    output.create(func, shapeO);
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < sizeI; i++){
        output_data[offset + i] = input_data[i];
    }
    /****************************************/
    for (size_t i = 0; i < offset; i++){
        output_data[i] = 0.0;
    }
    for (size_t i = offset + sizeI; i < sizeO; i++){
        output_data[i] = 0.0;
    }

    return output;

}


// ----------------------------------------------------------------------------
// namespace{lightgrad} -> class{Unsubscript}(Function) -> function{backward}
// ----------------------------------------------------------------------------
void lightgrad::Unsubscript::backward(TensorFloat grad_){
    this->input.backward(grad_[this->idx]);
    return;
}


// -----------------------------------------------------------------------------
// namespace{lightgrad} -> class{Unsubscript}(Function) -> function{type_name}
// -----------------------------------------------------------------------------
std::string lightgrad::Unsubscript::type_name(){
    return "Unsubscript";
}


// -----------------------------------------------------------------------------
// namespace{lightgrad} -> class{Unsubscript}(Function) -> function{clone_pre}
// -----------------------------------------------------------------------------
lightgrad::Function *lightgrad::Unsubscript::clone_pre(){
    Unsubscript *func = new Unsubscript;
    func->input = this->input.clone_pre();
    func->idx = this->idx;
    return func;
}


// ------------------------------------------------------------------------------
// namespace{lightgrad} -> class{Unsubscript}(Function) -> function{clone_post}
// ------------------------------------------------------------------------------
void lightgrad::Unsubscript::clone_post(){
    this->input.clone_post();
    return;
}
