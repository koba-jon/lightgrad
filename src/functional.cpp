#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <lightgrad/tensor.hpp>
#include <lightgrad/functional.hpp>


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


// ---------------------------------------------------------
// namespace{lightgrad} -> class{Sum} -> function{forward}
// ---------------------------------------------------------
lightgrad::TensorFloat lightgrad::Sum::forward(TensorFloat input_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::exit(1);
    }

    Sum *func = new Sum;
    /****************************************/
    func->input = input_;

    size_t size;
    float *input_data, *output_data;
    TensorFloat output;
    /****************************************/
    size = func->input.size();
    input_data = func->input.data();
    output.create(func, {});
    output_data = output.data();
    /****************************************/
    output_data[0] = 0.0;
    for (size_t i = 0; i < size; i++){
        output_data[0] += input_data[i];
    }

    return output;

}


// ----------------------------------------------------------
// namespace{lightgrad} -> class{Sum} -> function{backward}
// ----------------------------------------------------------
void lightgrad::Sum::backward(TensorFloat grad){
    this->input.backward(Expand(this->input.shape()).forward(grad));
    return;
}


// -----------------------------------------------------------
// namespace{lightgrad} -> class{Sum} -> function{type_name}
// -----------------------------------------------------------
std::string lightgrad::Sum::type_name(){
    return "Sum";
}


// ---------------------------------------------------------------
// namespace{lightgrad} -> class{Expand} -> original constructor
// ---------------------------------------------------------------
lightgrad::Expand::Expand(const std::vector<size_t> &shape_){

    this->size = 1;
    this->shape = shape_;
    for (const auto &length : this->shape){
        this->size *= length;
    }

    if (this->size == 0){
        std::cerr << "Error: The size of argument 'shape' is zero." << std::endl;
        std::exit(1);
    }

}


// ------------------------------------------------------------
// namespace{lightgrad} -> class{Expand} -> function{forward}
// ------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Expand::forward(TensorFloat input_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::exit(1);
    }
    else if ((input_.size() > 1) || (input_.shape().size() > 0)){
        std::cerr << "Error: The argument 'input' should be scalar." << std::endl;
        std::exit(1);
    }

    Expand *func = new Expand(this->shape);
    /****************************************/
    func->input = input_;

    float input_data;
    float *output_data;
    TensorFloat output;
    /****************************************/
    input_data = func->input.data()[0];
    output.create(func, func->shape);
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < this->size; i++){
        output_data[i] = input_data;
    }

    return output;

}


// -------------------------------------------------------------
// namespace{lightgrad} -> class{Expand} -> function{backward}
// -------------------------------------------------------------
void lightgrad::Expand::backward(TensorFloat grad){
    this->input.backward(Sum().forward(grad));
    return;
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{Expand} -> function{type_name}
// --------------------------------------------------------------
std::string lightgrad::Expand::type_name(){
    return "Expand";
}


// ---------------------------------------
// namespace{lightgrad} -> function{sum}
// ---------------------------------------
lightgrad::TensorFloat lightgrad::sum(TensorFloat tensor){
    return Sum().forward(tensor);
}


// ------------------------------------------
// namespace{lightgrad} -> function{expand}
// ------------------------------------------
lightgrad::TensorFloat lightgrad::expand(TensorFloat tensor, const std::vector<size_t> &shape){
    return Expand(shape).forward(tensor);
}


// ------------------------------------------
// namespace{lightgrad} -> function{detach}
// ------------------------------------------
lightgrad::TensorFloat lightgrad::detach(TensorFloat tensorI){

    if (!tensorI.exist){
        std::cerr << "Error: Couldn't execute 'detach' because the member 'exist' of Tensor is false." << std::endl;
        std::exit(1);
    }

    TensorFloat tensorO;
    /****************************************/
    TensorFloatStruct &tensorIS = *(tensorI.struct_ptr);
    /****************************************/
    tensorO.connect(new TensorFloatStruct);
    TensorFloatStruct &tensorOS = *(tensorO.struct_ptr);
    tensorOS.exist = tensorIS.exist;
    tensorOS.size = tensorIS.size;
    tensorOS.shape = tensorIS.shape;
    tensorOS.grad_on = false;
    tensorOS.grad = TensorFloat();
    tensorOS.created = false;
    tensorOS.creator = nullptr;

    tensorOS.data = new float[tensorOS.size];
    for (size_t i = 0; i < tensorOS.size; i++){
        tensorOS.data[i] = tensorIS.data[i];
    }

    return tensorO;
    
}


// ------------------------------------------------
// namespace{lightgrad} -> function{differential}
// ------------------------------------------------
lightgrad::TensorFloat lightgrad::differential(TensorFloat y, TensorFloat x, const unsigned int order){

    TensorFloat target(y);

    for (unsigned int i = 0; i < order; i++){
        x.new_grad();
        target.backward();
        target = x.grad();
    }
    x.delete_grad();

    return target;

}