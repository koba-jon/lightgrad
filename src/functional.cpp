#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <lightgrad/tensor.hpp>
#include <lightgrad/functional.hpp>


// ------------------------------------------------------------------------
// namespace{lightgrad} -> class{Identity}(Function) -> function{forward}
// ------------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Identity::forward(TensorFloat input_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::Identity::forward)" << std::endl;
        std::exit(1);
    }

    Identity *func = new Identity;
    /****************************************/
    func->input = input_;

    size_t size;
    float *input_data, *output_data;
    TensorFloat output;
    /****************************************/
    size = func->input.size();
    input_data = func->input.data();
    output.create(func, func->input.shape());
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < size; i++){
        output_data[i] = input_data[i];
    }

    return output;

}


// -------------------------------------------------------------------------
// namespace{lightgrad} -> class{Identity}(Function) -> function{backward}
// -------------------------------------------------------------------------
void lightgrad::Identity::backward(TensorFloat grad_){
    this->input.backward(grad_);
    return;
}


// --------------------------------------------------------------------------
// namespace{lightgrad} -> class{Identity}(Function) -> function{type_name}
// --------------------------------------------------------------------------
std::string lightgrad::Identity::type_name(){
    return "Identity";
}


// ----------------------------------------------------------------------
// namespace{lightgrad} -> class{Identity}(Function) -> function{clone}
// ----------------------------------------------------------------------
lightgrad::Function *lightgrad::Identity::clone(){
    Identity *func = new Identity;
    func->input = this->input.clone();
    return func;
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{View}(Function) -> function{forward}
// --------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::View::forward(TensorFloat input_, const std::vector<size_t> &shape_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::View::forward)" << std::endl;
        std::exit(1);
    }

    size_t size;
    /****************************************/
    size = 1;
    for (const auto &length : shape_){
        size *= length;
    }
    /****************************************/
    if (input_.size() != size){
        std::cerr << "Error: The size between 'input' and 'shape' is not equal." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::View::forward)" << std::endl;
        std::exit(1);
    }

    View *func = new View;
    /****************************************/
    func->input = input_;

    float *input_data, *output_data;
    TensorFloat output;
    /****************************************/
    input_data = func->input.data();
    output.create(func, shape_);
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < size; i++){
        output_data[i] = input_data[i];
    }

    return output;

}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{View}(Function) -> function{backward}
// ---------------------------------------------------------------------
void lightgrad::View::backward(TensorFloat grad_){
    this->input.backward(
        view(grad_, this->input.shape())
    );
    return;
}


// ----------------------------------------------------------------------
// namespace{lightgrad} -> class{View}(Function) -> function{type_name}
// ----------------------------------------------------------------------
std::string lightgrad::View::type_name(){
    return "View";
}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{View}(Function) -> function{clone}
// ------------------------------------------------------------------
lightgrad::Function *lightgrad::View::clone(){
    View *func = new View;
    func->input = this->input.clone();
    return func;
}


// -------------------------------------------------------------------
// namespace{lightgrad} -> class{Sum}(Function) -> function{forward}
// -------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Sum::forward(TensorFloat input_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::Sum::forward)" << std::endl;
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


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{Sum}(Function) -> function{backward}
// --------------------------------------------------------------------
void lightgrad::Sum::backward(TensorFloat grad_){
    this->input.backward(
        expand(grad_, this->input.shape())
    );
    return;
}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{Sum}(Function) -> function{type_name}
// ---------------------------------------------------------------------
std::string lightgrad::Sum::type_name(){
    return "Sum";
}


// -----------------------------------------------------------------
// namespace{lightgrad} -> class{Sum}(Function) -> function{clone}
// -----------------------------------------------------------------
lightgrad::Function *lightgrad::Sum::clone(){
    Sum *func = new Sum;
    func->input = this->input.clone();
    return func;
}


// ----------------------------------------------------------------------
// namespace{lightgrad} -> class{Expand}(Function) -> function{forward}
// ----------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::Expand::forward(TensorFloat input_, const std::vector<size_t> &shape_){

    if (input_.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::Expand::forward)" << std::endl;
        std::exit(1);
    }
    else if ((input_.size() > 1) || (input_.shape().size() > 0)){
        std::cerr << "Error: The argument 'input' should be scalar." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::Expand::forward)" << std::endl;
        std::exit(1);
    }

    size_t size;
    /****************************************/
    size = 1;
    for (const auto &length : shape_){
        size *= length;
    }
    /****************************************/
    if (size == 0){
        std::cerr << "Error: The size of argument 'shape' is zero." << std::endl;
        std::cerr << "       [functional.cpp](lightgrad::Expand::forward)" << std::endl;
        std::exit(1);
    }

    Expand *func = new Expand();
    /****************************************/
    func->input = input_;

    float input_data;
    float *output_data;
    TensorFloat output;
    /****************************************/
    input_data = func->input.scalar();
    output.create(func, shape_);
    output_data = output.data();
    /****************************************/
    for (size_t i = 0; i < size; i++){
        output_data[i] = input_data;
    }

    return output;

}


// -----------------------------------------------------------------------
// namespace{lightgrad} -> class{Expand}(Function) -> function{backward}
// -----------------------------------------------------------------------
void lightgrad::Expand::backward(TensorFloat grad_){
    this->input.backward(
        sum(grad_)
    );
    return;
}


// ------------------------------------------------------------------------
// namespace{lightgrad} -> class{Expand}(Function) -> function{type_name}
// ------------------------------------------------------------------------
std::string lightgrad::Expand::type_name(){
    return "Expand";
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{Expand}(Function) -> function{clone}
// --------------------------------------------------------------------
lightgrad::Function *lightgrad::Expand::clone(){
    Expand *func = new Expand;
    func->input = this->input.clone();
    return func;
}


// --------------------------------------------
// namespace{lightgrad} -> function{identity}
// --------------------------------------------
lightgrad::TensorFloat lightgrad::identity(TensorFloat tensor){
    return Identity().forward(tensor);
}


// ----------------------------------------
// namespace{lightgrad} -> function{view}
// ----------------------------------------
lightgrad::TensorFloat lightgrad::view(TensorFloat tensorI, const std::vector<size_t> &shape){
    return View().forward(tensorI, shape);
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
    return Expand().forward(tensor, shape);
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
