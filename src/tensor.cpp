#include <iostream>
#include <vector>
#include <cstdlib>
#include "tensor.hpp"
#include "functional.hpp"


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> copy constructor
// ----------------------------------------------------------------
lightgrad::TensorFloat::TensorFloat(const TensorFloat &tensor){
    this->copy(tensor, *this);
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> original constructor
// --------------------------------------------------------------------
lightgrad::TensorFloat::TensorFloat(const std::vector<size_t> &shape_){
    this->allocate(shape_);
}

lightgrad::TensorFloat::TensorFloat(const float scalar){
    this->from_scalar(scalar);
}

lightgrad::TensorFloat::TensorFloat(const float scalar, const std::vector<size_t> &shape_){
    this->from_scalar(scalar, shape_);
}

lightgrad::TensorFloat::TensorFloat(const float *array, const std::vector<size_t> &shape_){
    this->from_array(array, shape_);
}

lightgrad::TensorFloat::TensorFloat(const std::vector<float> &array, const std::vector<size_t> &shape_){
    this->from_array(array, shape_);
}

lightgrad::TensorFloat::TensorFloat(Function * const creator_, const std::vector<size_t> &shape_){
    this->create(creator_, shape_);
}


// -----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> addition operator
// -----------------------------------------------------------------
lightgrad::TensorFloat lightgrad::TensorFloat::operator+(const TensorFloat &tensor){
    return Addition().forward(*this, tensor);
}


// -----------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> multiplication operator
// -----------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::TensorFloat::operator*(const TensorFloat &tensor){
    return Multiplication().forward(*this, tensor);
}


// -------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> assignment operator
// -------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::TensorFloat::operator=(const TensorFloat &tensor){
    this->copy(tensor, *this);
    return *this;
}

lightgrad::TensorFloat lightgrad::TensorFloat::operator=(const float scalar){
    this->from_scalar(scalar);
    return *this;
}


// ----------------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> addition assignment operator
// ----------------------------------------------------------------------------
lightgrad::TensorFloat &lightgrad::TensorFloat::operator+=(const TensorFloat &tensor){
    *this = Addition().forward(*this, tensor);
    return *this;
}


// ----------------------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> multiplication assignment operator
// ----------------------------------------------------------------------------------
lightgrad::TensorFloat &lightgrad::TensorFloat::operator*=(const TensorFloat &tensor){
    *this = Multiplication().forward(*this, tensor);
    return *this;
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> (class{TensorFloat} ->) insertion operator
// --------------------------------------------------------------------
std::ostream &lightgrad::operator<<(std::ostream &os, const TensorFloat &tensor){

    // (1) Check for element existence of the array
    if (!tensor.exist){
        os << "[Not exist]";
        return os;
    }
    /******************************************/
    TensorFloatStruct &tensorS = *(tensor.struct_ptr);
    if (!tensorS.exist){
        os << "[Not exist]";
        return os;
    }

    // (2) Describe data of the tensor
    for (size_t i = 0; i < tensorS.size; i++){
        os << tensorS.data[i] << " ";
    }
    os << "\n";

    // (3) Describe information of the tensor
    os << "[";
    /******************************************/
    os << "Type: Float";
    /******************************************/
    os << ", " << "Shape: ";
    if (tensorS.shape.size() == 0){
        os << "Scalar";
    }
    else{
        os << tensorS.shape.size() << "D-Tensor";
        os << "(";
        for (size_t i = 0; i < tensorS.shape.size(); i++){
            if (i != 0){
                os << "x";
            }
            os << tensorS.shape[i];
        }
        os << ")";
    }
    /******************************************/
    os << ", " << "Creator: ";
    if (tensorS.created){
        os << tensorS.creator->type_name();
    }
    else{
        os << "None";
    }
    /******************************************/
    os << ", " << "Grad: ";
    if (tensorS.grad_on){
        os << "ON";
    }
    else{
        os << "OFF";
    }
    /******************************************/
    os << "]\n";

    return os;

}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{copy}
// --------------------------------------------------------------
void lightgrad::TensorFloat::copy(const TensorFloat &tensorI, TensorFloat &tensorO){
    tensorO.disconnect();
    if (tensorI.exist) tensorO.connect(tensorI.struct_ptr);
    return;
}


// -----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{connect}
// -----------------------------------------------------------------
void lightgrad::TensorFloat::connect(TensorFloatStruct * const struct_ptr_){
    this->disconnect();
    this->exist = true;
    this->struct_ptr = struct_ptr_;
    this->struct_ptr->count++;
    return;
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{disconnect}
// --------------------------------------------------------------------
void lightgrad::TensorFloat::disconnect(){
    if (this->exist){
        this->struct_ptr->count--;
        if (this->struct_ptr->count == 0){
            delete this->struct_ptr;
        }
        this->exist = false;
        this->struct_ptr = nullptr;
    }
    return;
}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{allocate}
// ------------------------------------------------------------------
void lightgrad::TensorFloat::allocate(const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatStruct);
    TensorFloatStruct &tensorS = *(this->struct_ptr);

    // (2) Set size and shape of the array
    tensorS.size = 1;
    tensorS.shape = shape_;
    for (const auto &length : tensorS.shape){
        tensorS.size *= length;
    }

    // (3) Set the parameters
    tensorS.exist = true;
    tensorS.data = new float[tensorS.size];
    tensorS.grad_on = false;
    tensorS.grad = TensorFloat();
    tensorS.created = false;
    tensorS.creator = nullptr;

    return;

}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{from_scalar}
// ---------------------------------------------------------------------
void lightgrad::TensorFloat::from_scalar(const float scalar){

    // (1) Connect new data
    this->connect(new TensorFloatStruct);
    TensorFloatStruct &tensorS = *(this->struct_ptr);

    // (2) Set size and shape of the array
    tensorS.size = 1;
    tensorS.shape = {};

    // (3) Set the parameters
    tensorS.exist = true;
    tensorS.data = new float[tensorS.size];
    tensorS.grad_on = false;
    tensorS.grad = TensorFloat();
    tensorS.created = false;
    tensorS.creator = nullptr;

    // (4) Set data
    tensorS.data[0] = scalar;

    return;

}

void lightgrad::TensorFloat::from_scalar(const float scalar, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatStruct);
    TensorFloatStruct &tensorS = *(this->struct_ptr);

    // (2) Set size and shape of the array
    tensorS.size = 1;
    tensorS.shape = shape_;
    for (const auto &length : tensorS.shape){
        tensorS.size *= length;
    }

    // (3) Set the parameters
    tensorS.exist = true;
    tensorS.data = new float[tensorS.size];
    tensorS.grad_on = false;
    tensorS.grad = TensorFloat();
    tensorS.created = false;
    tensorS.creator = nullptr;

    // (4) Set data
    for (size_t i = 0; i < tensorS.size; i++){
        tensorS.data[i] = scalar;
    }

    return;

}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{from_array}
// --------------------------------------------------------------------
void lightgrad::TensorFloat::from_array(const float *array, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatStruct);
    TensorFloatStruct &tensorS = *(this->struct_ptr);

    // (2) Check for element existence of the array
    if (array == nullptr){
        std::cerr << "Error: The pointer of array is null." << std::endl;
        std::exit(1);
    }

    // (3) Set size and shape of the array
    tensorS.size = 1;
    tensorS.shape = shape_;
    for (const auto &length : tensorS.shape){
        tensorS.size *= length;
    }

    // (4) Set the parameters
    tensorS.exist = true;
    tensorS.data = new float[tensorS.size];
    tensorS.grad_on = false;
    tensorS.grad = TensorFloat();
    tensorS.created = false;
    tensorS.creator = nullptr;

    // (5) Set data
    for (size_t i = 0; i < tensorS.size; i++){
        tensorS.data[i] = array[i];
    }

    return;

}

void lightgrad::TensorFloat::from_array(const std::vector<float> &array, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatStruct);
    TensorFloatStruct &tensorS = *(this->struct_ptr);

    // (2) Check for element existence of the array
    if (array.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::exit(1);
    }

    // (3) Set size and shape of the array
    tensorS.size = 1;
    tensorS.shape = shape_;
    for (const auto &length : tensorS.shape){
        tensorS.size *= length;
    }
    if (array.size() != tensorS.size){
        std::cerr << "Error: The number of elements and shape are not equal." << std::endl;
        std::exit(1);
    }

    // (4) Set the parameters
    tensorS.exist = true;
    tensorS.data = new float[tensorS.size];
    tensorS.grad_on = false;
    tensorS.grad = TensorFloat();
    tensorS.created = false;
    tensorS.creator = nullptr;

    // (5) Set data
    for (size_t i = 0; i < tensorS.size; i++){
        tensorS.data[i] = array[i];
    }

    return;

}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{create}
// ----------------------------------------------------------------
void lightgrad::TensorFloat::create(Function * const creator_, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatStruct);
    TensorFloatStruct &tensorS = *(this->struct_ptr);

    // (2) Set size and shape of the array
    tensorS.size = 1;
    tensorS.shape = shape_;
    for (const auto &length : tensorS.shape){
        tensorS.size *= length;
    }

    // (3) Set the parameters
    tensorS.exist = true;
    tensorS.data = new float[tensorS.size];
    tensorS.grad_on = false;
    tensorS.grad = TensorFloat();
    tensorS.created = true;
    tensorS.creator = creator_;

    return;

}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{backward}
// ------------------------------------------------------------------
void lightgrad::TensorFloat::backward(TensorFloat grad_){
    if (!this->exist){
        std::cerr << "Error: Couldn't execute 'backward' because the member 'exist' of Tensor is false." << std::endl;
        std::exit(1);
    }
    if (this->struct_ptr->grad_on){
        this->grad() += grad_;
    }
    if (this->struct_ptr->created){
        this->struct_ptr->creator->backward(grad_);
    }
    return;
}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{new_grad}
// ------------------------------------------------------------------
void lightgrad::TensorFloat::new_grad(){
    if (this->exist){
        TensorFloatStruct &tensorS = *(this->struct_ptr);
        if (tensorS.exist){
            tensorS.grad_on = true;
            tensorS.grad.from_scalar(0.0, tensorS.shape);
        }
    }
    return;
}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{delete_grad}
// ---------------------------------------------------------------------
void lightgrad::TensorFloat::delete_grad(){
    if (this->exist){
        TensorFloatStruct &tensorS = *(this->struct_ptr);
        if (tensorS.exist){
            tensorS.grad_on = false;
            tensorS.grad = TensorFloat();
        }
    }
    return;
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{size}
// --------------------------------------------------------------
size_t lightgrad::TensorFloat::size(){
    if (!this->exist){
        return 0;
    }
    return this->struct_ptr->size;
}


// ---------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{shape}
// ---------------------------------------------------------------
std::vector<size_t> lightgrad::TensorFloat::shape(){
    if (!this->exist){
        return std::vector<size_t>();
    }
    return this->struct_ptr->shape;
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{data}
// --------------------------------------------------------------
float *lightgrad::TensorFloat::data(){
    if (!this->exist){
        return nullptr;
    }
    return this->struct_ptr->data;
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{grad}
// --------------------------------------------------------------
lightgrad::TensorFloat &lightgrad::TensorFloat::grad(){
    if (!this->exist){
        std::cerr << "Error: Couldn't obtain 'grad' because the member 'exist' of Tensor is false." << std::endl;
        std::exit(1);
    }
    return this->struct_ptr->grad;
}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{scalar}
// ----------------------------------------------------------------
float lightgrad::TensorFloat::scalar(){
    if (!this->exist){
        std::cerr << "Error: Couldn't obtain 'scalar' because the member 'exist' of Tensor is false." << std::endl;
        std::exit(1);
    }
    else if (this->struct_ptr->size != 1){
        std::cerr << "Error: The number of elements in the tensor is not 1." <<  " (size = " << this->struct_ptr->size << ")" << std::endl;
        std::exit(1);
    }
    return this->struct_ptr->data[0];
}


// ----------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> destructor
// ----------------------------------------------------------
lightgrad::TensorFloat::~TensorFloat(){
    this->disconnect();
}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloatStruct} -> destructor
// ----------------------------------------------------------------
lightgrad::TensorFloatStruct::~TensorFloatStruct(){

    if (this->count > 0){
        std::cerr << "Error: Couldn't destruct this class because the reference count is greater than 0." << std::endl;
        std::exit(1);
    }

    if (this->exist){
        delete[] this->data;
        if (this->created){
            delete creator;
        }
    }

}
