#include <iostream>
#include <vector>
#include <cstdlib>
#include <lightgrad/tensor.hpp>
#include <lightgrad/operator.hpp>


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


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> subscript operator
// ------------------------------------------------------------------
lightgrad::TensorFloat lightgrad::TensorFloat::operator[](const size_t idx){
    return Subscript().forward(*this, idx);
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
    TensorFloatParam &tensorP = *(tensor.param);
    if (!tensorP.exist){
        os << "[Not exist]";
        return os;
    }

    // (2) Describe data of the tensor
    for (size_t i = 0; i < tensorP.size; i++){
        os << tensorP.data[i] << " ";
    }
    os << "\n";

    // (3) Describe information of the tensor
    os << "[";
    /******************************************/
    os << "Type: Float";
    /******************************************/
    os << ", " << "Shape: ";
    if (tensorP.shape.size() == 0){
        os << "Scalar";
    }
    else{
        os << tensorP.shape.size() << "D-Tensor";
        os << "(";
        for (size_t i = 0; i < tensorP.shape.size(); i++){
            if (i != 0){
                os << "x";
            }
            os << tensorP.shape[i];
        }
        os << ")";
    }
    /******************************************/
    os << ", " << "Number: " << tensorP.size;
    /******************************************/
    os << ", " << "Creator: ";
    if (tensorP.created){
        os << tensorP.creator->type_name();
    }
    else{
        os << "None";
    }
    /******************************************/
    os << ", " << "Grad: ";
    if (tensorP.grad_on){
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
    if (tensorI.exist) tensorO.connect(tensorI.param);
    return;
}


// -----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{connect}
// -----------------------------------------------------------------
void lightgrad::TensorFloat::connect(TensorFloatParam * const struct_ptr_){
    this->disconnect();
    this->exist = true;
    this->param = struct_ptr_;
    this->param->count++;
    return;
}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{disconnect}
// --------------------------------------------------------------------
void lightgrad::TensorFloat::disconnect(){
    if (this->exist){
        this->param->count--;
        if (this->param->count == 0){
            delete this->param;
        }
        this->exist = false;
        this->param = nullptr;
    }
    return;
}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{allocate}
// ------------------------------------------------------------------
void lightgrad::TensorFloat::allocate(const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatParam);
    TensorFloatParam &tensorP = *(this->param);

    // (2) Set size and shape of the array
    tensorP.size = 1;
    tensorP.shape = shape_;
    for (const auto &length : tensorP.shape){
        tensorP.size *= length;
    }

    // (3) Set the parameters
    tensorP.exist = true;
    tensorP.data = new float[tensorP.size];
    tensorP.grad_on = false;
    tensorP.grad = TensorFloat();
    tensorP.created = false;
    tensorP.creator = nullptr;

    return;

}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{from_scalar}
// ---------------------------------------------------------------------
void lightgrad::TensorFloat::from_scalar(const float scalar){

    // (1) Connect new data
    this->connect(new TensorFloatParam);
    TensorFloatParam &tensorP = *(this->param);

    // (2) Set size and shape of the array
    tensorP.size = 1;
    tensorP.shape = {};

    // (3) Set the parameters
    tensorP.exist = true;
    tensorP.data = new float[tensorP.size];
    tensorP.grad_on = false;
    tensorP.grad = TensorFloat();
    tensorP.created = false;
    tensorP.creator = nullptr;

    // (4) Set data
    tensorP.data[0] = scalar;

    return;

}

void lightgrad::TensorFloat::from_scalar(const float scalar, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatParam);
    TensorFloatParam &tensorP = *(this->param);

    // (2) Set size and shape of the array
    tensorP.size = 1;
    tensorP.shape = shape_;
    for (const auto &length : tensorP.shape){
        tensorP.size *= length;
    }

    // (3) Set the parameters
    tensorP.exist = true;
    tensorP.data = new float[tensorP.size];
    tensorP.grad_on = false;
    tensorP.grad = TensorFloat();
    tensorP.created = false;
    tensorP.creator = nullptr;

    // (4) Set data
    for (size_t i = 0; i < tensorP.size; i++){
        tensorP.data[i] = scalar;
    }

    return;

}


// --------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{from_array}
// --------------------------------------------------------------------
void lightgrad::TensorFloat::from_array(const float *array, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatParam);
    TensorFloatParam &tensorP = *(this->param);

    // (2) Check for element existence of the array
    if (array == nullptr){
        std::cerr << "Error: The pointer of array is null." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::from_array)" << std::endl;
        std::exit(1);
    }

    // (3) Set size and shape of the array
    tensorP.size = 1;
    tensorP.shape = shape_;
    for (const auto &length : tensorP.shape){
        tensorP.size *= length;
    }

    // (4) Set the parameters
    tensorP.exist = true;
    tensorP.data = new float[tensorP.size];
    tensorP.grad_on = false;
    tensorP.grad = TensorFloat();
    tensorP.created = false;
    tensorP.creator = nullptr;

    // (5) Set data
    for (size_t i = 0; i < tensorP.size; i++){
        tensorP.data[i] = array[i];
    }

    return;

}

void lightgrad::TensorFloat::from_array(const std::vector<float> &array, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatParam);
    TensorFloatParam &tensorP = *(this->param);

    // (2) Check for element existence of the array
    if (array.size() == 0){
        std::cerr << "Error: The number of elements in the array is zero." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::from_array)" << std::endl;
        std::exit(1);
    }

    // (3) Set size and shape of the array
    tensorP.size = 1;
    tensorP.shape = shape_;
    for (const auto &length : tensorP.shape){
        tensorP.size *= length;
    }
    if (array.size() != tensorP.size){
        std::cerr << "Error: The number of elements and shape are not equal." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::from_array)" << std::endl;
        std::exit(1);
    }

    // (4) Set the parameters
    tensorP.exist = true;
    tensorP.data = new float[tensorP.size];
    tensorP.grad_on = false;
    tensorP.grad = TensorFloat();
    tensorP.created = false;
    tensorP.creator = nullptr;

    // (5) Set data
    for (size_t i = 0; i < tensorP.size; i++){
        tensorP.data[i] = array[i];
    }

    return;

}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{create}
// ----------------------------------------------------------------
void lightgrad::TensorFloat::create(Function * const creator_, const std::vector<size_t> &shape_){

    // (1) Connect new data
    this->connect(new TensorFloatParam);
    TensorFloatParam &tensorP = *(this->param);

    // (2) Set size and shape of the array
    tensorP.size = 1;
    tensorP.shape = shape_;
    for (const auto &length : tensorP.shape){
        tensorP.size *= length;
    }

    // (3) Set the parameters
    tensorP.exist = true;
    tensorP.data = new float[tensorP.size];
    tensorP.grad_on = false;
    tensorP.grad = TensorFloat();
    tensorP.created = true;
    tensorP.creator = creator_;

    return;

}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{backward}
// ------------------------------------------------------------------
void lightgrad::TensorFloat::backward(TensorFloat grad_){
    if (!this->exist){
        std::cerr << "Error: Couldn't execute 'backward' because the member 'exist' of Tensor is false." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::backward)" << std::endl;
        std::exit(1);
    }
    if (this->param->grad_on){
        this->grad() += grad_;
    }
    if (this->param->created){
        this->param->creator->backward(grad_);
    }
    return;
}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{new_grad}
// ------------------------------------------------------------------
void lightgrad::TensorFloat::new_grad(){
    if (this->exist){
        TensorFloatParam &tensorP = *(this->param);
        if (tensorP.exist){
            tensorP.grad_on = true;
            tensorP.grad.from_scalar(0.0, tensorP.shape);
        }
    }
    return;
}


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{delete_grad}
// ---------------------------------------------------------------------
void lightgrad::TensorFloat::delete_grad(){
    if (this->exist){
        TensorFloatParam &tensorP = *(this->param);
        if (tensorP.exist){
            tensorP.grad_on = false;
            tensorP.grad = TensorFloat();
        }
    }
    return;
}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{detach}
// ----------------------------------------------------------------
lightgrad::TensorFloat lightgrad::TensorFloat::detach(){

    if (!this->exist){
        return TensorFloat();
    }

    TensorFloat tensorO;
    /****************************************/
    TensorFloatParam &tensorIS = *(this->param);
    /****************************************/
    tensorO.connect(new TensorFloatParam);
    TensorFloatParam &tensorOS = *(tensorO.param);
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


// ---------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{clone}
// ---------------------------------------------------------------
lightgrad::TensorFloat lightgrad::TensorFloat::clone(){

    if (!this->exist){
        return TensorFloat();
    }

    TensorFloat tensorO;
    /****************************************/
    TensorFloatParam &tensorIS = *(this->param);
    /****************************************/
    tensorO.connect(new TensorFloatParam);
    TensorFloatParam &tensorOS = *(tensorO.param);
    tensorOS.exist = tensorIS.exist;
    tensorOS.size = tensorIS.size;
    tensorOS.shape = tensorIS.shape;
    tensorOS.grad_on = tensorIS.grad_on;
    tensorOS.grad = tensorIS.grad.clone();
    tensorOS.created = tensorIS.created;
    if (tensorIS.created){
        tensorOS.creator = tensorIS.creator->clone();
    }

    tensorOS.data = new float[tensorOS.size];
    for (size_t i = 0; i < tensorOS.size; i++){
        tensorOS.data[i] = tensorIS.data[i];
    }

    return tensorO;
    
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{size}
// --------------------------------------------------------------
size_t lightgrad::TensorFloat::size(){
    if (!this->exist){
        return 0;
    }
    return this->param->size;
}


// ---------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{shape}
// ---------------------------------------------------------------
std::vector<size_t> lightgrad::TensorFloat::shape(){
    if (!this->exist){
        return std::vector<size_t>();
    }
    return this->param->shape;
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{data}
// --------------------------------------------------------------
float *lightgrad::TensorFloat::data(){
    if (!this->exist){
        return nullptr;
    }
    return this->param->data;
}


// --------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{grad}
// --------------------------------------------------------------
lightgrad::TensorFloat &lightgrad::TensorFloat::grad(){
    if (!this->exist){
        std::cerr << "Error: Couldn't obtain 'grad' because the member 'exist' of Tensor is false." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::grad)" << std::endl;
        std::exit(1);
    }
    return this->param->grad;
}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> function{scalar}
// ----------------------------------------------------------------
float lightgrad::TensorFloat::scalar(){
    if (!this->exist){
        std::cerr << "Error: Couldn't obtain 'scalar' because the member 'exist' of Tensor is false." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::scalar)" << std::endl;
        std::exit(1);
    }
    else if (this->param->size != 1){
        std::cerr << "Error: The number of elements in the tensor is not 1." <<  " (size = " << this->param->size << ")" << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloat::scalar)" << std::endl;
        std::exit(1);
    }
    return this->param->data[0];
}


// ----------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloat} -> destructor
// ----------------------------------------------------------
lightgrad::TensorFloat::~TensorFloat(){
    this->disconnect();
}


// ---------------------------------------------------------------
// namespace{lightgrad} -> class{TensorFloatParam} -> destructor
// ---------------------------------------------------------------
lightgrad::TensorFloatParam::~TensorFloatParam(){

    if (this->count > 0){
        std::cerr << "Error: Couldn't destruct this class because the reference count is greater than 0." << std::endl;
        std::cerr << "       [tensor.cpp](lightgrad::TensorFloatParam::~TensorFloatParam)" << std::endl;
        std::exit(1);
    }

    if (this->exist){
        delete[] this->data;
        if (this->created){
            delete creator;
        }
    }

}
