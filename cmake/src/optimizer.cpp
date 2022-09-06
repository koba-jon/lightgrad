#include <vector>
#include <lightgrad/tensor.hpp>
#include <lightgrad/optimizer.hpp>


// ---------------------------------------------------------------------
// namespace{lightgrad} -> class{SGD}(Optimizer) -> function{new_grad}
// ---------------------------------------------------------------------
void lightgrad::SGD::new_grad(){
    for (auto &param : this->params){
        param.new_grad();
    }
    return;
}


// ------------------------------------------------------------------------
// namespace{lightgrad} -> class{SGD}(Optimizer) -> function{delete_grad}
// ------------------------------------------------------------------------
void lightgrad::SGD::delete_grad(){
    for (auto &param : this->params){
        param.delete_grad();
    }
    return;
}


// ----------------------------------------------------------------
// namespace{lightgrad} -> class{SGD}(Optimizer) -> function{set}
// ----------------------------------------------------------------
void lightgrad::SGD::set(const std::vector<TensorFloat> &params_){
    this->delete_grad();
    this->params = params_;
    this->new_grad();
    return;
}

void lightgrad::SGD::set(const std::vector<TensorFloat> &params_, const float lr_){
    this->delete_grad();
    this->params = params_;
    this->lr = lr_;
    this->new_grad();
    return;
}


// ------------------------------------------------------------------
// namespace{lightgrad} -> class{SGD}(Optimizer) -> function{reset}
// ------------------------------------------------------------------
void lightgrad::SGD::reset(){
    this->new_grad();
    return;
}


// -------------------------------------------------------------------
// namespace{lightgrad} -> class{SGD}(Optimizer) -> function{update}
// -------------------------------------------------------------------
void lightgrad::SGD::update(){
    for (auto &param : this->params){
        float size = param.size();
        float *data = param.data();
        float *grad = param.param->grad.data();
        for (size_t i = 0; i < size; i++){
            data[i] -= this->lr * grad[i];
        }
    }
    return;
}


// --------------------------------------------------
// namespace{lightgrad} -> class{SGD} -> destructor
// --------------------------------------------------
lightgrad::SGD::~SGD(){
    this->delete_grad();
}
