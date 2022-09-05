#ifndef LIGHTGRAD_OPTIMIZER_HPP
#define LIGHTGRAD_OPTIMIZER_HPP

#include <vector>
#include <lightgrad/declare.hpp>


// ----------------------
// namespace{lightgrad}
// ----------------------
namespace lightgrad{


    // ------------------
    // class{Optimizer}
    // ------------------
    class Optimizer{
    
    private:

        // Function (pure virtual)
        virtual void new_grad() = 0;
        virtual void delete_grad() = 0;

    public:

        // Constructor
        Optimizer() = default;

        // Function (pure virtual)
        virtual void set(const std::vector<TensorFloat> &params_) = 0;
        virtual void reset() = 0;
        virtual void update() = 0;

        // Destructor (pure virtual)
        virtual ~Optimizer() = default;

    };


    // -----------------------
    // class{SGD}(Optimizer)
    // -----------------------
    class SGD : public Optimizer{

    private:

        // Member variable
        float lr = 0.1;
        std::vector<TensorFloat> params;

        // Function
        void new_grad();
        void delete_grad();

    public:

        // Constructor
        SGD() = default;  // Default

        // Function
        void set(const std::vector<TensorFloat> &params_) override;
        void set(const std::vector<TensorFloat> &params_, const float lr_);
        void reset() override;
        void update() override;

        // Destructor
        ~SGD();

    };


}


#endif