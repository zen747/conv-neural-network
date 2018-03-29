#include "conv-neural-network.h"
#include <cfloat>
#include <fstream>
#include <sys/time.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

using namespace std;

#define DO_NORMALIZATION  1
#define ENABLE_MAX_NORM 1

#define LEAKY_RELU_CONSTANT    0.005

const double epsilon = 1e-8;
int ConvNeuralNetwork::run_generation = 0;

MTRand g_rng(time(NULL));

class Uncopyable
{
protected:
    Uncopyable () {}
    ~Uncopyable () {}
private:
    Uncopyable (Uncopyable const&rhs);
    Uncopyable &operator=(Uncopyable const&rhs);
};

class FileCloser: Uncopyable
{
    FILE *f_;
public:
    FileCloser(FILE *f)
    :f_(f)
    {}
    
    ~FileCloser()
    {
        fclose(f_);
    }
};
//--------------------------------

template <int ACTIVATE_FUNC_TYPE>
double activate (double x)
{
    return 0;
}

template <int ACTIVATE_FUNC_TYPE>
double activate_drv (double y)
{
    return 0;
}

enum {
    SIGMOID,
    TANH,
    RELU
};

template<>
double activate<SIGMOID>(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

template<>
double activate_drv<SIGMOID> (double y)
{
    return y * (1.0 - y);
}

template<>
double activate<TANH>(double x)
{
    return tanh (x);
}

template<>
double activate_drv<TANH> (double y)
{
    return 1.0 - y * y;
}

template<>
double activate<RELU>(double x)
{
    return (x > 0) ? x : LEAKY_RELU_CONSTANT * x;
}

template<>
double activate_drv<RELU>(double y)
{
    return (y > 0) ? 1.0 : LEAKY_RELU_CONSTANT;
}

//---------
double activate_func (double x)
{
    return activate<RELU>(x);
}

double activate_drv_func (double y)
{
    return activate_drv<RELU>(y);
}

void Feature::init_weights(double mean, double variance, double magnitude)
{
    assert (weights_.empty() && "feature weights initialized!");
    if (!weights_.empty()) return;
    
    weights_.resize(nRow_ * nCol_ * nChannel_ + 1); // plus bias
    pdrvErrorWrtWeights_.resize(weights_.size());
    deltaWeight_.resize(weights_.size());
    momentum_.resize(weights_.size());

    double sqrtn = sqrt(2.0 * weights_.size()-1);
    for (size_t i=0; i < weights_.size()-1; ++i) {
        weights_[i] = magnitude * g_rng.randNorm(mean, variance) / sqrtn;
    }
}

size_t Feature::set_input_weights(const double* weights)
{
    assert (!weights_.empty());
    weights_.assign(weights, weights + weights_.size());
    return weights_.size();
}

void Feature::update_weights(double learning_rate, double regulariz, int numShares)
{
    double f = 1.0 / numShares;
    double norm = 0;
    for (size_t i=0; i < weights_.size(); ++i) {
        double dx = pdrvErrorWrtWeights_[i] * f;
        /* momentum
        deltaWeight_[i] = momentum * deltaWeight_[i] - learning_rate * dx;
        weights_[i] += deltaWeight_[i];
        //*/
        
        /* Nesterov momentum
        double deltaWeightKeep = deltaWeight_[i];
        deltaWeight_[i] = momentum * deltaWeight_[i] - learning_rate * dx;
        weights_[i] += -momentum * deltaWeightKeep + (1 + momentum) * deltaWeight_[i];
        //*/
        
        /* adagrad
        deltaWeight_[i] += dx * dx;
        weights_[i] += -learning_rate * dx / (sqrt(deltaWeight_[i]) + epsilon);
        //*/
        
        /* RMSprop
        double decay_rate = 0.99;
        deltaWeight_[i] = decay_rate * deltaWeight_[i] + (1.0 - decay_rate) * dx * dx;
        weights_[i] += -learning_rate * dx / (sqrt(deltaWeight_[i]) + epsilon);
        //*/
        //* Adam simplified
        double beta1=0.9;
        double beta2=0.999;
        momentum_[i] = beta1*momentum_[i] + (1.0 - beta1) * dx;
        double mt = momentum_[i];
        deltaWeight_[i] = beta2 * deltaWeight_[i] + (1.0 - beta2) * dx * dx;
        double vt = deltaWeight_[i];
        weights_[i] += -learning_rate * mt / (sqrt(vt) + epsilon) - learning_rate * regulariz * weights_[i];
        //*/
        /* Adam
        double beta1=0.9;
        double beta2=0.999;
        momentum_[i] = beta1*momentum_[i] + (1.0 - beta1) * dx;
        double mt = momentum_[i] / (1.0 - pow(beta1, ConvNeuralNetwork::run_generation));
        deltaWeight_[i] = beta2 * deltaWeight_[i] + (1.0 - beta2) * dx * dx;
        double vt = deltaWeight_[i] / (1.0 - pow(beta2, ConvNeuralNetwork::run_generation));
        weights_[i] += -learning_rate * mt / (sqrt(vt) + epsilon) - learning_rate * regulariz * weights_[i];
        //*/
        
        norm += weights_[i] * weights_[i];
    }
    
#if ENABLE_MAX_NORM
    if (norm > network_config_->max_norm_ * network_config_->max_norm_) {
        norm = sqrt(norm);
        double r = network_config_->max_norm_ / norm;
        for (size_t i=0; i < weights_.size(); ++i) {
            weights_[i] *= r;
        }
    }
#endif
}

void Feature::clear_pdrvs()
{
    pdrvErrorWrtWeights_.clear();
    pdrvErrorWrtWeights_.resize(weights_.size());
}

//======================================================================================

Neuron::Neuron()
: pooling_(false)
, droped_out_(false)
, idxPooledX_(-1)
, idxPooledY_(-1)
{
}

void Neuron::init_weights(int nInputs, double mean, double variance, double magnitude)
{
    assert (inputWeights_.empty() && "weights initialized!");
    if (!inputWeights_.empty()) return;
    
    inputWeights_.resize(nInputs+1); // plus weight for bias
    pdrvErrorWrtWeights_.resize(inputWeights_.size());
    deltaWeight_.resize(inputWeights_.size());
    momentum_.resize(inputWeights_.size());
    
    double sqrtn = sqrt(2.0 * nInputs);
    for (size_t i=0; i < nInputs; ++i) {
        inputWeights_[i] = magnitude * g_rng.randNorm(mean, variance) / sqrtn;
    }
}

double Neuron::activate(double output_rate)
{
    output_value_ = activate_func(input_sum_) * output_rate;
    return output_value_;
}

void Neuron::calculate_input_sum_from (const volume& inputs)
{
    if (droped_out_) {
        input_sum_ = 0;
        return;
    }
    
    input_sum_=0;
    int idxW=0;
    for (size_t i=0; i < inputs.size(); ++i) {
        for (size_t j=0; j < inputs[i].size(); ++j) {
            for (size_t k=0; k < inputs[i][j].size(); ++k) {
                input_sum_ += inputs[i][j][k] * inputWeights_[idxW++];
            }
        }
    }
    input_sum_ += inputWeights_.back(); // bias
}

void Neuron::calculate_input_sum_with_feature_from(const vector< vector< vector< double > > >& inputs, int xoffset, int yoffset,
                                                  const Feature& feature
                                                 )
{
    if (droped_out_) {
        input_sum_ = 0;
        return;
    }
    
    double v = 0;
    int idxW=0;
    int xend = xoffset + feature.nCol_;
    int yend = yoffset + feature.nRow_;
    if (xoffset < 0) xoffset = 0;
    if (yoffset < 0) yoffset = 0;
    if (xend > inputs.size()) xend = inputs.size();
    if (yend > inputs[0].size()) yend = inputs[0].size();
    
    for (size_t i=xoffset; i < xend; ++i) {
        for (size_t j=yoffset; j < yend; ++j) {
            for (size_t k=0; k < inputs[i][j].size(); ++k) {
                v += inputs[i][j][k] * feature.weights_[idxW++];
            }
        }
    }
    v += feature.weights_.back(); // bias
    input_sum_ =  v;
}

double Neuron::calculate_output_of_pool_from(const vector< vector< vector< double > > >& inputs, int szFilterStride, int szFilterSize)
{
    int xoffset = x_ * szFilterStride;
    int yoffset = y_ * szFilterStride;
    double v=-DBL_MAX;
    int xend = xoffset + szFilterSize;
    int yend = yoffset + szFilterSize;
    for (size_t i=xoffset; i < xend; ++i) {
        for (size_t j=yoffset; j < yend; ++j) {
            if (v < inputs[i][j][z_]) {
                v = inputs[i][j][z_];
                idxPooledX_ = i;
                idxPooledY_ = j;
            }
        }
    }
    output_value_ = v;
    return v;
}


void Neuron::calculate_pdrv_of_error_wrt_input_weights(const vector<double> &inputs, double drvLoss)
{
    //assert(layer->etLayer_ == Layer::eOutputLayer);
    pdrvErrorWrtInput_ = drvLoss * activate_drv_func(output_value_);
    size_t i=0;
    for (; i < pdrvErrorWrtWeights_.size() - 1; ++i) {
        pdrvErrorWrtWeights_[i] += pdrvErrorWrtInput_ * inputs[i];
    }
    pdrvErrorWrtWeights_[i] += pdrvErrorWrtInput_; // and bias which is 1.0
}

void Neuron::calculate_pdrv_wrt_output(const Layer& layer, const Layer& output_layer)
{
    if (droped_out_) {
        pdrvErrorWrtInput_ = 0;
        return;
    }
    
    double drv = activate_drv_func(output_value_);
    if (layer.etLayer_ == Layer::ePoolLayer) {
        drv = 1.0;
    }

    if (output_layer.etLayer_ == Layer::eOutputLayer || output_layer.etLayer_ == Layer::eFCLayer) {
        double d = output_layer.calculate_pdrv_for_input_neuron (layer.serial_of_neuron(*this), x_, y_, z_);
        pdrvErrorWrtInput_ = d * drv / layer.deviation_;
    } else if (output_layer.etLayer_ == Layer::ePoolLayer) {
        double d = output_layer.calculate_pdrv_for_input_neuron_to_pool (x_, y_, z_);
        pdrvErrorWrtInput_ = d * drv;
    } else if (output_layer.etLayer_ == Layer::eConvLayer) {
        double d = output_layer.calculate_pdrv_for_input_neuron_to_conv (x_, y_, z_);
        pdrvErrorWrtInput_ = d * drv / layer.deviation_;
    }
}

void Neuron::calculate_pdrv_of_input_weights(Layer &layer, const Layer& input_layer)
{
    if (droped_out_) return;
    
    if (layer.etLayer_ == Layer::eFCLayer) { 
        size_t idxW=0;
        for (size_t i=0; i < input_layer.nRow_; ++i) {
            for (size_t j=0; j < input_layer.nCol_; ++j) {
                for (size_t k=0; k < input_layer.nChannel_; ++k) {
                    pdrvErrorWrtWeights_[idxW++] += pdrvErrorWrtInput_ * input_layer.get_neuron_at (i, j, k).output_value_;
                }
            }
        }
        pdrvErrorWrtWeights_[idxW] += pdrvErrorWrtInput_; // and bias which is 1.0
    } else if (layer.etLayer_ == Layer::eConvLayer) {
        const volume & inputs = input_layer.get_outputs();
        int xoffset = x_ * layer.szFilterStride_ - layer.szFilterPadding_;
        int yoffset = y_ * layer.szFilterStride_ - layer.szFilterPadding_;
        
        Feature &feature = layer.features_[z_];
        int idxW=0;
        int xend = xoffset + feature.nCol_;
        int yend = yoffset + feature.nRow_;
        if (xoffset < 0) xoffset = 0;
        if (yoffset < 0) yoffset = 0;
        if (xend > inputs.size()) xend = inputs.size();
        if (yend > inputs[0].size()) yend = inputs[0].size();
        
        for (size_t i=xoffset; i < xend; ++i) {
            for (size_t j=yoffset; j < yend; ++j) {
                for (size_t k=0; k < inputs[i][j].size(); ++k) {
                    feature.pdrvErrorWrtWeights_[idxW++] += pdrvErrorWrtInput_ * inputs[i][j][k];
                }
            }
        }
        feature.pdrvErrorWrtWeights_[idxW] += pdrvErrorWrtInput_; // and bias which is 1.0    
    }
}



const vector< double >& Neuron::weights() const
{
    return inputWeights_;
}

size_t Neuron::set_input_weights(const double* weights)
{
    if (inputWeights_.empty()) return 0;
    
    inputWeights_.assign(weights, weights + inputWeights_.size());
    pdrvErrorWrtWeights_.resize(inputWeights_.size());
    deltaWeight_.resize(inputWeights_.size());
    momentum_.resize(inputWeights_.size());
    return inputWeights_.size();
}

void Neuron::clear_pdrvs()
{
    pdrvErrorWrtWeights_.clear();
    pdrvErrorWrtWeights_.resize(inputWeights_.size());
}

void Neuron::update_weights(double learning_rate, double regulariz)
{
    double norm = 0;
    for (size_t i=0; i < inputWeights_.size(); ++i) {
        double dx = pdrvErrorWrtWeights_[i];
        /* momentum
        deltaWeight_[i] = momentum * deltaWeight_[i] - learning_rate * dx;
        inputWeights_[i] += deltaWeight_[i];
        //*/
        
        /* Nesterov momentum
        double deltaWeightKeep = deltaWeight_[i];
        deltaWeight_[i] = momentum * deltaWeight_[i] - learning_rate * dx;
        inputWeights_[i] += -momentum * deltaWeightKeep + (1 + momentum) * deltaWeight_[i];
        //*/
        
        /* adagrad
        deltaWeight_[i] += dx * dx;
        inputWeights_[i] += -learning_rate * dx / (sqrt(deltaWeight_[i]) + epsilon);
        //*/
        
        /* RMSprop
        double decay_rate = 0.99;
        deltaWeight_[i] = decay_rate * deltaWeight_[i] + (1.0 - decay_rate) * dx * dx;
        inputWeights_[i] += -learning_rate * dx / (sqrt(deltaWeight_[i]) + epsilon) - learning_rate * regulariz * inputWeights_[i];
        //*/
        //* Adam simplified
        double beta1=0.9;
        double beta2=0.999;
        momentum_[i] = beta1*momentum_[i] + (1.0 - beta1) * dx;
        double mt = momentum_[i];
        deltaWeight_[i] = beta2 * deltaWeight_[i] + (1.0 - beta2) * dx * dx;
        double vt = deltaWeight_[i];
        inputWeights_[i] += -learning_rate * mt / (sqrt(vt) + epsilon) - learning_rate * regulariz * inputWeights_[i];
        //*/
        /* Adam
        double beta1=0.9;
        double beta2=0.999;
        momentum_[i] = beta1*momentum_[i] + (1.0 - beta1) * dx;
        double mt = momentum_[i] / (1.0 - pow(beta1, ConvNeuralNetwork::run_generation));
        deltaWeight_[i] = beta2 * deltaWeight_[i] + (1.0 - beta2) * dx * dx;
        double vt = deltaWeight_[i] / (1.0 - pow(beta2, ConvNeuralNetwork::run_generation));
        inputWeights_[i] += -learning_rate * mt / (sqrt(vt) + epsilon) - learning_rate * regulariz * inputWeights_[i];
        //*/
        
        norm += inputWeights_[i] * inputWeights_[i];
    }
    /*
    for (int i=0; i < 10; ++i) {
        cout << pdrvErrorWrtWeights_[i] << " ";
    }
    cout << "\n"; 
    */
#if ENABLE_MAX_NORM
    if (norm > network_config_->max_norm_ * network_config_->max_norm_) {
        norm = sqrt(norm);
        double r = network_config_->max_norm_ / norm;
        for (size_t i=0; i < inputWeights_.size(); ++i) {
            inputWeights_[i] *= r;
        }
    }
#endif
}

void Neuron::set_coordinate(int x, int y, int z)
{
    x_ = x;
    y_ = y;
    z_ = z;
}

//================================================================================

Layer::Layer(NetworkConfig * config, int nRow, int nCol, int nChannel)
: network_config_(config)
, share_weights_(false)
, nRow_(nRow), nCol_(nCol), nChannel_(nChannel)
, etLayer_(eUnknownLayer)
, dropout_rate_(0)
, output_rate_(1.0)
, mean_(0)
, deviation_(1.0)
{
    nodes_.resize(nRow);
    for (int i=0; i < nRow; ++i) {
        nodes_[i].resize(nCol);
        for (int j=0; j < nCol; ++j) {
            nodes_[i][j].resize(nChannel);
            for (int k=0; k < nodes_[i][j].size(); ++k) {
                nodes_[i][j][k].network_config_ = config;
                nodes_[i][j][k].set_coordinate (i, j, k);
            }
        }
    }
    outputs_.resize(nRow);
    for (int i=0; i < nRow; ++i) {
        outputs_[i].resize(nCol);
        for (int j=0; j < nCol; ++j) {
            outputs_[i][j].resize(nChannel);
        }
    }
}

Layer::~Layer()
{
    nodes_.clear();
}

void Layer::init_weights(double mean, double variance, double magnitude, const Layer &last_layer)
{
    size_t nNeuronsOfLastLayer = last_layer.num_of_neurons();
    if (etLayer_ == eConvLayer) {
        for (size_t i=0; i < features_.size(); ++i) {
            features_[i].network_config_ = this->network_config_;
            features_[i].init_weights (mean, variance, magnitude);
        }
    } else if (etLayer_ == eInputLayer) {
    } else if (etLayer_ == ePoolLayer) {
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    nodes_[i][j][k].pooling_ = true;
                }
            }
        }
    } else { // eOutputLayer and eFCLayer
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    nodes_[i][j][k].init_weights (nNeuronsOfLastLayer, mean, variance, magnitude);
                }
            }
        }
    }
}

void Layer::forward_pass(const Layer &last_layer)
{
    const volume & inputs = last_layer.get_outputs();
    double output_rate = network_config_->training_ ? 1.0 : output_rate_;

    if (etLayer_ == eConvLayer) {
        int xoffset, yoffset;
        for (size_t i=0; i < nRow_; ++i) {
            xoffset = i * szFilterStride_ - szFilterPadding_;
            for (size_t j=0; j < nCol_; ++j) {
                yoffset = j * szFilterStride_ - szFilterPadding_;
                for (size_t k=0; k < nChannel_; ++k) {
                    nodes_[i][j][k].calculate_input_sum_with_feature_from (inputs, xoffset, yoffset, features_[k]);
                }
            }
        }
#if DO_NORMALIZATION
        calculate_normal();
        normalize ();
#endif 
        for (size_t i=0; i < nRow_; ++i) {
            xoffset = i * szFilterStride_ - szFilterPadding_;
            for (size_t j=0; j < nCol_; ++j) {
                yoffset = j * szFilterStride_ - szFilterPadding_;
                for (size_t k=0; k < nChannel_; ++k) {
                    outputs_[i][j][k] = nodes_[i][j][k].activate(output_rate);
                }
            }
        }
        
    } else if (etLayer_ == ePoolLayer) {
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    outputs_[i][j][k] = nodes_[i][j][k].calculate_output_of_pool_from (inputs, szFilterStride_, szFilterSize_);
                }
            }
        }
                
    } else {
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    nodes_[i][j][k].calculate_input_sum_from (inputs);
                }
            }
        }
#if DO_NORMALIZATION
        if (etLayer_ != eOutputLayer) {
            calculate_normal();            
            normalize();
        }
#endif
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    outputs_[i][j][k] = nodes_[i][j][k].activate(output_rate);
                }
            }
        }
        
    }
}

void Layer::calculate_normal()
{
    double nItems = nRow_ * nCol_ * nChannel_;
    double mean=0;
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                mean += nodes_[i][j][k].input_sum_;
            }
        }
    }
    mean_ = 0.9 * mean / nItems + 0.1 * mean_;
    
    double deviation = 0;
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                deviation += pow(nodes_[i][j][k].input_sum_ - mean_, 2);
            }
        }
    }
    deviation /= nItems;
    deviation_ = 0.9 * sqrt(deviation) + 0.1 * deviation_;
}

void Layer::calculate_adjusted_normal()
{
    double nItems = nRow_ * nCol_ * nChannel_;
    double mean=0;
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                mean += nodes_[i][j][k].input_sum_ * deviation_ + mean_;
            }
        }
    }
    mean_ = mean / nItems;
    
    deviation_ = 0;
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                deviation_ += pow(nodes_[i][j][k].input_sum_ - mean_, 2);
            }
        }
    }
    deviation_ /= nItems;
    deviation_ = sqrt(deviation_);
}

void Layer::normalize()
{
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                nodes_[i][j][k].input_sum_ = (nodes_[i][j][k].input_sum_ - mean_)/deviation_;
            }
        }
    }
}

void Layer::calculate_output_layer_pdrv_of_error_wrt_input_weights(vector<double> const&inputs,
                                                                   vector<double> const&outputsExp,
                                                                   double totalExp,
                                                                   int right_ans
                                                                  )
{
    assert (outputsExp.size() == nodes_[0][0].size());
    for (size_t k=0; k < nodes_[0][0].size(); ++k) {
        double drvLoss;
        if (k == right_ans) {
            drvLoss = outputsExp[k] / totalExp - 1.0;
        } else {
            drvLoss = outputsExp[k] / totalExp;
        }
        nodes_[0][0][k].calculate_pdrv_of_error_wrt_input_weights (inputs, drvLoss);
    }
}

void Layer::calculate_middle_layer_pdrv_of_error_wrt_input_weights(const Layer& input_layer, const Layer& output_layer)
{
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                nodes_[i][j][k].calculate_pdrv_wrt_output (*this, output_layer);
                nodes_[i][j][k].calculate_pdrv_of_input_weights (*this, input_layer);
            }
        }
    }
}

double Layer::calculate_pdrv_for_input_neuron(int serial, int x, int y, int z) const
{
    double d = 0;
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                d += nodes_[i][j][k].inputWeights_[serial] * nodes_[i][j][k].pdrvErrorWrtInput_;
            }
        }
    }

    return d;
}

double Layer::calculate_pdrv_for_input_neuron_to_pool(int x, int y, int z) const
{
    assert (etLayer_ == ePoolLayer);
    double d=0;
    int nx = x / this->szFilterStride_;
    int ny = y / this->szFilterStride_;
    Neuron const&neuron = nodes_[nx][ny][z];
    if (neuron.idxPooledX_ == x && neuron.idxPooledY_ == y) {
        d = neuron.pdrvErrorWrtInput_;
    }
    
    return d;
}

double Layer::calculate_pdrv_for_input_neuron_to_conv(int x, int y, int z) const
{
    assert (etLayer_ == eConvLayer);
    
    double d = 0;
    
    int nx = (x + szFilterPadding_) / szFilterStride_ - szFilterSize_;
    int ny = (y + szFilterPadding_) / szFilterStride_ - szFilterSize_;
    int nxend = nx + szFilterSize_ * 2;
    int nyend = ny + szFilterSize_ * 2;
    if (nx < 0) nx = 0;
    if (ny < 0) ny = 0;
    if (nxend >= nRow_) nxend = nRow_ - 1;
    if (nyend >= nCol_) nyend = nCol_ - 1;
    for (int i=nx; i < nxend; ++i) {
        for (int j=ny; j < nyend; ++j) {
            Neuron const&neuron = nodes_[i][j][z];
            d += neuron.pdrvErrorWrtInput_;
        }
    }
    return d;
}


int Layer::serial_of_neuron(const Neuron& neuron) const
{
    return neuron.z_  * this->nRow_ * this->nCol_ + neuron.y_ * this->nRow_ + neuron.x_;
}

Neuron& Layer::get_neuron_at_serial(int no)
{
    int z = no / (nRow_ * nCol_);
    int y = no / nRow_;
    int x = no % nRow_;
    return nodes_[x][y][z];
}

const Neuron& Layer::get_neuron_at_serial(int no) const
{
    int z = no / (nRow_ * nCol_);
    int y = no / nRow_;
    int x = no % nRow_;
    return nodes_[x][y][z];
}

const Neuron& Layer::get_neuron_at(int x, int y, int z) const
{
    return nodes_[x][y][z];
}

Neuron& Layer::get_neuron_at(int x, int y, int z)
{
    return nodes_[x][y][z];
}


void Layer::save_weights_to(vector< double >& weights) const
{
    if (share_weights_) {
        for (size_t i=0; i < features_.size(); ++i) {
            const vector<double> &w = features_[i].weights_;
            weights.insert(weights.end(), w.begin(), w.end());
        }
    } else {
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    const vector<double> &w = nodes_[i][j][k].weights();
                    weights.insert(weights.end(), w.begin(), w.end());
                }
            }
        }
    }
}

size_t Layer::set_input_weights(const double* weights)
{
    size_t pos = 0;
    if (share_weights_) {
        for (size_t i=0; i < features_.size(); ++i) {
            pos += features_[i].set_input_weights(weights + pos);
        }
    } else {
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    pos += nodes_[i][j][k].set_input_weights(weights + pos);
                }
            }
        }
    }
    return pos;
}

void Layer::set_outputs_of_input_layer(const volume & value)
{
    outputs_ = value;
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                if (nodes_[i][j][k].droped_out_) {
                    outputs_[i][j][k] = 0;
                }
                nodes_[i][j][k].output_value_ = outputs_[i][j][k];
            }
        }
    }
}

const volume& Layer::get_outputs() const
{
    return outputs_;
}


size_t Layer::num_of_neurons() const
{
    return nRow_*nCol_*nChannel_;
}

void Layer::clear_pdrvs()
{
    if (etLayer_ == eConvLayer) {
        for (size_t i=0; i < features_.size(); ++i) {
            features_[i].clear_pdrvs ();
        }
    } else {    
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    nodes_[i][j][k].clear_pdrvs();
                }
            }
        }
    }
}

void Layer::update_weights(double learning_rate, double regulariz)
{
    if (etLayer_ == eConvLayer) {
        int dimn = nRow_ * nCol_;
        for (size_t i=0; i < features_.size(); ++i) {
            features_[i].update_weights(learning_rate, regulariz, dimn);
        }
    } else {
        for (size_t i=0; i < nRow_; ++i) {
            for (size_t j=0; j < nCol_; ++j) {
                for (size_t k=0; k < nChannel_; ++k) {
                    nodes_[i][j][k].update_weights(learning_rate, regulariz);
                }
            }
        }
    }
    
#if DO_NORMALIZATION
    if (etLayer_ != ePoolLayer && etLayer_ != eOutputLayer) {
        //calculate_adjusted_normal();
    }
#endif

}

void Layer::mark_dropout()
{
    if (dropout_rate_ <= 0) return;
    
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                nodes_[i][j][k].droped_out_ = (g_rng.rand() < dropout_rate_);
            }
        }
    }
}

void Layer::mark_no_dropout()
{
    for (size_t i=0; i < nRow_; ++i) {
        for (size_t j=0; j < nCol_; ++j) {
            for (size_t k=0; k < nChannel_; ++k) {
                nodes_[i][j][k].droped_out_ = false;
            }
        }
    }
}

void Layer::set_dropout_rate(double r)
{
    dropout_rate_ = r;
    output_rate_ = 1.0 - r;
}


//===========================================================================================

ConvNeuralNetwork::ConvNeuralNetwork()
: constructed_(false)
, last_error_(-DBL_MAX)
, network_config_(new NetworkConfig)
{
    layers_.reserve(100);
}

ConvNeuralNetwork::~ConvNeuralNetwork()
{
    delete network_config_;
    
    for (size_t i=0; i < layers_.size(); ++i) {
        delete layers_[i];
    }
    layers_.clear();
}

Layer * ConvNeuralNetwork::add_input_layer(int width, int height, int nChannel, double dropout_rate)
{
    assert(!constructed_ && "network already constructed!");
    assert(layers_.empty() && "input layer must be added first!");
    Layer * l = new Layer(network_config_, width, height, nChannel);
    l->etLayer_ = Layer::eInputLayer;
    l->idxLayer_ = layers_.size();
    l->set_dropout_rate(dropout_rate);
    layers_.push_back(l);
    return l;
}

Layer * ConvNeuralNetwork::add_conv_layer(int nFilters, int szFilterSize, int nStride, int nPadding, double dropout_rate)
{
    assert(!constructed_ && "network already constructed!");
    assert(!layers_.empty() && "input layer must be added first!");
    const Layer &last_layer = *layers_.back();
    if ((last_layer.nCol_- szFilterSize + 2 * nPadding)%nStride != 0) {
        cout << "invalid config: invalid conv width";
        throw std::runtime_error("invalid conv config");
    }
    if ((last_layer.nRow_- szFilterSize + 2 * nPadding)%nStride != 0) {
        cout << "invalid config: invalid conv height";
        throw std::runtime_error("invalid conv config");
    }
    int width= (last_layer.nCol_- szFilterSize + 2 * nPadding)/nStride + 1;
    int height= (last_layer.nRow_ - szFilterSize + 2 * nPadding)/nStride + 1;
    int szChannel = last_layer.nChannel_;
    Layer * l = new Layer(network_config_, width, height, nFilters);
    l->etLayer_ = Layer::eConvLayer;
    l->share_weights_ = true;
    l->szFilterSize_ = szFilterSize;
    l->szFilterStride_ = nStride;
    l->szFilterPadding_ = nPadding;
    l->idxLayer_ = layers_.size();
    l->set_dropout_rate(dropout_rate);
    layers_.push_back(l);
    layers_.back()->features_.resize(nFilters);
    for (size_t i=0; i < layers_.back()->features_.size(); ++i) {
        layers_.back()->features_[i].nRow_ = szFilterSize;
        layers_.back()->features_[i].nCol_ = szFilterSize;
        layers_.back()->features_[i].nChannel_ = szChannel;
    }
    return l;
}

Layer * ConvNeuralNetwork::add_pool_layer(int szFilterSize, int stride)
{
    assert(!constructed_ && "network already constructed!");
    assert(!layers_.empty() && "input layer must be added first!");
    const Layer &last_layer = *layers_.back();
    assert((last_layer.nCol_ - szFilterSize)%stride == 0);
    assert((last_layer.nRow_ - szFilterSize)%stride == 0);
    if ((last_layer.nCol_ - szFilterSize)%stride != 0) {
        cout << "invalid pool config: invalid width";
        throw std::runtime_error("invalid pool config");
    }
    if ((last_layer.nRow_ - szFilterSize)%stride != 0) {
        cout << "invalid pool config: invalid height";
        throw std::runtime_error("invalid pool config");
    }
    
    int width= (last_layer.nCol_ - szFilterSize)/stride + 1;
    int height= (last_layer.nRow_ - szFilterSize)/stride + 1;
    Layer * l = new Layer(network_config_, width, height, last_layer.nChannel_);
    l->etLayer_ = Layer::ePoolLayer;
    l->szFilterSize_ = szFilterSize;
    l->szFilterStride_ = stride;
    l->idxLayer_ = layers_.size();
    layers_.push_back(l);

    return l;
}

void ConvNeuralNetwork::add_fc_layer(const vector< int >& topology, double dropout_rate)
{
    assert(!constructed_ && "network already constructed!");
    for (size_t i=0; i < topology.size(); ++i) {
        Layer * l = new Layer(network_config_, 1, 1, topology[i]);
        l->etLayer_ = Layer::eFCLayer;
        l->idxLayer_ = layers_.size();
        l->set_dropout_rate(dropout_rate);
        layers_.push_back(l);
    }
}

Layer * ConvNeuralNetwork::add_fc_layer(int nNeurons, double dropout_rate)
{
    assert(!constructed_ && "network already constructed!");
    Layer * l = new Layer(network_config_, 1, 1, nNeurons);
    l->etLayer_ = Layer::eFCLayer;
    l->idxLayer_ = layers_.size();
    l->set_dropout_rate(dropout_rate);
    layers_.push_back(l);
    
    return l;
}


Layer * ConvNeuralNetwork::add_output_layer(int nOutputs)
{
    assert(!constructed_ && "network already constructed!");
    this->constructed_ = true;
    Layer * l = new Layer(network_config_, 1, 1, nOutputs);
    l->etLayer_ = Layer::eOutputLayer;
    l->idxLayer_ = layers_.size();
    layers_.push_back(l);
    
    return l;
}

void ConvNeuralNetwork::set_learning_rate(double learning_rate)
{
    network_config_->learning_rate_ = learning_rate;
}

void ConvNeuralNetwork::set_batch_size(int bs)
{
    network_config_->batch_size_ = bs;
}

void ConvNeuralNetwork::set_regularization_factor(double v)
{
    network_config_->regularization_factor_ = v;
}

void ConvNeuralNetwork::set_regularization_on(bool yes)
{
    network_config_->regularization_on_ = yes;
}

void ConvNeuralNetwork::set_max_norm(double v)
{
    network_config_->max_norm_ = v;
}

void ConvNeuralNetwork::report_config() const
{
    cout << "\tLearningRate: " << learning_rate() << "\n";
    cout << "\tRegularizationFactor: " << regularization_factor() << "\n";
    cout << "\tBatchSize: " << batch_size() << "\n";
    cout << "\tTopology:\n";
    for (size_t i=0; i < layers_.size(); ++i) {
        Layer const&l = *layers_[i];
        cout << "\t\t";
        if (l.etLayer_ == Layer::eInputLayer) {
            cout << "Input " << l.nRow_ << " " << l.nCol_ << " " << l.nChannel_ << " " << l.dropout_rate_ << "\n"; 
        } else if (l.etLayer_ == Layer::eConvLayer) {
            cout << "Conv " << l.features_.size() << " " << l.szFilterSize_ << " " << l.szFilterStride_ << " " << l.szFilterPadding_ << " " << l.dropout_rate_ << "\n";
        } else if (l.etLayer_ == Layer::ePoolLayer) {
            cout << "Pool " << l.szFilterSize_ << " " << l.szFilterStride_ << "\n";
        } else if (l.etLayer_ == Layer::eFCLayer) {
            cout << "Full " << l.nChannel_ << " " << l.dropout_rate_ << "\n";
        } else if (l.etLayer_ == Layer::eOutputLayer) {
            cout << "Output " << l.nChannel_ << "\n";
        }
    }

}


void ConvNeuralNetwork::init_weights(double mean, double variance, double magnitude)
{
    for (size_t i=1; i < layers_.size(); ++i) {
        layers_[i]->init_weights(mean, variance, magnitude, *layers_[i-1]);
    }    
}

size_t ConvNeuralNetwork::number_of_weights() const
{
    vector<double> weights;
    this->save_weights_to(weights);
    return weights.size();
}

void ConvNeuralNetwork::run(const volume& inputs, vector< double >& outputs)
{
    assert (constructed_ && "network not constructed yet!");
    if (network_config_->training_) {
        network_config_->training_ = false;
        mark_no_dropout();
    }
    forward_pass(inputs);
    outputs = layers_.back()->get_outputs()[0][0];    
}

void ConvNeuralNetwork::train(const volume& inputs, vector< double >& outputs)
{
    assert (constructed_ && "network not constructed yet!");
    if (!network_config_->training_) {
        network_config_->training_ = true;
        //mark_dropout();
    }
    mark_dropout();
    forward_pass(inputs);
    outputs = layers_.back()->get_outputs()[0][0];    
}


void ConvNeuralNetwork::forward_pass(const volume& inputs)
{
    // feed forward
    // set inputs as outputs of first layer
    layers_[0]->set_outputs_of_input_layer(inputs);
    feed_forward ();
}

void ConvNeuralNetwork::backward_pass(vector< double >& outputs, int right_ans)
{
    // calculate errors/loss
    double denom = 0;
    double numer = 0;
    double max_out=-DBL_MAX;
    for (size_t i=0; i < outputs.size(); ++i) {
        if (outputs[i] > max_out) {
            max_out = outputs[i];
        }
    }
    
    vector<double> outputsExp; outputsExp.reserve(outputs.size());
    double totalExp = 0;
    for (size_t i=0; i < outputs.size(); ++i) {
        outputsExp.push_back(exp(outputs[i] - max_out));
        if (i == right_ans) {
            numer = outputsExp.back();
        }
        denom += outputsExp.back();
    }
    totalExp = denom;
    last_error_ = -log(numer / denom);
    if (isinf(last_error_)) {
       // cout << "infinity and beyond!" << endl;
    }
    
    // calculate partial derivatives
    // first the last layer(output layer)
    const vector<double> & layer_inputs = layers_[layers_.size()-2]->get_outputs()[0][0];
    /*
    cout << "\n";
    for (int i=0; i < layer_inputs.size() && i < 20; ++i) {
        cout << layer_inputs[i] << " ";
    }
    cout << "\n";
    for (size_t i=0; i < outputsExp.size(); ++i) {
        cout << outputsExp[i] << " ";
    } 
    cout << "\n";
    cout << totalExp << " " << right_ans;
    //*/
    layers_.back()->calculate_output_layer_pdrv_of_error_wrt_input_weights(layer_inputs, outputsExp, totalExp, right_ans);
    
    // back propagate
    for (int i=layers_.size()-2; i > 0; --i) {
        const Layer &prelayer = *layers_[i-1];
        const Layer &nextlayer = *layers_[i+1];
        layers_[i]->calculate_middle_layer_pdrv_of_error_wrt_input_weights(prelayer, nextlayer);
    }
    
}


void ConvNeuralNetwork::feed_forward()
{
    for (size_t i=1; i < layers_.size(); ++i) {
        layers_[i]->forward_pass(*layers_[i-1]);
    }    
}

void ConvNeuralNetwork::mark_dropout()
{
    for (size_t i=0; i < layers_.size()-1; ++i) {
        layers_[i]->mark_dropout();
    }    
}

void ConvNeuralNetwork::mark_no_dropout()
{
    for (size_t i=0; i < layers_.size()-1; ++i) {
        layers_[i]->mark_no_dropout();
    }    
}


void ConvNeuralNetwork::clear_pdrvs()
{
    for (size_t i=1; i < layers_.size(); ++i) {
        layers_[i]->clear_pdrvs();
    }
}

void ConvNeuralNetwork::update_weights()
{
    assert (network_config_->training_ && "why update weights when not training?");
    double regulariz = network_config_->regularization_on_ ? network_config_->regularization_factor_ : 0;
    for (size_t i=1; i < layers_.size(); ++i) {
        layers_[i]->update_weights(network_config_->learning_rate_, regulariz);
    }
    
    //mark_dropout ();
    
    this->clear_pdrvs();
}

void ConvNeuralNetwork::save_weights_to(vector< double >& weights) const
{
    weights.clear();
    for (size_t i=1; i < layers_.size(); ++i) {
        layers_[i]->save_weights_to (weights);
    }

}

void ConvNeuralNetwork::set_weights(const vector< double >& weights)
{
    int pos = 0;
    for (size_t i=1; i < layers_.size(); ++i) {
        pos += layers_[i]->set_input_weights (&weights[pos]);
    }
}

size_t ConvNeuralNetwork::number_of_layers() const
{
    return layers_.size();
}

const Neuron& ConvNeuralNetwork::get_neuron_at(int layerNo, int x, int y, int z) const
{
    return layers_[layerNo]->get_neuron_at (x, y, z);
}

Neuron& ConvNeuralNetwork::get_neuron_at(int layerNo, int x, int y, int z)
{
    return layers_[layerNo]->get_neuron_at (x, y, z);
}

ConvNeuralNetwork *ConvNeuralNetwork::load_network_file(const string &network_file)
{
    ConvNeuralNetwork *network=0;
    
    FILE *file = fopen(network_file.c_str(), "rb");
    if (file == NULL) {
        cout << "network file '" << network_file << "' can't be opened.\n";
        return network;
    }
    FileCloser fcloser(file);
    
    char buf[16]={0};
    size_t ret = fread(buf, 1, 6, file);
    if (buf != string("CNRN01")) {
        cout << "not my supported neural network format" << endl;
        return network;
    }
    
    enum FloatFormat {
        eFLOAT,
        eDOUBLE,
    } fformat;
    
    network = new ConvNeuralNetwork;
    size_t nItemRead;
    double dvalue;
    
    while (fread(buf, 1, 4, file) > 0) {
        if (buf == string("fmt")) {
            nItemRead = fread (buf, 1, 1, file);
            if (buf[0] == 'd') {
                fformat = eDOUBLE;
            } else if (buf[0] == 'f') {
                fformat = eFLOAT;
            } else {
                assert (0 && "parse file failed");
            }
        } else if (buf == string("geo")) {
            int32_t nLayers;
            nItemRead = fread (&nLayers, 1, sizeof(int32_t), file);
            for (int32_t idxL=0; idxL < nLayers; ++idxL) {
                int32_t tpLayer;
                nItemRead = fread (&tpLayer, 1, sizeof(int32_t), file);
                if (tpLayer == Layer::eInputLayer) {
                    int32_t row, col, channel;
                    nItemRead = fread (&row, 1, sizeof(int32_t), file);
                    nItemRead = fread (&col, 1, sizeof(int32_t), file);
                    nItemRead = fread (&channel, 1, sizeof(int32_t), file);
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    network->add_input_layer(row, col, channel, dvalue);
                } else if (tpLayer == Layer::eConvLayer) {
                    int32_t nFilters, szFilterSize, szFilterStride, szFilterPadding;
                    nItemRead = fread (&nFilters, 1, sizeof(int32_t), file);
                    nItemRead = fread (&szFilterSize, 1, sizeof(int32_t), file);
                    nItemRead = fread (&szFilterStride, 1, sizeof(int32_t), file);
                    nItemRead = fread (&szFilterPadding, 1, sizeof(int32_t), file);
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    Layer *l = network->add_conv_layer(nFilters, szFilterSize, szFilterStride, szFilterPadding, dvalue);
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    l->mean_ = dvalue;
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    l->deviation_ = dvalue;
                } else if (tpLayer == Layer::ePoolLayer) {
                    int32_t szFilterSize, szFilterStride;
                    nItemRead = fread (&szFilterSize, 1, sizeof(int32_t), file);
                    nItemRead = fread (&szFilterStride, 1, sizeof(int32_t), file);
                    network->add_pool_layer(szFilterSize, szFilterStride);
                } else if (tpLayer == Layer::eFCLayer) {
                    int32_t nNeurons;
                    nItemRead = fread (&nNeurons, 1, sizeof(int32_t), file);
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    Layer * l = network->add_fc_layer(nNeurons, dvalue);
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    l->mean_ = dvalue;
                    nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
                    l->deviation_ = dvalue;
                } else if (tpLayer == Layer::eOutputLayer) {
                    int32_t nNeurons;
                    nItemRead = fread (&nNeurons, 1, sizeof(int32_t), file);
                    network->add_output_layer(nNeurons);
                }
            }
        } else if (buf == string("wei")) {
            network->init_weights(0, 1.0, 1.0);
            int32_t numWeights;
            nItemRead = fread (&numWeights, 1, sizeof(int32_t), file);
            if (fformat == eFLOAT) {
                //vector<float> weights;
                //nItemRead = fread (&weights[0], numWeights, sizeof(double), file);
                //network->set_weights(weights);
            } else {
                vector<double> weights;
                weights.resize(numWeights);
                nItemRead = fread (&weights[0], numWeights, sizeof(double), file);
                network->set_weights(weights);
            }
        } else if (buf == string("trn")) {
            double dvalue;
            nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
            network->set_learning_rate(dvalue);
            int batch_size;
            nItemRead = fread(&batch_size, 1, sizeof(batch_size), file);
            network->set_batch_size(batch_size);
            nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
            network->set_regularization_factor(dvalue);
            nItemRead = fread (&dvalue, 1, sizeof(dvalue), file);
            network->set_max_norm(dvalue);
        } else {
            assert (0 && "format error!");
        }
    
    
    }
        
    cout << "network loaded from " << network_file << endl;
    return network;
}

void ConvNeuralNetwork::save_network_file (ConvNeuralNetwork*network, const string&network_file)
{
    FILE *file = fopen(network_file.c_str(), "wb");
    FileCloser fcloser(file);
    
    int32_t value;
    
    fwrite ("CNRN01", 1, 6, file); // convolution neural network format version 1
    fwrite ("fmt", 1, 4, file);
    if (sizeof(network->layers_[0]->nodes_[0][0][0]) == 4) {
        fwrite ("f", 1, 1, file); //<= double->d, float->f
    } else { // == 8
        fwrite ("d", 1, 1, file);
    }
    
    // block topology
    fwrite ("geo", 1, 4, file); // network topology
    value = int32_t(network->layers_.size());
    fwrite (&value, 1, sizeof(int32_t), file);
    
    size_t nItemWritten;
    double dvalue;
    
    for (size_t idxL=0; idxL < network->layers_.size(); ++idxL) {
        value = network->layers_[idxL]->etLayer_;
        nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
        if (network->layers_[idxL]->etLayer_ == Layer::eInputLayer) {
            value = network->layers_[idxL]->nRow_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            value = network->layers_[idxL]->nCol_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            value = network->layers_[idxL]->nChannel_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            dvalue = network->layers_[idxL]->dropout_rate_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
        } else if (network->layers_[idxL]->etLayer_ == Layer::eConvLayer) {
            value = network->layers_[idxL]->features_.size();
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            value = network->layers_[idxL]->szFilterSize_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            value = network->layers_[idxL]->szFilterStride_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            value = network->layers_[idxL]->szFilterPadding_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            dvalue = network->layers_[idxL]->dropout_rate_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
            dvalue = network->layers_[idxL]->mean_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
            dvalue = network->layers_[idxL]->deviation_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
        } else if (network->layers_[idxL]->etLayer_ == Layer::ePoolLayer) {
            value = network->layers_[idxL]->szFilterSize_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            value = network->layers_[idxL]->szFilterStride_;
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
        } else if (network->layers_[idxL]->etLayer_ == Layer::eFCLayer) {
            value = network->layers_[idxL]->nodes_[0][0].size();
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
            dvalue = network->layers_[idxL]->dropout_rate_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
            dvalue = network->layers_[idxL]->mean_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
            dvalue = network->layers_[idxL]->deviation_;
            nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
        } else if (network->layers_[idxL]->etLayer_ == Layer::eOutputLayer) {
            value = network->layers_[idxL]->nodes_[0][0].size();
            nItemWritten = fwrite (&value, 1, sizeof(int32_t), file);
        }
    }

    // block weights
    nItemWritten = fwrite("wei", 1, 4, file); // weights
    vector<double> weights;
    network->save_weights_to(weights);
    int32_t numWeights = weights.size();
    nItemWritten = fwrite (&numWeights, 1, sizeof(int32_t), file);
    nItemWritten = fwrite (&weights[0], weights.size(), sizeof(double), file);
    
    // block training setting
    nItemWritten = fwrite("trn", 1, 4, file); // training
    dvalue = network->learning_rate();
    nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
    int batch_size = network->batch_size();
    nItemWritten = fwrite (&batch_size, 1, sizeof(batch_size), file);
    dvalue = network->regularization_factor();
    nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);
    dvalue = network->max_norm();
    nItemWritten = fwrite (&dvalue, 1, sizeof(dvalue), file);

    fflush(file);
    
    cout << "saved network to " << network_file << endl;
}


class StringGetWord
{
public:
    StringGetWord (const char*buf, size_t start_pos=0, std::vector<char> const&seps=std::vector<char> ());

    void set_start_at (size_t pos) {
        pos_ = pos;
    }
    std::string next_word ();
    std::string remnant () {
        return bufstr_.substr (pos_);
    }
    std::string const&buffer () const {
        return bufstr_;
    }
    bool end () const {
        return pos_ >= bufstr_.length();
    }
    StringGetWord &operator>>(std::string &str);

    bool is_sep (char c) const;
    
private:
    std::string       bufstr_;
    std::vector<char> seps_;
    size_t            pos_;
};


StringGetWord::StringGetWord (const char*buf, size_t start_pos, std::vector<char> const&seps)
:bufstr_(buf), pos_(start_pos)
{
    if (!seps.empty()) {
        seps_ = seps;
    } else {
        seps_.reserve(4);
        seps_.push_back(' ');
        seps_.push_back('\t');
        seps_.push_back('\r');
        seps_.push_back('\n');
    }
}

bool StringGetWord::is_sep(char c) const
{
    for (size_t i=0; i < seps_.size(); ++i) {
        if (seps_[i] == c) return true;
    }
    return false;
}

std::string StringGetWord::next_word ()
{
    std::string str;
    str.reserve(16);
    bool encount=false;
    bool start = false;
    bool single_quote_active = false;
    bool double_quote_active = false;
    size_t i=pos_;
    for (; i < bufstr_.length(); ++i) {
        if (single_quote_active && bufstr_[i] == '\'') {
            if (start) {
                encount=true;
                single_quote_active = false;
            }
        } else if (double_quote_active && bufstr_[i] == '\"') {
            if (start) {
                encount=true;
                double_quote_active = false;
            }
        } else if (!single_quote_active && !double_quote_active && is_sep (bufstr_[i])) {
            if (start) {
                encount=true;
            }
        } else {
            if (encount) {
                break;
            } else {
                start = true;
                if (bufstr_[i] == '\"') {
                    double_quote_active = true;
                    ++i;
                } else if (bufstr_[i] == '\'') {
                    single_quote_active = true;
                    ++i;
                }
                str.push_back (bufstr_[i]);
            }
        }
    }
    if (!encount) {
        pos_ = bufstr_.length();
    } else {
        pos_ = i;
    }
    return str;
}

StringGetWord& StringGetWord::operator>>(string& str)
{
    if (this->end()) return *this;
    str = this->next_word();
    return *this;
}


ConvNeuralNetwork * ConvNeuralNetwork::construct_cnn_with_config_file (const string& config_file)
{
    ConvNeuralNetwork *network = new ConvNeuralNetwork;
    ifstream f;
    f.open(config_file.c_str(), ifstream::in);
    for (std::string line; std::getline(f, line); ) {
        StringGetWord worder(line.c_str());
        string type = worder.next_word();
        if (type == "Input") {
            int width = atoi(worder.next_word().c_str());
            int height = atoi(worder.next_word().c_str());
            int channelno = atoi(worder.next_word().c_str());
            double dropout_rate = atof (worder.next_word().c_str());
            network->add_input_layer(width, height, channelno, dropout_rate);
        } else if (type == "Conv") {
            int nFeatures = atoi(worder.next_word().c_str());
            int nFilterSize = atoi(worder.next_word().c_str());
            int nStride = atoi(worder.next_word().c_str());
            int nPadding = atoi(worder.next_word().c_str());
            double dropout_rate = atof (worder.next_word().c_str());
            network->add_conv_layer(nFeatures, nFilterSize, nStride, nPadding, dropout_rate);
        } else if (type == "Pool") {
            int nFilterSize = atoi(worder.next_word().c_str());
            int nStride = atoi(worder.next_word().c_str());
            network->add_pool_layer(nFilterSize, nStride);
        } else if (type == "Full") {
            int nNeurons = atoi(worder.next_word().c_str());
            double dropout_rate = atof (worder.next_word().c_str());
            network->add_fc_layer(nNeurons, dropout_rate);
        } else if (type == "Output") {
            int nNeurons = atoi(worder.next_word().c_str());
            network->add_output_layer(nNeurons);
        } else if (type == "LearningRate") {
            double learning_rate = atof(worder.next_word().c_str());
            network->set_learning_rate(learning_rate);
        } else if (type == "RegularizationFactor") {
            double v = atof(worder.next_word().c_str());
            network->set_regularization_factor(v);
        } else if (type == "MaxNorm") {
            double v = atof(worder.next_word().c_str());
            network->set_max_norm(v);
        } else if (type == "BatchSize") {
            int bs = atoi(worder.next_word().c_str());
            network->set_batch_size(bs);
        }
    }
    network->init_weights(0, 1.0, 1.0);
    f.close();
    return network;
}

