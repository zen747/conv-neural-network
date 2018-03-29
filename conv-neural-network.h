#ifndef CONV_NEURAL_NETWORK
#define CONV_NEURAL_NETWORK

#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include <ctime>
#include <cstdio>
#include <pthread.h>
#include "MesenneTwister.h"

//-----------
class Layer;
class ConvNeuralNetwork;

typedef std::vector<std::vector<std::vector<double> > > volume;

struct NetworkConfig
{
    bool training_;
    bool regularization_on_;
    double input_dropout_rate_;
    double learning_rate_;
    double regularization_factor_;
    double max_norm_;
    int    batch_size_;
    
    NetworkConfig()
    : training_(false)
    , regularization_on_(true)
    , input_dropout_rate_(0)
    , learning_rate_(0.001)
    , regularization_factor_(0.01) // L2 regularization
    , max_norm_(1e4)
    , batch_size_(100)
    {}
};

struct Feature
{
    int nRow_;
    int nCol_;
    int nChannel_;
    std::vector<double> weights_;
    std::vector<double> pdrvErrorWrtWeights_;
    std::vector<double> deltaWeight_;
    std::vector<double> momentum_;
    NetworkConfig *network_config_;

    void init_weights (double mean, double variance, double magnitude);
    size_t set_input_weights(const double* weights);
    void update_weights (double learning_rate, double regulariz, int numShares);
    void clear_pdrvs ();
};

class Neuron
{
public:
    Neuron ();

    void init_weights (int nWeights, double mean, double variance, double magnitude);
    void calculate_input_sum_from (const volume & inputs);
    void calculate_input_sum_with_feature_from (const volume & inputs, int xoffset,
                                               int yoffset, Feature const&feature);
    double calculate_output_of_pool_from (const volume & inputs, int szFilterStride, int szFilterSize);
    void calculate_pdrv_of_error_wrt_input_weights(const std::vector<double> &inputs, double drvLoss);
    void calculate_pdrv_wrt_output(const Layer &layer, const Layer &output_layer);
    void calculate_pdrv_of_input_weights (Layer &layer, const Layer &input_layer);
    const std::vector<double> & weights() const;
    size_t set_input_weights(const double *weights);
    void clear_pdrvs ();
    void update_weights (double learning_rate, double regulariz);
    void set_coordinate (int x, int y, int z);
    double activate (double output_rate);
    
public:
    NetworkConfig *network_config_;
    bool pooling_;
    bool droped_out_;
    char idxPooledX_;
    char idxPooledY_;
    
    std::vector<double> inputWeights_;
    double         input_sum_;
    double         output_value_;
    int            x_, y_, z_;
    
    double         pdrvErrorWrtInput_;
    
    std::vector<double> deltaWeight_;
    std::vector<double> pdrvErrorWrtWeights_;
    std::vector<double> momentum_;
};

class Layer
{
public:
    enum LayerType {
        eUnknownLayer,
        eInputLayer,
        eConvLayer,
        ePoolLayer,
        eFCLayer,
        eOutputLayer
    };
    
public:
    Layer(NetworkConfig * config, int nRow, int nCol, int nChannel);
    ~Layer();

    void init_weights (double mean, double variance, double magnitude, const Layer &last_layer);
    void forward_pass(const Layer &last_layer);
    void calculate_output_layer_pdrv_of_error_wrt_input_weights (std::vector<double> const&inputs,
                                                                 std::vector<double> const&outputsExp,
                                                                 double totalExp,
                                                                 int right_ans
                                                                );
    void calculate_middle_layer_pdrv_of_error_wrt_input_weights (const Layer &input_layer, const Layer &output_layer);
    
    double calculate_pdrv_for_input_neuron (int serial, int x, int y, int z) const;
    double calculate_pdrv_for_input_neuron_to_pool (int x, int y, int z) const;
    double calculate_pdrv_for_input_neuron_to_conv (int x, int y, int z) const;
    
    void calculate_normal();
    void calculate_adjusted_normal();
    void normalize();
    
    int serial_of_neuron (const Neuron &neuron) const;
    const Neuron &get_neuron_at_serial (int no) const;
    Neuron &get_neuron_at_serial (int no);
    const Neuron &get_neuron_at (int x, int y, int z) const;
    Neuron &get_neuron_at (int x, int y, int z);
    
    void save_weights_to(std::vector<double> &weights) const;
    size_t set_input_weights(const double *weights);
    const volume & get_outputs() const;
    void set_outputs_of_input_layer(const volume &value);
    size_t num_of_neurons () const;
    void clear_pdrvs ();
    void update_weights (double learning_rate, double regulariz);
    void mark_dropout ();
    void mark_no_dropout ();
    void set_dropout_rate (double r);
    double dropout_rate () const { return dropout_rate_; }
    
public:
    NetworkConfig *network_config_;
    std::vector<std::vector<std::vector<Neuron> > > nodes_;
    volume outputs_;
    std::vector<Feature> features_;
    bool share_weights_;
    
    int nRow_;
    int nCol_;
    int nChannel_;
    
    int szFilterSize_, szFilterStride_, szFilterPadding_;
    
    LayerType etLayer_;
    int idxLayer_;
    double dropout_rate_;
    double output_rate_;
    double mean_;
    double deviation_;
};

struct ActionData
{
    bool work_done_;
    int layer_;
    size_t begin_index_;
    size_t end_index_;
    enum {
        eForward,
        eBackward
    } type_;
    ActionData()
    : work_done_(false)
    {}
};

class ConvNeuralNetwork
{
public:
    ConvNeuralNetwork();
    ~ConvNeuralNetwork();
    Layer * add_input_layer(int width, int height, int nChannel, double dropout_rate=0);
    Layer * add_conv_layer(int nFilters, int szFilterSize, int nStride, int nPadding, double dropout_rate=0);
    Layer * add_pool_layer(int szFilterSize, int stride);
    void    add_fc_layer (const std::vector<int> &topology, double dropout_rate=0);
    Layer * add_fc_layer (int nNeurons, double dropout_rate=0);
    Layer * add_output_layer (int nOutputs);
    void set_learning_rate (double learning_rate);
    double learning_rate () const { return network_config_->learning_rate_; }
    void set_batch_size (int bs);
    int batch_size () const { return network_config_->batch_size_;}
    void set_regularization_factor (double v);
    double regularization_factor () const { return network_config_->regularization_factor_; }
    bool training () const { return network_config_->training_; }
    void set_regularization_on (bool yes);
    double regularization_on () const { return network_config_->regularization_on_; }
    void set_max_norm (double v);
    double max_norm () const { return network_config_->max_norm_; }
    void report_config () const;
    
    void init_weights(double mean_value, double variance_value, double magnitude);
    size_t number_of_weights () const;
    void run (volume const &inputs, std::vector<double> &outputs);
    void train (volume const &inputs, std::vector<double> &outputs);
    void forward_pass (volume const &inputs);
    void backward_pass (std::vector<double> &outputs, int right_ans);
    void save_weights_to(std::vector<double> &weights) const;
    void set_weights(std::vector<double> const &weights);
    void clear_pdrvs ();
    void update_weights ();
    double error () const { return last_error_; }
    size_t number_of_layers () const;
    const Neuron &get_neuron_at (int layerNo, int x, int y, int z) const;
    Neuron &get_neuron_at (int layerNo, int x, int y, int z);

    static ConvNeuralNetwork *load_network_file(const std::string &network_file);
    static void save_network_file (ConvNeuralNetwork*network, const std::string&network_file);
    static ConvNeuralNetwork * construct_cnn_with_config_file (const std::string& config_file);
    static int run_generation;
    
protected:
    void feed_forward ();
    void mark_dropout ();
    void mark_no_dropout ();
    
private:
    bool constructed_;
    std::vector<Layer *> layers_;
    double last_error_;
    NetworkConfig *network_config_;
    
    ConvNeuralNetwork(const ConvNeuralNetwork &rhs) {}
    ConvNeuralNetwork &operator=(const ConvNeuralNetwork &rhs) { return *this; }
};

extern MTRand g_rng;

#endif
