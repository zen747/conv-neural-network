/* 
 * a fizz buzz example
 * http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
 */

#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include <ctime>
#include <cfloat>
#include <sys/time.h>

#include "MesenneTwister.h"
#include "conv-neural-network.h"

using namespace std;

#define USE_DEEP 1

// The regularization scale the weights toward 0
#define REGULARIZATION_FACTOR  1e-3
#define DROPOUT_RATE           0.1
#define INPUT_DROPOUT_RATE     0
#define MAX_NORM               1000
#define LEARNING_RATE          0.001
#define BATCH_SIZE             32

struct Genome
{
    double fitness_;
    vector<double> genes_;
    
    Genome(size_t numGenes=0);
    void copy_genes_from (Genome const&source, size_t from_index, size_t to_index, size_t replace_start_index, double mutate_rate);
    void report () const;
    
    bool operator<(const Genome &rhs) const
    {
        return fitness_ > rhs.fitness_;
    }
};

Genome::Genome(size_t numGenes)
{
    assert (numGenes && "number of genes can't be zero");
    genes_.resize(numGenes);
}


void Genome::copy_genes_from(const Genome& source, size_t from_index, size_t to_index, size_t replace_start_index, double mutate_rate)
{
    for (size_t i=from_index,cgene=0; i < to_index; ++i,++cgene) {
        genes_[replace_start_index+cgene] = source.genes_[i];
    }
}

void Genome::report() const
{
    cout << "weights ";
    for (size_t i=0; i < genes_.size(); ++i) {
        cout << genes_[i] << " ";
    }
}

const int numInputPoints = 11;
const int numOutputPoints = 4;


struct Data {
    int value_;
    volume inputs_;        // binray representation
    
    int class_;// classify into 4 classes. 0 - no-change, 1 - fizz, 2 - buzz, 3 - fizzbuzz
};

struct DataSet
{
    vector<Data> samples_;
    void randomize ()
    {
        vector<Data> sam;
        while (!samples_.empty()) {
            int idx = g_rng.randInt() % samples_.size();
            sam.push_back(samples_[idx]);
            samples_[idx] = samples_.back();
            samples_.pop_back();
        }
        sam.swap(samples_);
    }
};

DataSet sData;
DataSet sDataTest;

Data encode_binary (int value)
{
    Data d;
    
    d.value_ = value;
    
    d.inputs_.resize(1);
    d.inputs_[0].resize(1);
    d.inputs_[0][0].reserve(numInputPoints);
    for (int i=0; i < numInputPoints; ++i) {
        int bin = (1 << i);
        if (value & bin) {
            d.inputs_[0][0].push_back(1.0);
        } else {
            d.inputs_[0][0].push_back(0);
        }
    }
    
    bool fizz = (value % 3 == 0);
    bool buzz = (value % 5 == 0);
    if (fizz && buzz) {
        d.class_ = 3;
    } else if (buzz) {
        d.class_ = 2;
    } else if (fizz) {
        d.class_ = 1;
    } else {
        d.class_ = 0;
    }
    
    return d;
    
}

void generate_data_set (DataSet &sData, int iFrom, int iTo)
{
    for (int iC=iFrom; iC <= iTo; ++iC) {
        sData.samples_.push_back(encode_binary(iC));
    }
}

int answer_from_outputs (vector<double> const&outputs)
{
    double max_term=-DBL_MAX;
    int ans = -1;
    for (size_t i=0; i < outputs.size(); ++i) {
        if (outputs[i] > max_term) {
            max_term = outputs[i];
            ans = i;
        }
    }
    return ans;
}

bool answer_correct (vector<double> const&outputs, int right_class)
{
    return answer_from_outputs(outputs) == right_class;
}

int test_check (ConvNeuralNetwork &network, bool print=false)
{
    vector<double> outputs(numOutputPoints);

    cout << "test...  ";
    if (sDataTest.samples_.empty()) generate_data_set (sDataTest, 1, 100);
    int correct_answer = 0;
    for (size_t i=0; i < sDataTest.samples_.size(); ++i) {
        const Data &d = sDataTest.samples_[i];
        network.run(d.inputs_, outputs);
        int ans = answer_from_outputs(outputs);
        if (ans == d.class_) {
            ++correct_answer;
        }
        if (print) {
            if (ans == 0) {
                cout << d.value_ << " ";
            } else if (ans == 1) {
                cout << "fizz" << " ";
            } else if (ans == 2) {
                cout << "buzz" << " ";
            } else if (ans == 3) {
                cout << "fizzbuzz" << " ";
            } else {
                assert (0 && "what happened?");
            }
        }
    }
    cout << "test-set precision: " << (100.0 * double(correct_answer) / sDataTest.samples_.size()) << "%" << endl;
    
    return correct_answer;
}

std::string timestamp_repr (timeval const * tv=0, bool with_milli_seconds=true)
{
    timeval tvl;
    if (!tv) {
        gettimeofday (&tvl, 0);
        tv = &tvl;
    }

    time_t t = tv->tv_sec;
    struct tm *tmc = localtime (&t);
    char time_buf[32] = {0};
    char buf[32] = {0};
    strftime (time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tmc);
    if (with_milli_seconds) {
        int millis = tv->tv_usec / 1000;
#if defined(WIN32)
        _snprintf (buf, 31, "%s.%03d", time_buf, millis);
#else
        snprintf (buf, 31, "%s.%03d", time_buf, millis);
#endif
        return std::string(buf);
    } else {
        return std::string(time_buf);
    }
}

int main()
{
    int iSeed = time(NULL)%1000000;
    iSeed = 587983;
    g_rng.seed(iSeed);
    cout << "use seed " << iSeed << endl;
       
    generate_data_set (sData, 101, 1000);
    sData.randomize();
/*
    for (size_t i=0; i < 10; ++i) {
        for (size_t j=0; j < sData.samples_[i].inputs_[0][0].size(); ++j) {
            cout << " " << sData.samples_[i].inputs_[0][0][j];
        }
    }
*/
    //NeuralNetwork network(numInputPoints, hidden_layers_topology, numOutputPoints);
    ConvNeuralNetwork network;
    network.add_input_layer(1, 1, numInputPoints, INPUT_DROPOUT_RATE);
#if USE_DEEP
    network.add_fc_layer(70, DROPOUT_RATE);
    network.add_fc_layer(70, DROPOUT_RATE);
    network.add_fc_layer(70, DROPOUT_RATE);
#else
    network.add_fc_layer(700, DROPOUT_RATE);
#endif
    network.add_output_layer(numOutputPoints);
    network.init_weights(0, 1.0, 1.0);
    
    const size_t numGenes = network.number_of_weights();
    cout << "(" << numGenes << ") weights\n";

    Genome entity(numGenes);
    network.save_weights_to(entity.genes_);
    
    vector<double> outputs(numOutputPoints);
        
    network.set_weights(entity.genes_);
    
    int g_generation = 0;
    
    network.set_regularization_factor(REGULARIZATION_FACTOR);
    network.set_max_norm(MAX_NORM);
    network.set_learning_rate(LEARNING_RATE);
    network.set_batch_size(BATCH_SIZE);
    
    network.save_weights_to(entity.genes_);
    for (int i=1; i < 20; ++i) {
        cout << " " << entity.genes_[numGenes - i];
    }
    cout << "\n";
    
    while (g_generation < 3000) {
//        cout << "generation " << g_generation << "    ";
        ++g_generation;
        int run_count = 0;
        double max_error = 0;
        int correct_answer = 0;
        
        for (size_t i=0; i < sData.samples_.size(); ++i) {
            network.train(sData.samples_[i].inputs_, outputs);
            if (answer_correct(outputs, sData.samples_[i].class_)) {
                ++correct_answer;
            }
            network.backward_pass(outputs, sData.samples_[i].class_);
            max_error = network.error() > max_error ? network.error() : max_error;
            ++run_count;
            if (run_count >= BATCH_SIZE) {
                network.update_weights();
                run_count = 0;
            }            
        }
        if (run_count != 0) {
            network.update_weights();
            run_count = 0;
        }            
                
        if (g_generation % 10 == 0) {
            cout <<  timestamp_repr() << " generation " << g_generation << " ";
            cout << "train-set precision: " << (100.0 * double(correct_answer) / sData.samples_.size()) << "%" << endl;
            int correct = test_check(network);
            if (!network.regularization_on () && double(correct_answer)/sData.samples_.size() - double(correct)/sDataTest.samples_.size() > 0.05 ) {
               // network.set_regularization_on (true);
               // cout << "\t\t**** regularization on ****\n" << endl;
            }
            
            if (correct == sDataTest.samples_.size()) {
                cout << "done!\n";
                break;
            }
            
            sData.randomize();
        }
        
    }
    
    test_check(network, true);
    
    network.save_weights_to(entity.genes_);
    cout << "(" << entity.genes_.size() << ") weights\n";
    double weights = 0;
    for (size_t i=0; i < entity.genes_.size(); ++i) {
        weights += fabs(entity.genes_[i]);
    }
    cout << "|totalWeights| = " << weights << "\n";
    entity.report();
    cout << "\n";

    return 0;
}
