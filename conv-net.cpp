/* 
 * This sample use gradient descend to classify a spiral dataset
 * http://cs231n.github.io/neural-networks-case-study/
 */
#include "conv-neural-network.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <sys/time.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cfloat>

using namespace std;

int g_generation = 0;
ConvNeuralNetwork *g_network = 0;
const int ImageWidth = 32;
const int ImageHeight = 32;
const int nClass = 10;

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

struct ImgData {
    uint8_t class_;
    uint8_t r_[32*32];
    uint8_t g_[32*32];
    uint8_t b_[32*32];
};
struct InputData {
    uint8_t class_;
    vector<vector<vector<double> > > data_;
};

struct FileGuard
{
    FILE *f_;
    FileGuard(FILE *f)
    : f_(f)
    {
    }
    ~FileGuard()
    {
        fclose(f_);
    }
};

vector<InputData> g_images;
vector<InputData> test_images;
vector<string> g_classLabels;

void load_cifar_images (string const&file_path, vector<InputData> & images)
{
    FILE *f = fopen(file_path.c_str(), "rb");
    FileGuard g(f);
    
    fseek(f, 0, SEEK_END);
    uint64_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    int original_size = images.size();
    
    vector<ImgData> imgs;
    imgs.resize(file_size / sizeof(ImgData));
    size_t nItem = fread(&imgs[0], 1, file_size, f);
    
    for (size_t in=0; in < imgs.size(); ++in) {
        const ImgData &img = imgs[in];
        InputData ip;
        ip.class_ = img.class_;
        ip.data_.resize(ImageWidth);
        for (int px=0; px < ImageWidth; ++px) {
            ip.data_[px].resize(ImageHeight);
            for (int py=0; py < ImageHeight; ++py) {
                int pi = py * ImageWidth + px;
                ip.data_[px][py].push_back(img.r_[pi]/255.0 - 0.5);
                ip.data_[px][py].push_back(img.g_[pi]/255.0 - 0.5);
                ip.data_[px][py].push_back(img.b_[pi]/255.0 - 0.5);
            }
        }
        images.push_back(ip);
    }
}

#define RENDER_WINDOW_WIDTH 600
#define RENDER_WINDOW_HEIGHT 600

struct Color {
    Uint8 r_, g_, b_;
    Color(Uint8 r,Uint8 g, Uint8 b)
    : r_(r), g_(g), b_(b)
    {
    }
};

class Renderer {
    SDL_Window *win_;
    SDL_Renderer *ren_;
    SDL_Texture *tex_;
    
    int init_sdl ()
    {
        if (SDL_Init(SDL_INIT_VIDEO) != 0){
            std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;
            return 1;
        }
        
        win_ = SDL_CreateWindow("Convolute the World!", 100, 100, RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
        if (win_ == 0){
            std::cout << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return 1;
        }
            
        ren_ = SDL_CreateRenderer(win_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (ren_ == 0){
            SDL_DestroyWindow(win_);
            std::cout << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return 1;
        }    

        std::string imagePath = "circle.png";
        tex_ = IMG_LoadTexture(ren_, imagePath.c_str());
        if (tex_ == 0) {
            SDL_DestroyRenderer(ren_);
            SDL_DestroyWindow(win_);
            std::cout << "SDL_CreateTextureFromSurface Error: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return 1;
        }


        return 0;
    }
        

public:
    Renderer ()
    : win_(0), ren_(0), tex_(0)
    {
    }
    
    ~Renderer ()
    {
        if (tex_) SDL_DestroyTexture(tex_);
        if (ren_) SDL_DestroyRenderer(ren_);
        if (win_) SDL_DestroyWindow(win_);
        SDL_Quit();

    }
    
    void set_title (const string &title)
    {
        SDL_SetWindowTitle (win_, title.c_str());
    }
    
    int start ()
    {
        return init_sdl ();
    }
    
    void render_cifar_image (int nImg)
    {
        SDL_SetRenderDrawColor(ren_, 255, 255, 255, 255);
        SDL_RenderClear(ren_);
        
        // cifar-10 image of dimention 32x32
        Uint32 rmask=0, gmask=0, bmask=0, amask=0;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        rmask = 0xff000000;
        gmask = 0x00ff0000;
        bmask = 0x0000ff00;
        amask = 0x000000ff;
#else
        rmask = 0x000000ff;
        gmask = 0x0000ff00;
        bmask = 0x00ff0000;
        amask = 0xff000000;
#endif        
        vector<SDL_Texture *> textures;
        
        for (size_t in=0; in < nImg; ++in) {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, 32, 32, 32, rmask, gmask, bmask, amask);
            if (!surface) {
                cout << "out of memory to create sdl surface." << endl;
                return;
            }

            if( SDL_MUSTLOCK( surface ) )
            {
                //Lock the surface
                SDL_LockSurface( surface );
            }
            
            // load pixels
            const InputData &img = g_images[in];
            for (int py=0; py < 32; ++py) {
                for (int px=0; px < 32; ++px) {
                    //Convert the pixels to 32 bit
                    uint8_t *pixels = (uint8_t *)surface->pixels;                
                    //Set the pixel
                    int p = py * surface->pitch + px * 4;
                    int pi = py * 32 + px;
                    pixels[ p + 0 ] = (img.data_[px][py][0] + 0.5)*255;
                    pixels[ p + 1 ] = (img.data_[px][py][1] + 0.5)*255;
                    pixels[ p + 2 ] = (img.data_[px][py][2] + 0.5)*255;
                    pixels[ p + 3 ] = 255;
                }
            }
            
            cout << g_classLabels[img.class_] << "\t";
            
            if( SDL_MUSTLOCK( surface ) )
            {
                //Lock the surface
                SDL_UnlockSurface( surface );
            }
            
            SDL_Texture *tex = SDL_CreateTextureFromSurface (ren_, surface);
            
            SDL_Rect dst;
            dst.w = dst.h = 32;
            dst.x = (in % 15) * 33;
            dst.y = (in / 15) * 33;
            SDL_RenderCopy(ren_, tex, NULL, &dst);
            //textures.push_back(tex);
            SDL_DestroyTexture(tex);
        }

        SDL_RenderPresent(ren_);

        //for (size_t i=0; i < textures.size(); ++i) {
        //    SDL_DestroyTexture(textures[i]);
        //}
        
        SDL_Delay(300);
    }
};
    
bool answer_correct (vector<double> const&outputs, int right_ans)
{
    double max_term=-DBL_MAX;
    int ans = -1;
    for (size_t i=0; i < outputs.size(); ++i) {
        if (outputs[i] > max_term) {
            max_term = outputs[i];
            ans = i;
        }
    }
    return ans == right_ans;
}

int answer_of (vector<double> const&outputs)
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

double now ()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

double test_check ()
{
    cout << "testing...";
    double begin_time, end_time;
    int correct_answer;
    vector<double> outputs;
    vector<pair<int,int> > scores;
    scores.resize(g_classLabels.size());
    
    begin_time = now ();
    correct_answer = 0;
    for (size_t in=0; in < test_images.size(); ++in) {
        InputData &img = test_images[in];
        g_network->run(img.data_, outputs);
        ++scores[test_images[in].class_].first;
        if (answer_correct(outputs, test_images[in].class_)) {
            ++correct_answer;
            ++scores[test_images[in].class_].second;
        }
    }
    
    end_time = now ();
    cout << "\ntakes time " << (end_time - begin_time) << " seconds" << endl;

    for (size_t i=0; i < scores.size(); ++i) {
        cout << "\t" << g_classLabels[i] << ": " << scores[i].second << "/" << scores[i].first << "\n";
    }
    cout << "precision: " << (100.0 * double(correct_answer) / test_images.size()) << "%" << endl;
    
    Genome entity;
    g_network->save_weights_to(entity.genes_);
    double regularz = 0;
    double max_w = -DBL_MAX;
    double min_w = DBL_MAX;
    for (size_t i=0; i < entity.genes_.size(); ++i) {
        regularz += fabs(entity.genes_[i]);
        if (entity.genes_[i] > max_w) max_w = entity.genes_[i];
        if (entity.genes_[i] < min_w) min_w = entity.genes_[i];
    }
    cout << "|" << entity.genes_.size() << "| weights\n";
    cout << "|totalWeights| = " << regularz << " (max: " << max_w << " min: " << min_w << " ) ";
    cout << " @time " << timestamp_repr() << "\n" << endl;
    
    return (100.0 * double(correct_answer) / test_images.size());
}

class SDLSurfaceGuard
{
    SDL_Surface *surface_;
public:
    SDLSurfaceGuard(SDL_Surface *f)
    : surface_(f)
    {}
    ~SDLSurfaceGuard()
    {
        SDL_FreeSurface(surface_);
    }
};

void checkout_image_class (string const&file_path)
{
    SDL_Surface *s = IMG_Load(file_path.c_str());
    if (!s) {
        cout << "Load image '" << file_path << "' failed: " << SDL_GetError() << endl;
        return;
    }
    SDLSurfaceGuard guard(s);
    
    SDL_Surface *surface = SDL_CreateRGBSurface(0, 32, 32, s->format->BitsPerPixel, s->format->Rmask, s->format->Gmask, s->format->Bmask, s->format->Amask);
    if (!surface) {
        cout << "can't create a 32x32 surface:" << SDL_GetError() << endl;
        return;
    }
    SDLSurfaceGuard g2(surface);
    
    SDL_BlitScaled (s, NULL, surface, NULL);
    
    vector<double> outputs;
    
    double begin_time = now ();
    
    if( SDL_MUSTLOCK( surface ) )
    {
        //Lock the surface
        SDL_LockSurface( surface );
    }
    
    InputData ip;
    
    ip.data_.resize(surface->w);
    uint8_t *pixels = (uint8_t *)surface->pixels;                

    for (int px=0; px < surface->w; ++px) {
        ip.data_[px].resize(surface->h);
        for (int py=0; py < surface->h; ++py) {
            int pi = py * surface->pitch + px * surface->format->BytesPerPixel;
            ip.data_[px][py].push_back(pixels[pi++]/255.0 - 0.5);
            ip.data_[px][py].push_back(pixels[pi++]/255.0 - 0.5);
            ip.data_[px][py].push_back(pixels[pi++]/255.0 - 0.5);
        }
    }
    
    if( SDL_MUSTLOCK( surface ) )
    {
        //Lock the surface
        SDL_UnlockSurface( surface );
    }
    
    g_network->run(ip.data_, outputs);
    int answer = answer_of (outputs);    
    double end_time = now ();
    cout << "it is a " << g_classLabels[answer] << ", right?\n";
    cout << "\ntakes time " << (end_time - begin_time) << " seconds" << endl;
    
}

bool isPrime(int number)
{
    if (number < 2) return false;
    if (number % 2 == 0) return (number == 2);
    int root = (int)sqrt((double)number);
    for (int i = 3; i <= root; i += 2)
    {
        if (number % i == 0) return false;
    }
    return true;
}
    
int getNextPrime(int n)
{
    int i = n;
    for (; i < 2 * n; ++i)
    {
        if (isPrime(i)) return i;
    }
}

class FakeRandom
{
public:
    FakeRandom(int numElements);    
    int next (); // return -1 when all number traversed
    void restart (bool reskip=false);
    
private:
    int numElements_;
    int numSkip_;
    int numBase_; // the smallest prime bigger than numMax_
    int numCurrent_;
    int numUsed_;
};

FakeRandom::FakeRandom(int numMax)
: numElements_(numMax)
, numCurrent_(0)
, numUsed_(0)
{
    assert (numElements_ > 0);
    numBase_ = getNextPrime(numMax);
    restart(true);
}

int FakeRandom::next()
{
    if (numUsed_ == numElements_) return -1;
    
    int num = numCurrent_;
    while (true) {
        num += numSkip_;
        num %= numBase_;
        if (num < numElements_) {
            numCurrent_ = num;
            ++numUsed_;
            break;
        }
    }
    
    return numCurrent_;
}

void FakeRandom::restart(bool reskip)
{
    numUsed_ = 0;
    if (reskip) {
        numSkip_ = g_rng.randInt() * numElements_ + g_rng.randInt(); // some big number
        if (numSkip_ % numBase_ == 0) ++numSkip_;
        numSkip_ &= ~0xff000000;
    }
}

void show_usage (const char **argv)
{
    cout << "usage: " << argv[0] << " -f network_file  -c config_file -l learning_rate -d dropout_rate -i input_dropout_rate -r regularization_factor -b batch_size \
    -s save_every_x_gen -test -check test_image_file_path " << endl;
}

int main(int argc, const char **argv)
{
    const string cifar_test_img_file_path =  "cifar-10-batches-bin/test_batch.bin";

    Renderer renderer;
    
    if (renderer.start() != 0) {
        cout << "SDL2 initialization failed." << endl;
        return 1;
    }
    cout << "ImgData size = " << sizeof(ImgData) << endl;
    
    g_classLabels.push_back("airplane");
    g_classLabels.push_back("automobile");
    g_classLabels.push_back("bird");
    g_classLabels.push_back("cat");
    g_classLabels.push_back("deer");
    g_classLabels.push_back("dog");
    g_classLabels.push_back("frog");
    g_classLabels.push_back("horse");
    g_classLabels.push_back("ship");
    g_classLabels.push_back("truck");
    
    const int numOutputPoints = 10;

    int iSeed = time(NULL)%1000000;
    //iSeed = 7;
    g_rng.seed(iSeed);
    cout << "use seed " << iSeed << endl;
    
    string network_file;
    string config_file;
    int save_period = 1;
    int check_period = 1;
    string input_sequence;
    bool training = true;
    double learningRateParam = -1;
    int batchSizeParam = -1;
    double regularzParam = -1;
    string testImageFilePath;
    
    for (int i=1; i < argc; ++i) {
        if (argv[i] == string("-f")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            network_file = argv[i];
        } else if (argv[i] == string("-c")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            config_file = argv[i];
        } else if (argv[i] == string("-s")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            save_period = atoi(argv[i]);
        } else if (argv[i] == string("-l")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            learningRateParam = atof(argv[i]);
        } else if (argv[i] == string("-r")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            regularzParam = atof(argv[i]);
        } else if (argv[i] == string("-b")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            batchSizeParam = atoi(argv[i]);
        } else if (argv[i] == string("-test")) {
            training = false;
        } else if (argv[i] == string("-check")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            training = false;
            testImageFilePath = argv[i];
        } else if (argv[i] == string("-k")) {
            if (++i >= argc) {
                show_usage(argv);
                return 1;
            }
            check_period = atoi(argv[i]);
        } else {
            cout << "error parsing params\n";
            show_usage(argv);
            return 2;
        }
    }
    
    renderer.set_title ("Convolute    -    " + network_file);
    
    if (!network_file.empty()) {
        g_network = ConvNeuralNetwork::load_network_file (network_file);
    } 
    
    if (!g_network && !config_file.empty()) {
        g_network = ConvNeuralNetwork::construct_cnn_with_config_file (config_file);
    }
    
    if (training) {
        g_images.reserve(50000);
        test_images.reserve(10000);
        load_cifar_images ("cifar-10-batches-bin/data_batch_1.bin", g_images);
        load_cifar_images ("cifar-10-batches-bin/data_batch_2.bin", g_images);
        load_cifar_images ("cifar-10-batches-bin/data_batch_3.bin", g_images);
        load_cifar_images ("cifar-10-batches-bin/data_batch_4.bin", g_images);
        load_cifar_images ("cifar-10-batches-bin/data_batch_5.bin", g_images);
        renderer.render_cifar_image(30);
    }
    
    if (training || testImageFilePath.empty()) {
        load_cifar_images(cifar_test_img_file_path, test_images);
        //load_cifar_images ("cifar-10-batches-bin/data_batch_1.bin", test_images);
        //load_cifar_images ("cifar-10-batches-bin/data_batch_2.bin", test_images);
        //load_cifar_images ("cifar-10-batches-bin/data_batch_3.bin", test_images);
        //load_cifar_images ("cifar-10-batches-bin/data_batch_4.bin", test_images);
        //load_cifar_images ("cifar-10-batches-bin/data_batch_5.bin", test_images);
    }
    
    if (!g_network) {
        // config file
        /*
            Input 32 32 3
            Conv 32 3 1 1
            Pool 2 2
            Conv 64 3 1 1
            Pool 2 2 
            Full 128
            Full 64
            Output 10
            LearningRate 0.0001
            RegularizationFactor 0.0001
            BatchSize 100
            DropoutRate 0.5
        */
        g_network = new ConvNeuralNetwork;
        int imgWidth=32, imgHeight=32, nImgChannel=3;
        g_network->add_input_layer(imgWidth, imgHeight, nImgChannel);
        g_network->add_conv_layer(32, 3, 1, 1);
        g_network->add_pool_layer(2, 2);
        g_network->add_conv_layer(48, 3, 1, 1);
        g_network->add_pool_layer(2, 2);
        g_network->add_fc_layer(48);
        g_network->add_fc_layer(30);
        g_network->add_output_layer(numOutputPoints);
        g_network->init_weights(0, 1.0, 1.0);
    }
    
    const size_t numGenes = g_network->number_of_weights();

    Genome entity(numGenes);
    g_network->save_weights_to(entity.genes_);
    
    vector<double> outputs(numOutputPoints);
        
    //g_network->set_weights(entity.genes_);
    
    if (regularzParam >= 0) g_network->set_regularization_factor (regularzParam);
    if (learningRateParam > 0) g_network->set_learning_rate (learningRateParam);
    if (batchSizeParam > 0) g_network->set_batch_size (batchSizeParam);
    
    cout << "config:\n";
    g_network->report_config ();
    
    const int batch_size = g_network->batch_size();
    const int TrainNumber = g_images.size();
    double begin_time = now();
 
    int correct_answer = 0;
    if (training) {
        cout << "start training...\n";
        FakeRandom fr_rng(g_images.size());
        
        while (g_generation < 10000) {
            cout << "generation " << g_generation << "    ";
            ++g_generation;
            int run_count = 0;
            double max_error = 0;
            correct_answer = 0;
            fr_rng.restart(g_generation % 10 == 0);
            
            size_t numImages = g_images.size();
            
            for (size_t in=0; in < TrainNumber; ++in) {
                size_t chooseImg = fr_rng.next();
                InputData &img = g_images[chooseImg];
                g_network->train(img.data_, outputs);
                if (answer_correct(outputs, img.class_)) {
                    ++correct_answer;
                }

                g_network->backward_pass(outputs, img.class_);
                max_error = g_network->error() > max_error ? g_network->error() : max_error;
                ++run_count;
                if (run_count >= batch_size) {
                    ++ConvNeuralNetwork::run_generation;
                    g_network->update_weights();
                    run_count = 0;
                }
                
            }
            if (run_count != 0) {
                g_network->update_weights();
                run_count = 0;
            }            
            double correct_answer_percent = (100.0 * double(correct_answer) / TrainNumber);
            cout << "\tmax network error: " << max_error; 
            cout << "\tprecision: " << correct_answer_percent << "%";
            cout << " @time " << timestamp_repr() << endl;

            if (g_generation % save_period == 0 && !network_file.empty()) {
                ConvNeuralNetwork::save_network_file(g_network, network_file);
            }
            
            if (g_generation % check_period == 0) {
                double test_correct_percent = test_check();
                
                //if (!g_network->regularization_on() && (correct_answer_percent - test_correct_percent) > 5.0) {
                //    g_network->set_regularization_on(true);
                //}
            }
            
            if (correct_answer_percent > 95) {
                break;
            }
        }
        double end_time = now ();
        cout << "\ntakes time " << (end_time - begin_time) << " seconds" << endl;
    }
    
    if (testImageFilePath.empty()) {
        test_check ();
    } else {
        checkout_image_class (testImageFilePath);
    }
 
    return 0;
}
