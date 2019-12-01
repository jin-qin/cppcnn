#ifndef BASIC_CNN_H
#define BASIC_CNN_H

#include <vector>
#include <memory>
#include "cnn_defines.h"
#include "cppmlnn/cppmlnn.h"

class LayerBase;

class CppCNN : public std::enable_shared_from_this<CppCNN>
{
public:
    CppCNN();
    ~CppCNN();

	inline void set_output_size(size_t value) { __output_size = value; __fcnn.set_output_size(value); }
	inline void set_hidden_layers(const std::vector<size_t>& hidden_layers) { 
		__hidden_layers_size = hidden_layers; __fcnn.set_hidden_layers(hidden_layers);
	}
	inline void set_learn_rate(double value) { 
		__learn_rate_init = value; __learn_rate = value; 
		__fcnn.set_learn_rate(value);
	}
	inline void set_learn_rate_decay(double value) { __learn_rate_decay = value; __fcnn.set_learn_rate_decay(value); }
	inline void set_momentum(double value) { __beta1 = value; __fcnn.set_momentum(value); }
	inline void set_enable_sgd(bool enable) { __sgd_enabled = enable; __fcnn.set_enable_sgd(enable); }
	inline void set_enable_momentum(bool enable) { __momentum_enabled = enable; __fcnn.set_enable_momentum(enable); }
	inline void set_enable_adam(bool enable) { __adam_enabled = enable; __fcnn.set_enable_adam(enable); }
	inline void set_enable_learn_rate_decay(bool enable) { __learn_rate_decay_enabled = enable; __fcnn.set_enable_learn_rate_decay(enable); }
	inline void set_rmsprop(double value) { __beta2 = value; __fcnn.set_rmsprop(value); }
	inline void set_max_epoch(int64_t value) { __epoch_max = value; }
	inline void set_min_loss(double value) { __minimum_loss = value; }
	inline void set_minibatch_size(size_t sz) { __mini_batch_size = sz; }

    /**
     * @brief add a convolution layer.
     * @param f, filter size
     * @param p, padding size
     * @param s, stride, should greater than 0
     * @param nf, number of filters
     */
    void add_conv_layer(int f, int nf, int p = 0, int s = 1);

    /**
     * @brief add a max pooling layer.
     */
    void add_max_pool_layer(int f, int s = 1);

	/**
	 * @brief run train procedure.
	 * @param examples, should be m-by-n matrix, m is the number of the examples
	 * @param labels, should be 1-by-m vector, m is the number of the examples
	 */
	void train(const ImageVolume& examples, const Eigen::VectorXd& labels);
	void test(const ImageVolume& examples, const Eigen::VectorXd& labels);

private:
	std::pair< AllLayersValuesBatch, AllLayersValuesBatch > forward_conv(const ImageVolume& inputs);
	std::pair< LayerValues, LayerValues > forward_fc(const AllLayersValuesBatch& input_batch);

	Eigen::MatrixXd backprop_fc(const LayerValues& Z, const LayerValues& A, 
								const Eigen::MatrixXd& train_labels, LayerValues& grads_fc);
	void backprop_conv(const Eigen::MatrixXd& delta_prev_fc, 
					   const AllLayersValuesBatch& Z_batch, const AllLayersValuesBatch& A_batch);

	Eigen::VectorXd flatten_layer(const LayerValues& input);
	LayerValues unflatten_layer(const Eigen::VectorXd& input, const LayerSize& layer_sz);

	std::pair<ImageVolume, Eigen::MatrixXd> preprocess_data(const ImageVolume& examples, const Eigen::VectorXd& labels);
	ImageVolume normalize_examples(const ImageVolume& examples);
	Eigen::MatrixXd build_labels_matrix(const Eigen::VectorXd& train_labels);

	bool is_conv_layer(const LayerBase& layer);
	bool is_pool_layer(const LayerBase& layer);
	bool is_fc_layer(const LayerBase& layer);

	void print_info(bool verbose = false);

private:
	friend class LayerConv;
	friend class LayerPool;

	std::vector< std::shared_ptr<LayerBase> > __layers;

	CppMLNN __fcnn; // Fully connected neural network.

	size_t __input_size = 0;
	size_t __output_size = 0;

	std::vector<size_t> __hidden_layers_size;

	size_t __examples_num = 0;

	// hyperparameters
	double __learn_rate_init = 0.0003;
	double __learn_rate = 0.0003;
	double __learn_rate_decay = 0.95;
	double __beta1 = 0.9; // for momentum
	double __beta2 = 0.999; // for RMSProp
	double __epsilon_rmsprop = std::pow(10, -8); // for RMSProp
	double __lambda = 0.00; // coefficient of the regularization term.

	bool __sgd_enabled = true;
	bool __momentum_enabled = true;
	bool __adam_enabled = false;
	bool __learn_rate_decay_enabled = false;

	int64_t __epoch_batch = 0;
	int64_t __epoch = 0;
	int64_t __epoch_max = -1; // -1 means infinite
	double __loss = INFINITY;
	double __minimum_loss = 0.0;
	size_t __mini_batch_size = 256;
};

#endif // BASIC_CNN_H