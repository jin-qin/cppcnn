#ifndef CPPMLNN_H
#define CPPMLNN_H

#include <vector>
#include <Eigen/Dense>

class CppMLNN {
	friend class CppCNN;

    typedef std::pair<Eigen::MatrixXd, Eigen::VectorXd> Theta;

public:
    CppMLNN();

    inline void set_hidden_layers(const std::vector<size_t> &hidden_layers) { m_hidden_layers_size = hidden_layers; }
    inline void set_learn_rate(double value) { m_learn_rate_init = value; m_learn_rate = value; }
    inline void set_learn_rate_decay(double value) { m_learn_rate_decay = value; }
    inline void set_momentum(double value) { m_beta1 = value; }
    inline void set_enable_sgd(bool enable) { m_sgd_enabled = enable; }
    inline void set_enable_momentum(bool enable) { m_momentum_enabled = enable; }
    inline void set_enable_adam(bool enable) { m_adam_enabled = enable; }
    inline void set_enable_learn_rate_decay(bool enable) { m_learn_rate_decay_enabled = enable; }
    inline void set_rmsprop(double value) { m_beta2 = value; }
    inline void set_max_epoch(int64_t value) { m_epoch_max = value; }
	inline void set_min_loss(double value) { m_minimum_loss = value; }
	inline void set_minibatch_size(size_t sz) { m_mini_batch_size = sz; }
    inline void set_output_size(size_t sz) { m_output_size = sz; }

	/**
	 * @brief do forward propagation
	 * @param Z, Z is the vector that stores all the layer values except the input layer.
	 * which has not been applied with activation function yet.
	 * @param A, A is the vector that stores all the layer activation values except the input layer.
	 * @param model, if you specify a model, then use this model instead of the model posessed by the object.
	 */
	void forward_propagation(const Eigen::MatrixXd& examples,
		std::vector<Eigen::MatrixXd>& Z,
		std::vector<Eigen::MatrixXd>& A,
		std::vector<Theta> model = std::vector<Theta>());

	Eigen::MatrixXd back_propagation(const std::vector<Eigen::MatrixXd> &Z, const std::vector<Eigen::MatrixXd> &A,
									 const Eigen::MatrixXd& train_labels,
									 std::vector < Eigen::MatrixXd >& gradients);

    /**
     * @brief run train procedure.
     * @param train_examples, should be m-by-n matrix, m is the number of the examples
     * @param train_labels, should be 1-by-m vector, m is the number of the examples
     */ 
    void run_train(const Eigen::MatrixXd &train_examples, const Eigen::VectorXd &train_labels);
    void run_test(const Eigen::MatrixXd &test_examples,  const Eigen::VectorXd &test_labels);

private:
    inline size_t hidden_layers_num() const { return m_hidden_layers_size.size(); }

	std::pair<Eigen::MatrixXd, Eigen::MatrixXd> preprocess_data(const Eigen::MatrixXd& examples, const Eigen::VectorXd& labels);
	Eigen::MatrixXd normalize_examples(const Eigen::MatrixXd& examples);
	Eigen::MatrixXd build_labels_matrix(const Eigen::VectorXd& train_labels);

    void init_model(size_t input_layer_size);

	double loss(const Eigen::MatrixXd& y_predict, const Eigen::MatrixXd& labels);
	Eigen::MatrixXd softmax(Eigen::MatrixXd z);
	Eigen::MatrixXd cross_entropy(const Eigen::MatrixXd& probs, const Eigen::MatrixXd& labels);

    void decay_learn_rate();

    void check_gradient(const Eigen::MatrixXd& examples, const Eigen::MatrixXd& labels);

private:
    size_t m_input_size = 0;
	size_t m_output_size = 0;

    std::vector<size_t> m_hidden_layers_size;

	size_t m_examples = 0;

    // hyperparameters
    double m_learn_rate_init = 0.0003;
    double m_learn_rate = 0.0003;
    double m_learn_rate_decay = 0.95;
    double m_beta1 = 0.9; // for momentum
    double m_beta2 = 0.999; // for RMSProp
    double m_epsilon_rmsprop = std::pow(10, -8); // for RMSProp
    double m_lambda = 0.00; // coefficient of the regularization term.

    bool m_sgd_enabled = true;
    bool m_momentum_enabled = true;
    bool m_adam_enabled = false;
    bool m_learn_rate_decay_enabled = false;

    int64_t m_epoch_batch = 0;
    int64_t m_epoch = 0;
    int64_t m_epoch_max = -1; // -1 means infinite
	double m_loss = INFINITY;
	double m_minimum_loss = 0.0;
	size_t m_mini_batch_size = 256;

    // pair.first is W, pair.second is b.
    std::vector< Theta > m_model; // Theta of each layer and the layer after.
    std::vector< Theta > m_velocities; // velocities for each layers parameters
    std::vector< Theta > m_rmsprop;
};

#endif // CPPMLNN_H