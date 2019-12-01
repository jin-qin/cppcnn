#include <iostream>
#include <Eigen/Dense>
#include "util.h"
#include "cppcnn.h"

using Eigen::MatrixXd;

int main() {
    using namespace std;

	cout.sync_with_stdio(false);
	cout << "Threads:" << Eigen::nbThreads() << endl;

	/* inputs matrix: [n * m], m is the pixels number of one image, n is the number of examples.
	 * labels matrix: [m * 1], m is the number of labels.
	 */
    auto train_inputs = util::read_mnist("../../../data/mnist/train-images.idx3-ubyte", 
										 "../../../data/mnist/train-labels.idx1-ubyte");
	auto test_inputs = util::read_mnist("../../../data/mnist/t10k-images.idx3-ubyte",
										"../../../data/mnist/t10k-labels.idx1-ubyte");

	auto cnn = make_shared<CppCNN>();
	cnn->set_enable_momentum(false);
	cnn->set_enable_adam(false);
	cnn->set_learn_rate_decay(false);
	cnn->set_output_size(10);
	cnn->set_learn_rate(0.005);
	cnn->set_max_epoch(3);
	cnn->set_minibatch_size(1);

	cnn->add_conv_layer(3, 8);
	cnn->add_max_pool_layer(2, 2);
	cnn->set_hidden_layers({32, 16});

	cnn->train(train_inputs.first, train_inputs.second);

	cnn->test(test_inputs.first, test_inputs.second);

    return 0;
}