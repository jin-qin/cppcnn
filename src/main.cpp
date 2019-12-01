#include <iostream>
#include <Eigen/Dense>
#include "util.h"

using Eigen::MatrixXd;

int main() {
    using namespace std;

	cout.sync_with_stdio(false);

	/* inputs matrix: [n * m], m is the pixels number of one image, n is the number of examples.
	 * labels matrix: [m * 1], m is the number of labels.
	 */
    // auto train_inputs = util::read_mnist("../data/mnist/train-images.idx3-ubyte", true);
    // auto train_labels = util::read_mnist("../data/mnist/train-labels.idx1-ubyte", false);
	// auto test_inputs = util::read_mnist("../data/mnist/t10k-images.idx3-ubyte", true);
	// auto test_labels = util::read_mnist("../data/mnist/t10k-labels.idx1-ubyte", false);

    // CppMLNN bpnn;
    // bpnn.set_hidden_layers(vector<size_t> ({256}));
    // bpnn.set_output_size(10);
    // bpnn.set_learn_rate(0.5);
    // bpnn.set_max_epoch(2000);
	// bpnn.set_minibatch_size(500);
    // bpnn.run_train(train_inputs, train_labels, true);
	// bpnn.run_test(test_inputs, test_labels);
    return 0;
}