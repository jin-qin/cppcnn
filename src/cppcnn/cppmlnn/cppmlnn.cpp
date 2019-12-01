#include "cppmlnn.h"

#include <iostream>
#include <random>
#include "util.h"

using namespace std;
using namespace Eigen;

CppMLNN::CppMLNN() {
}

void CppMLNN::run_train(const MatrixXd &train_examples, const VectorXd &train_labels) {
	cout << __FUNCTION__ << endl;

	auto input_data = preprocess_data(train_examples, train_labels);
	const auto& all_examples = input_data.first;
	const auto& all_labels = input_data.second;
	auto examples = all_examples;
	auto labels = all_labels;

	m_examples = all_examples.cols();
	m_input_size = all_examples.rows();

	//check_gradient(examples, labels);
    
	cout << __FUNCTION__ << ":: " << "start training..." << endl;

    // loop to minimize loss function.
	random_device rd;
    while ((m_epoch_max > 0 && m_epoch < m_epoch_max) && m_loss >= m_minimum_loss) {
		vector<MatrixXd> gradients;

		if (m_sgd_enabled) {
			// perform mini-batch.
			auto rand_sample = rd() % (all_examples.cols() - m_mini_batch_size);
			examples = all_examples.block(0, rand_sample, all_examples.rows(), m_mini_batch_size);
			labels = all_labels.block(0, rand_sample, all_labels.rows(), m_mini_batch_size);
		}

		vector<MatrixXd> Z, A;
		forward_propagation(examples, Z, A);
		auto y_p = A.back();
		back_propagation(Z, A, labels, gradients);

		m_epoch_batch++;
		if ((m_epoch_batch * m_mini_batch_size) % m_examples == 0) {
			m_epoch++;

			m_loss = loss(y_p, labels);

			if (m_learn_rate_decay_enabled) decay_learn_rate();

			cout << __FUNCTION__ << ":: " << "epoch: " << m_epoch << "/" << m_epoch_max
				<< ", loss: " << m_loss << "; ";
			cout << "learning rate: " << m_learn_rate << endl;
		}

		cout << __FUNCTION__ << ":: " << "epoch: " << m_epoch << "/" << m_epoch_max
			<< ", loss: " << m_loss << endl;
    }

    cout << endl << __FUNCTION__ << ":: " << "finish training model!" << endl;
}

void CppMLNN::run_test(const MatrixXd &test_examples,  const VectorXd &test_labels) {
	auto input_data = preprocess_data(test_examples, test_labels);
	vector<MatrixXd> Z, A;
	forward_propagation(input_data.first, Z, A);
	auto y_predict = Z.back();
	auto probs = softmax(y_predict);
	
	size_t num_correct_predicted = ((test_labels - util::argmax(probs, 1)).array() == 0.0).count();

	double correctness = num_correct_predicted / (test_labels.rows() * 1.0);

	cout << num_correct_predicted << "/" << test_labels.rows() << "=" << correctness << endl;
}

void CppMLNN::forward_propagation(const MatrixXd &examples, vector<MatrixXd> &Z, vector<MatrixXd> &A, vector<Theta> model) {
	if (m_model.size() <= 0) init_model(examples.rows());

	MatrixXd pa = examples; // previous layer activation vectors
	Z.push_back(pa);
	A.push_back(pa);

    auto f_relu = [] (double value) { return value <= 0 ? 0.0 : value; }; // ReLU
	auto f_sigmoid = [](double value) { return 1.0 / (1.0 + exp(-1.0 * value)); }; // Sigmoid

	auto& _model = (model.size() > 0) ? model : m_model;
    for (auto theta : _model) {
		auto& W = theta.first;
		auto& b = theta.second;
		// compute z vectors of next layer.
		pa = (W * pa).colwise() + b; // pa is still z values here.
        Z.push_back(pa);

		// apply activation function on all the values.
		pa = pa.unaryExpr(ref(f_relu));
        A.push_back(pa);
    }
}

MatrixXd CppMLNN::back_propagation(const vector<MatrixXd> &Z, const vector<MatrixXd> &A, const MatrixXd& train_labels, vector<MatrixXd> &gradients) {
	auto df_relu = [] (double value) { return value <= 0 ? 0.0 : 1; }; // Derivative of ReLU

	auto f_sigmoid = [](double value) { return 1.0 / (1.0 + exp(-1.0 * value)); }; // Sigmoid
	auto df_sigmoid = [&f_sigmoid](double value) { return f_sigmoid(value) * (1.0 - f_sigmoid(value)); };
    
    auto m = train_labels.cols(); // m is the number of examples.

	const MatrixXd& y_predict = A.back();
	auto probs = softmax(y_predict);

    // computer the delta matrix of the output layer.
	MatrixXd delta_prev = (probs - train_labels); // n-by-m

    // m_model[m_model.size() - 1] is weight of layer <n_l - 1>
    // Z[m_model.size() - 1] is z values of layer <n_l - 1>
    for (int64_t i = static_cast<int64_t>(m_model.size()) - 1; i >= 0; i--) {
        // compute gradient of w and b
		MatrixXd grad_w = (delta_prev * A[i].transpose()) / (m * 1.0);
        MatrixXd grad_b = delta_prev.rowwise().sum() / (m * 1.0);

		MatrixXd gradient(grad_w.rows(), grad_w.cols() + grad_b.cols());
		gradient << grad_w , grad_b;
		gradients.push_back(gradient);

		auto& W = m_model[i].first;
		auto& B = m_model[i].second;

		// update previous layer's delta
		auto dz = Z[i];
		dz = dz.unaryExpr(ref(df_relu));
		delta_prev = (W.transpose() * delta_prev).cwiseProduct(dz);

        // update the model parameters
		if (m_momentum_enabled && !m_adam_enabled) {
			// apply momentum
			auto& vW = m_velocities[i].first;
			auto& vB = m_velocities[i].second;
			vW = m_beta1 * vW + (1 - m_beta1) * grad_w;
			vB = m_beta1 * vB + (1 - m_beta1) * grad_b;

			W -= m_learn_rate * (vW + W * m_lambda); // update wieght
			B -= m_learn_rate * vB; // update bias
		} else if (m_adam_enabled) {
			// apply momentum
			auto& vW = m_velocities[i].first;
			auto& vB = m_velocities[i].second;
			vW = m_beta1 * vW + (1 - m_beta1) * grad_w;
			vB = m_beta1 * vB + (1 - m_beta1) * grad_b;

			// apply rmsprop
			auto& sW = m_rmsprop[i].first;
			auto& sB = m_rmsprop[i].second;
			sW = (m_beta2 * sW).array() + (1 - m_beta2) * grad_w.array().square();
			sB = (m_beta2 * sB).array() + (1 - m_beta2) * grad_b.array().square();

			// apply adam
			double denomi_v = 1 - pow(m_beta1, m_epoch + 1);
			double demoni_s = 1 - pow(m_beta2, m_epoch + 1);
			MatrixXd vcW = vW.array() / denomi_v;
			VectorXd vcB = vB.array() / denomi_v;
			MatrixXd scW = sW.array() / demoni_s;
			VectorXd scB = sB.array() / demoni_s;

			W = W.array() - m_learn_rate * (vcW.array() / (scW.array().sqrt() + m_epsilon_rmsprop) + W.array() * m_lambda); // update wieght
			B = B.array() - m_learn_rate * (vcB.array() / (scB.array().sqrt() + m_epsilon_rmsprop)); // update bias
		} else {
			W -= m_learn_rate * (grad_w + W * m_lambda); // update wieght
			B -= m_learn_rate * grad_b; // update bias
		}
    }

	return delta_prev;
}

double CppMLNN::loss(const MatrixXd &y_predict, const MatrixXd &labels) {
	MatrixXd probs = softmax(y_predict);
	MatrixXd entropy = cross_entropy(probs, labels);

	return entropy.mean();
}

MatrixXd CppMLNN::softmax(MatrixXd z) {
	// minus max value of each output vector to avoid NaN
	// this is actually making softmax values to normalize between [0, 1]
	// and will not affect the results of softmax.
	VectorXd v_max = z.colwise().maxCoeff();
	z = z.rowwise() - v_max.transpose();

	MatrixXd exp_z = z.array().exp();
	VectorXd sum_exp_z = exp_z.colwise().sum();
	return static_cast<MatrixXd>(exp_z.array().rowwise() / sum_exp_z.transpose().array());
}
MatrixXd CppMLNN::cross_entropy(const MatrixXd& probs, const MatrixXd& labels) {
	MatrixXd log_probs = (probs.array() + pow(10, -10)).log(); // plus 10^-10 to avoid log(0)
	return MatrixXd(-1 * log_probs.cwiseProduct(labels).colwise().sum());
}

void CppMLNN::decay_learn_rate() {
	m_learn_rate = m_learn_rate_init / (1.0 + m_learn_rate_decay * (m_epoch + 1));
}

pair<MatrixXd, MatrixXd> CppMLNN::preprocess_data(const MatrixXd& examples, const VectorXd& labels) {
	cout << __FUNCTION__ << endl;

	MatrixXd labels_mat = build_labels_matrix(labels);
	MatrixXd examples_norm = normalize_examples(examples);

	return make_pair(examples_norm, labels_mat);
}

Eigen::MatrixXd CppMLNN::normalize_examples(const MatrixXd& examples) {
	cout << __FUNCTION__ << endl;
	return examples / 255.0;
}

MatrixXd CppMLNN::build_labels_matrix(const Eigen::VectorXd& train_labels) {
	cout << __FUNCTION__ << endl;

	MatrixXd labels_mat = MatrixXd::Constant(m_output_size, train_labels.rows(), 0);
	for (int i = 0; i < train_labels.rows(); i++)
		labels_mat(static_cast<size_t>(train_labels.coeff(i)), i) = 1;

	return labels_mat;
}

void CppMLNN::init_model(size_t input_layer_size) {
	cout << __FUNCTION__ << endl;

    // construct layers sizes.
	std::vector<size_t> layers_size { input_layer_size };
    layers_size.insert(layers_size.end(), m_hidden_layers_size.begin(), m_hidden_layers_size.end());
    layers_size.push_back(m_output_size);

    // initialize weights W and b with normal distribution values.
	random_device rd;
	normal_distribution<double> dis(0.0, 1.0);
	for (int i = 0; i < layers_size.size() - 1; i++) {
		MatrixXd W = MatrixXd::NullaryExpr(layers_size[i + 1], layers_size[i], [&]() { return dis(rd) / (1.0 * layers_size[i]); });
		//MatrixXd W = MatrixXd::NullaryExpr(layers_size[i + 1], layers_size[i], [&]() { return 0.0001; });
		VectorXd b = VectorXd::Constant(layers_size[i + 1], 0);
		m_model.push_back(make_pair(W, b));

		auto vW = MatrixXd::Constant(layers_size[i + 1], layers_size[i], 0);
		auto& vb = b;

		if (m_momentum_enabled || m_adam_enabled)
			m_velocities.push_back(make_pair(vW, vb));

		if (m_adam_enabled)
			m_rmsprop.push_back(make_pair(vW, vb));
	}
}

void CppMLNN::check_gradient(const MatrixXd& examples, const MatrixXd& labels) {
	double epsilon = pow(10, -7);
	cout << "espsilon:" << epsilon << endl;

	MatrixXd examples_batch = examples.block(0, 0, examples.rows(), 2);
	MatrixXd labels_batch = labels.block(0, 0, labels.rows(), 2);

	auto model_cp = m_model;
	auto& _model = model_cp.back();
	auto& W = _model.first;
	auto& b = _model.second;
	W.conservativeResize(W.rows(), W.cols() + 1);
	W.col(W.cols() - 1) = b;
	auto& theta = W;

	vector<MatrixXd> gradients, Z, A;
	forward_propagation(examples_batch, Z, A);
	back_propagation(Z, A, labels_batch, gradients);
	auto grad_bp = gradients.front();

	MatrixXd grad_compute = MatrixXd::Constant(theta.rows(), theta.cols(), 0);
	for (int i = 0; i < theta.rows(); i++) {
		for (int j = 0; j < theta.cols(); j++) {
			vector<MatrixXd> Z, A;
			theta(i, j) += epsilon;
			forward_propagation(examples_batch, Z, A, model_cp);
			auto J_plus = loss(Z.back(), labels_batch);

			theta(i, j) -= 2 * epsilon;
			forward_propagation(examples_batch, Z, A, model_cp);
			auto J_minus = loss(Z.back(), labels_batch);

			grad_compute(i,j) = (J_plus - J_minus) / (2 * epsilon);

			theta(i, j) += epsilon;
		}
	}

	cout << "difference: " << (grad_compute - grad_bp).norm() / max(grad_compute.norm(), grad_bp.norm());

	exit(0);
}