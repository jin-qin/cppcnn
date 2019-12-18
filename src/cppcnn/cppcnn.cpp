#include "cppcnn.h"
#include <iostream>
#include <random>
#include "util.h"
#include "layer_conv.h"
#include "layer_pool.h"

using namespace std;
using namespace Eigen;

CppCNN::CppCNN() {
}

CppCNN::~CppCNN() {
}

void CppCNN::add_conv_layer(int f, int nf, int p, int s) {
	auto p_conv = make_shared<LayerConv>(shared_from_this(), f, nf, p, s);
	__layers.push_back(p_conv);
}

void CppCNN::add_max_pool_layer(int f, int s) {
	auto p_pool = make_shared<LayerPool>(shared_from_this(), f, LayerPool::PoolMethod::MAX, s);
	__layers.push_back(p_pool);
}

void CppCNN::train(const ImageVolume& examples, const VectorXd& labels) {
	cout << __FUNCTION__ << endl;

	auto input_data = preprocess_data(examples, labels);
	const auto& all_examples = input_data.first;
	const auto& all_labels = input_data.second;
	auto examples_batch = all_examples;
	auto labels_batch = all_labels;

	__examples_num = all_examples.size();

	cout << __FUNCTION__ << ":: " << "start training..." << endl;

	// loop to minimize loss function
	random_device rd;
	while ((__epoch_max > 0 && __epoch < __epoch_max)) {
		if (__sgd_enabled) {
			// perform mini-batch.
			auto rand_sample = rd() % (all_examples.size() - __mini_batch_size);
			auto slice_start = all_examples.begin() + rand_sample;
			examples_batch = ImageVolume(slice_start, slice_start + __mini_batch_size);
			labels_batch = all_labels.block(0, rand_sample, all_labels.rows(), __mini_batch_size);
		}
		
		// forward
		LayerValues grads_fc;
		auto conv_out = forward_conv(examples_batch);
		auto fcnn_out = forward_fc(conv_out.second);

		__loss = __fcnn.loss(fcnn_out.second.back(), labels_batch);

		// backpropagation
		auto delta_fc = backprop_fc(fcnn_out.first, fcnn_out.second, labels_batch, grads_fc);
		backprop_conv(delta_fc, conv_out.first, conv_out.second);

		__epoch_batch++;

		print_info(true);
	}

	cout << endl << __FUNCTION__ << ":: " << "finish training model!" << endl;
}

void CppCNN::test(const ImageVolume& examples, const VectorXd& labels) {
	auto input_data = preprocess_data(examples, labels);
	auto output_conv = forward_conv(input_data.first);
	auto output_fcnn = forward_fc(output_conv.second);
	auto probs = __fcnn.softmax(output_fcnn.second.back());

	size_t num_correct_predicted = ((labels - util::argmax(probs, 1)).array() == 0.0).count();

	double correctness = num_correct_predicted / (labels.rows() * 1.0);

	cout << num_correct_predicted << "/" << labels.rows() << "=" << correctness << endl;
}

pair< AllLayersValuesBatch, AllLayersValuesBatch > CppCNN::forward_conv(const ImageVolume &input) {
	AllLayersValuesBatch BatchZ;
	AllLayersValuesBatch BatchA;
	ImageVolume pa_conv = input; // previous layer's activation values, as the input for the next layer.

	BatchZ.push_back(pa_conv);
	BatchA.push_back(pa_conv);

	for (auto& p_layer : __layers) {
		if (is_conv_layer(*p_layer)) {
			auto layer_conv = dynamic_pointer_cast<LayerConv>(p_layer);
			auto output = layer_conv->convolution(pa_conv);
			pa_conv = output.second;
			BatchZ.push_back(output.first);
			BatchA.push_back(output.second);
		}
		else if (is_pool_layer(*p_layer)) {
			auto layer_pool = dynamic_pointer_cast<LayerPool>(p_layer);
			pa_conv = layer_pool->pool(BatchA.back());
			BatchZ.push_back(pa_conv);
			BatchA.push_back(pa_conv);
		}
		else {
			break;
		}
	}

	return make_pair(BatchZ, BatchA);
}

pair<LayerValues, LayerValues> CppCNN::forward_fc(const AllLayersValuesBatch& input_batch) {
	auto& last_A = input_batch.back();
	auto& one_img = last_A[0];
	auto mat_rows = one_img.size() * one_img[0].size();
	MatrixXd input_mat = MatrixXd::Constant(mat_rows, last_A.size(), 0);
	for (int i = 0; i < last_A.size(); i++) {
		auto& input = last_A[i];
		auto pa = flatten_layer(input); // previous layer activation vectors
		input_mat.col(i) = pa;
	}
	
	vector<MatrixXd> Z, A;
	__fcnn.forward_propagation(input_mat, Z, A);

	return make_pair(Z, A);
}

MatrixXd CppCNN::backprop_fc(const LayerValues& Z, const LayerValues& A, 
							   const MatrixXd& train_labels, LayerValues& grads_fc) {
	return __fcnn.back_propagation(Z, A, train_labels, grads_fc);
}

void CppCNN::backprop_conv(const MatrixXd& delta_prev_fc, 
							 const AllLayersValuesBatch& Z_batch, const AllLayersValuesBatch& A_batch) {
	auto& A = A_batch.back();
	auto& Z = Z_batch.back();

	ImageVolume DeltaPrev;
	auto& lv_fc = A[0][0]; // one of layer values before fully-connected layer.
	auto lv_fc_sz = LayerSize(lv_fc.rows(), lv_fc.cols(), A.back().size());
	for (auto i = 0; i < delta_prev_fc.cols(); i++) {
		VectorXd v_delta_prev_fc = delta_prev_fc.col(i);
		DeltaPrev.push_back(unflatten_layer(v_delta_prev_fc, lv_fc_sz));
	}
	
	for (int64_t i = __layers.size() - 1; i >= 0; i--) {
		auto& p_layer = __layers[i];
		if (is_conv_layer(*p_layer)) {
			const auto& layer_conv = dynamic_pointer_cast<LayerConv>(p_layer);

			layer_conv->compute_gradients(A_batch[i], DeltaPrev);

			// compute next layer's dZ from current layer's dZ, direction : dZ^{l-1} <-- dZ^{l}
			auto& img = Z_batch[i][0][0];
			auto next_layer_size = LayerSize(img.rows(), img.cols(), Z_batch[i][0].size());
			DeltaPrev = layer_conv->convolution_backward(DeltaPrev, Z_batch[i], next_layer_size);
		}
		else if (is_pool_layer(*p_layer)) {
			const auto& layer_pool = dynamic_pointer_cast<LayerPool>(p_layer);
			DeltaPrev = layer_pool->unpool(DeltaPrev);
		}
	}
}

Eigen::VectorXd CppCNN::flatten_layer(const LayerValues& input) {
	VectorXd v_flatten(input[0].size() * input.size());
	Index start = 0;
	for (auto& c : input) {
		MatrixXd c_value = c;
		c_value.transposeInPlace();
		c_value.resize(c.size(), 1);
		v_flatten.segment(start, c_value.size())= c_value;
		start += c_value.size();
	}
	return v_flatten;
}

LayerValues CppCNN::unflatten_layer(const Eigen::VectorXd& input, const LayerSize& layer_sz) {
	LayerValues unflatten;
	
	auto rows = layer_sz.rows;
	auto cols = layer_sz.cols;
	auto sz = rows * cols;
	auto channels = layer_sz.channels;

	Index start = 0;
	for (int i = 0; i < channels; i++) {
		MatrixXd img = input.segment(start, sz);
		img.resize(cols, rows);
		unflatten.push_back(img.transpose());
		start += sz;
	}

	return unflatten;
}

pair<ImageVolume, MatrixXd> CppCNN::preprocess_data(const ImageVolume& examples, const VectorXd& labels) {
	cout << __FUNCTION__ << endl;

	auto labels_mat = build_labels_matrix(labels);
	auto examples_norm = normalize_examples(examples);

	return make_pair(examples_norm, labels_mat);
}

ImageVolume CppCNN::normalize_examples(const ImageVolume& examples) {
	cout << __FUNCTION__ << endl;

	ImageVolume examples_norm = examples;
	for (auto& input : examples_norm)
		for (auto& channel : input)
			channel /= 255.0;

	return examples_norm;
}

MatrixXd CppCNN::build_labels_matrix(const Eigen::VectorXd& train_labels) {
	cout << __FUNCTION__ << endl;

	MatrixXd labels_mat = MatrixXd::Constant(__output_size, train_labels.rows(), 0);
	for (int i = 0; i < train_labels.rows(); i++)
		labels_mat(static_cast<size_t>(train_labels.coeff(i)), i) = 1;

	return labels_mat;
}

bool CppCNN::is_conv_layer(const LayerBase& layer) {
	return layer.type() == LayerBase::LayerType::CONV;
}

bool CppCNN::is_pool_layer(const LayerBase& layer) {
	return layer.type() == LayerBase::LayerType::POOL;
}

bool CppCNN::is_fc_layer(const LayerBase& layer)
{
	return layer.type() == LayerBase::LayerType::FC;
}

void CppCNN::print_info(bool verbose) {
	if (__epoch_batch % 100 == 0) {
		cout << "\r" << __FUNCTION__ << ":: " << "step: " << __epoch_batch << "/" << __epoch_max * __examples_num / __mini_batch_size
			<< ", loss: " << __loss << ", " << "learning rate: " << __learn_rate << flush;
	}

	if ((__epoch_batch * __mini_batch_size) % __examples_num == 0) {
		__epoch++;

		//if (__learn_rate_decay_enabled) decay_learn_rate();

		cout << endl << __FUNCTION__ << ":: " << "epoch: " << __epoch << "/" << __epoch_max
			<< ", loss: " << __loss << ", " << "learning rate: " << __learn_rate << endl;
	}
}
