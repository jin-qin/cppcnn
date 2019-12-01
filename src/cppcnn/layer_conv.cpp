#include "layer_conv.h"
#include <random>
#include <iostream>

#include "util.h"

using namespace Eigen;
using namespace std;

LayerConv::LayerConv(const weak_ptr<CppCNN>& p_cnn, int f, int nf, int p, int s)
	: LayerBase(p, s, LayerBase::LayerType::CONV, p_cnn)
	, __filter_size(f)
	, __num_filters(nf) {
}

pair< ImageVolume, ImageVolume > LayerConv::convolution(const ImageVolume& input) {
	if (input.size() <= 0) {
		cout << __FUNCTION__ << ":: fatal error: " << "empty input is not allowed!" << endl;
		return pair< ImageVolume, ImageVolume >();
	}
	auto channels = input[0].size(); // filter channels number

	if (__filters.size() <= 0) init_filters(channels);

	LayerSize output_size;
	output_size.channels = __num_filters;
	output_size.rows = static_cast<Index>(floor((input[0][0].rows() + 2.0 * _padding - __filter_size) / (1.0 * _stride) + 1));
	output_size.cols = static_cast<Index>(floor((input[0][0].cols() + 2.0 * _padding - __filter_size) / (1.0 * _stride) + 1));

	auto input_mat = img2mat(input, output_size, __filter_size, _padding, _stride);
	auto filters_mat = filters2mat(__filters);

	MatrixXd conv_mat = input_mat * filters_mat;
	conv_mat.rowwise() += __baises.transpose();
	auto output_z = mat2img(conv_mat, output_size);
	auto output_a = apply_activation(output_z);

	return make_pair(output_z, output_a);
}

ImageVolume LayerConv::convolution_backward(const ImageVolume& dZ, const ImageVolume& next_z, const LayerSize& output_size) {
	if (dZ.size() <= 0) {
		cout << __FUNCTION__ << ":: fatal error: " << "empty input is not allowed!" << endl;
		return ImageVolume();
	}
	auto f_channels = dZ[0].size(); // filter channels number
	auto f_num = output_size.channels; // filters number
	auto padding = floor((output_size.rows + __filter_size - dZ[0][0].rows() - 1) / 2.0);

	ImageVolume filters_reshape;
	for (auto c = 0; c < f_channels; c++) {
		LayerValues filter;
		for (auto f_ind = 0; f_ind < f_num; f_ind++)
			filter.push_back(__filters[f_ind][c]);
		filters_reshape.push_back(filter);
	}

	auto input_mat = img2mat(dZ, output_size, __filter_size, padding, 1);
	auto filters_mat = filters2mat(filters_reshape, true);
	MatrixXd conv_mat = input_mat * filters_mat;
	auto output = mat2img(conv_mat, output_size);

	// f_num is the new channels.
	for (auto img_ind = 0; img_ind < output.size(); img_ind++) {
		for (auto c = 0; c < f_num; c++) {
			MatrixXd prev_z_df = next_z[img_ind][c].unaryExpr(&util::df_relu);
			output[img_ind][c] = output[img_ind][c].cwiseProduct(prev_z_df);
		}
	}

	return output;
}

void LayerConv::compute_gradients(const ImageVolume& a, const ImageVolume& delta_prev) {
	// delta_prev is filter map here
	if (a.size() <= 0 || delta_prev.size() <= 0) {
		cout << __FUNCTION__ << ":: fatal error: " << "empty input is not allowed!" << endl;
		return;
	}

	auto f_channels = a[0].size(); // filter channels number
	auto f_num = delta_prev[0].size(); // filters number
	auto output_size = LayerSize(__filter_size, __filter_size, a.size());
	auto padding = floor((a[0][0].rows() - delta_prev[0][0].rows() + __filter_size- 1) / 2.0);

	for (auto img_ind = 0; img_ind < a.size(); img_ind++) {
		for (auto f_ind = 0; f_ind < f_num; f_ind++) {
			for (auto c = 0; c < f_channels; c++) {
				auto dw = conv(a[img_ind][c], delta_prev[img_ind][f_ind], output_size, padding, 1);
				__filters_grad[c][f_ind] += dw;
			}
			__baises_grad[f_ind] += delta_prev[img_ind][f_ind].sum();
		}
	}

	update_model(a.size());
}

void LayerConv::update_model(size_t batch_size) {
	auto cnn = _cnn.lock();
	if (!cnn) return;

	// update the model parameters
	if (cnn->__momentum_enabled && !cnn->__adam_enabled) {
		// TODO
	} else if (cnn->__adam_enabled) {
		// TODO
	} else {
		for (auto c = 0; c < __filters.size(); c++)
			for (auto i = 0; i < __filters[0].size(); i++)
				__filters[c][i] = __filters[c][i] - cnn->__learn_rate * (__filters_grad[c][i] / (1.0 * batch_size));

		__baises = __baises - cnn->__learn_rate * (__baises_grad / (1.0 * batch_size));
	}

	reset_grads();
}

MatrixXd LayerConv::conv(const MatrixXd& s_input, const MatrixXd& s_filter, const LayerSize& output_size, int p, int s) {
	MatrixXd pad_input = MatrixXd::Constant(s_input.rows() + 2 * p, s_input.cols() + 2 * p, 0);
	pad_input.block(p, p, s_input.rows(), s_input.cols()) = s_input;
	
	auto f = s_filter.rows();
	MatrixXd conv_result = MatrixXd::Constant(output_size.rows, output_size.cols, 0);
	for (int i = 0; i < output_size.rows; i += s)
		for (int j = 0; j < output_size.cols; j += s)
			conv_result(i, j) = pad_input.block(i, j, f, f).cwiseProduct(s_filter).sum();

	return conv_result;
}

ImageVolume LayerConv::apply_activation(const ImageVolume& input) {
	using namespace util;
	ImageVolume layer_active;
	for (auto& img : input) {
		Image img_active;
		for (auto& channel : img)
			img_active.push_back(channel.unaryExpr(&f_relu));
		layer_active.push_back(img_active);
	}

	return layer_active;
}

Eigen::MatrixXd LayerConv::img2mat(const ImageVolume& input, const LayerSize& output_size, size_t f, int p, int s) {
	auto f_size = f * f; // filter total size.
	auto rows = input[0][0].rows();
	auto cols = input[0][0].cols();
	auto conv_size = output_size.rows * output_size.cols;
	auto rows_input_mat = conv_size * input.size();
	auto cols_input_mat = input[0].size() * f_size;
	MatrixXd channel_pad = MatrixXd::Constant(rows + 2 * p, cols + 2 * p, 0);
	MatrixXd input_mat = MatrixXd::Constant(rows_input_mat, cols_input_mat, 0);
	for (auto img_ind = 0; img_ind < input.size(); img_ind++) {
		for (auto c = 0; c < input[0].size(); c++) {
			auto& channel = input[img_ind][c];
			channel_pad.block(p, p, rows, cols) = channel;
			for (int i = 0, row = 0; i < output_size.rows; i += s) {
				for (int j = 0; j < output_size.cols; j += s, row++) {
					MatrixXd v = channel_pad.block(i, j, f, f);
					v.transposeInPlace();
					v.resize(1, f_size);
					input_mat.block(img_ind * conv_size + row, c * f_size, 1, f_size) = v;
				}
			}
		}
	}

	return input_mat;
}

Eigen::MatrixXd LayerConv::filters2mat(const ImageVolume& filters, bool filter_rot_180) {
	auto f_channels = filters.size();
	auto f_num = filters[0].size();
	auto f_size = filters[0][0].size();

	MatrixXd filter_mat = MatrixXd::Constant(f_channels * f_size, f_num, 0);
	for (auto i = 0; i < f_num; i++) {
		for (auto c = 0; c < f_channels; c++) {
			auto filter = filters[c][i];
			if (filter_rot_180) {
				filter.colwise().reverseInPlace();
				filter.rowwise().reverseInPlace();
			}
				
			filter.transposeInPlace();
			filter.resize(f_size, 1);
			filter_mat.block(c * f_size, i, f_size, 1) = filter;
		}
	}
	return filter_mat;
}

ImageVolume LayerConv::mat2img(const Eigen::MatrixXd& conv_mat, const LayerSize& output_size) {
	auto rows = output_size.rows;
	auto cols = output_size.cols;
	auto opt_img_sz = rows * cols; // output image size.

	ImageVolume output_volume;
	for (auto img_ind = 0; img_ind < conv_mat.rows(); img_ind += opt_img_sz) {
		Image output_img;
		for (auto c = 0; c < conv_mat.cols(); c++) {
			MatrixXd img_channel = conv_mat.block(img_ind, c, opt_img_sz, 1);
			img_channel.resize(cols, rows);
			img_channel.transposeInPlace();

			output_img.push_back(img_channel);
		}

		output_volume.push_back(output_img);
	}
	
	return output_volume;
}

void LayerConv::init_filters(size_t channels) {
	__channels = channels;

	random_device rd;
	normal_distribution<double> dis(0.0, 1.0);
	MatrixXd filter_grad = MatrixXd::Constant(__filter_size, __filter_size, 0);

	__baises = VectorXd::Constant(__num_filters, 0);
	__baises_grad = VectorXd::Constant(__num_filters, 0);
	for (auto c = 0; c < __channels; c++) {
		LayerValues filter_volume;
		LayerValues filter_grad_volume;
		for (auto i = 0; i < __num_filters; i++) {
			MatrixXd filter = MatrixXd::NullaryExpr(__filter_size, __filter_size, [&]() { return dis(rd) / pow(__filter_size, 2); });
			//MatrixXd filter = MatrixXd::NullaryExpr(__filter_size, __filter_size, [&]() { return 0.001; });
			filter_volume.push_back(filter);
			filter_grad_volume.push_back(filter_grad);
		}

		__filters.push_back(filter_volume);
		__filters_grad.push_back(filter_grad_volume);
	}
}

void LayerConv::reset_grads() {
	__baises_grad.setConstant(0);
	for (auto& c : __filters_grad)
		for (auto& f : c)
			f.setConstant(0);
}
