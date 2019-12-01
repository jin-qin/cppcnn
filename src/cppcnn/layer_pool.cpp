#include "layer_pool.h"
#include <iostream>

using namespace Eigen;
using namespace std;

ImageVolume LayerPool::pool(const ImageVolume& input)
{
	if (__pool_method == PoolMethod::MAX) return max_pool(input);
	if (__pool_method == PoolMethod::AVERAGE) return avg_pool(input);
	
	return ImageVolume();
}

ImageVolume LayerPool::unpool(const ImageVolume& input)
{
	if (input.size() <= 0) {
		cout << __FUNCTION__ << ":: fatal error: " << "empty input is not allowed!" << endl;
		return ImageVolume();
	}

	auto f = __filter_size;
	auto in_rows = input[0][0].rows();
	auto in_cols = input[0][0].cols();
	LayerSize output_size;
	output_size.channels = input[0].size();
	output_size.rows = static_cast<Index>(f + (in_rows - 1) * _stride);
	output_size.cols = static_cast<Index>(f + (in_cols - 1) * _stride);

	ImageVolume input_volume_unpooled;
	for (auto& img : input) {
		Image input_unpooled;
		for (auto& c : img) {
			MatrixXd channel_unpooled = MatrixXd::Constant(output_size.rows, output_size.cols, 0);
			for (auto i = 0, i_ = 0; i < output_size.rows; i += _stride, i_++) {
				for (auto j = 0, j_ = 0; j < output_size.cols; j += _stride, j_++) {
					MatrixXd block_unpooled = MatrixXd::Constant(f, f, 0);
					block_unpooled(0, 0) = c(i_, j_);
					channel_unpooled.block(i, j, f, f) = block_unpooled;
				}
			}

			input_unpooled.push_back(channel_unpooled);
		}
		input_volume_unpooled.push_back(input_unpooled);
	}
	

	return input_volume_unpooled;
}

ImageVolume LayerPool::max_pool(const ImageVolume& input)
{
	if (input.size() <= 0) {
		cout << __FUNCTION__ << ":: fatal error: " << "empty input is not allowed!" << endl;
		return ImageVolume();
	}

	auto f = __filter_size;
	LayerSize output_size;
	output_size.channels = input[0].size();
	output_size.rows = static_cast<Index>(floor((input[0][0].rows() - 1.0 * f) / (1.0 * _stride) + 1));
	output_size.cols = static_cast<Index>(floor((input[0][0].cols() - 1.0 * f) / (1.0 * _stride) + 1));

	ImageVolume input_volume_pooled;
	for (auto& img : input) {
		Image input_pooled;
		for (auto& c : img) {
			MatrixXd channel_pooled = MatrixXd::Constant(output_size.rows, output_size.cols, 0);
			for (auto i = 0; i < output_size.rows; i += 1)
				for (auto j = 0; j < output_size.cols; j += 1)
					channel_pooled(i, j) = c.block(i * _stride, j * _stride, f, f).maxCoeff();
			input_pooled.push_back(channel_pooled);
		}

		input_volume_pooled.push_back(input_pooled);
	}
	

	return input_volume_pooled;
}

ImageVolume LayerPool::avg_pool(const ImageVolume& input)
{
	return ImageVolume();
}
