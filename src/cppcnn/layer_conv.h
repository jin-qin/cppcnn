#ifndef LAYER_CONV_H
#define LAYER_CONV_H

#include <Eigen/Dense>
#include <vector>
#include "layer_base.h"
#include "cnn_defines.h"

class LayerConv : public LayerBase {
public:
	LayerConv(const std::weak_ptr<CppCNN>& p_cnn, int f, int nf, int p = 0, int s = 1);
    ~LayerConv() {}

	inline size_t filter_size() const { return __filter_size; }
	inline size_t num_filters() const { return __num_filters; }
	inline ImageVolume& filters() { return __filters; }

	inline void set_filter_size(int f) { __filter_size = f; }
	inline void set_filter_nums(int nf) { __num_filters = nf; }

	/**
	* @brief, convolution layer convolve with the input.
	*/
	std::pair< ImageVolume, ImageVolume > convolution(const ImageVolume& input);
	ImageVolume convolution_backward(const ImageVolume& input, const ImageVolume& prev_z, const LayerSize& output_size);

	void compute_gradients(const ImageVolume& a, const ImageVolume& delta_prev);

	void update_model(size_t batch_size);

private:
	void init_filters(size_t channels);

	void reset_grads();

	/**
	* @brief, one of filter channels convolve with one of the input channels
	*/
	Eigen::MatrixXd conv(const Eigen::MatrixXd& s_input, 
						 const Eigen::MatrixXd& s_filter,
						 const LayerSize& outpu_size, int p, int s);

	ImageVolume apply_activation(const ImageVolume& input);
	Eigen::MatrixXd img2mat(const ImageVolume& input, const LayerSize& output_size, size_t f, int p, int s);
	Eigen::MatrixXd filters2mat(const ImageVolume& filters, bool filter_rot_180 = false);
	ImageVolume mat2img(const Eigen::MatrixXd& conv_mat, const LayerSize& output_size);

private:
	size_t __filter_size = 0;
	size_t __num_filters = 0;
	size_t __channels = 0; // filter channels

	ImageVolume __filters; // C, N, W, H, __filters[0][1] means the second filter at channel 0
	ImageVolume __filters_grad;
	Eigen::VectorXd __baises;
	Eigen::VectorXd __baises_grad;
};

#endif // LAYER_CONV_H