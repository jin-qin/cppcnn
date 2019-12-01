#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <Eigen/Dense>
#include "../cppcnn/cnn_defines.h"

namespace util {
    Eigen::MatrixXd read_mnist_matrix(const std::string &file_path, bool is_image);

	std::pair< ImageVolume, Eigen::MatrixXd> read_mnist(
		const std::string& file_path_examples,
		const std::string& file_path_labels);

	uint32_t swap_endian(uint32_t val);

	/**
	 * @brief Returns the indices of the maximum values along an axis
	 * @param m, should be a 2 dimension matrix
	 * @param axis, should be 0 (for rowwise) or 1(for colwise)
	 */ 
	Eigen::VectorXd argmax(const Eigen::MatrixXd &m, int axis);

	double f_relu(double value);
	double df_relu(double value);
}

#endif // UTIL_H