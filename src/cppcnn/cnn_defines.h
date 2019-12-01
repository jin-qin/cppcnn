#ifndef CNN_DEFINES_H
#define CNN_DEFINES_H
#include <vector>
#include <Eigen/Dense>

typedef std::vector< Eigen::MatrixXd > LayerValues, Image;
typedef std::vector< LayerValues > AllLayersValues, ImageVolume;
typedef std::vector< AllLayersValues > AllLayersValuesBatch;

typedef struct Dimension {
	Dimension() {
		this->rows = 0;
		this->cols = 0;
		this->channels = 0;
	}

	Dimension(Eigen::Index rows, Eigen::Index cols, int channels) {
		this->rows = rows;
		this->cols = cols;
		this->channels = channels;
	}
	Eigen::Index rows;
	Eigen::Index cols;
	int channels;
} LayerSize;

#endif // CNN_DEFINES_H
