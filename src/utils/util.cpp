#include "util.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace util {
    Eigen::MatrixXd read_mnist_matrix(const std::string &file_path, bool is_image) {
        using namespace std;
        using namespace Eigen;
		cout << __FUNCTION__ << ":: " << "loading data: " << file_path << endl;
        ifstream infs(file_path, std::ios::binary);
        if (!infs.is_open()) {
            cout << __FUNCTION__ << ":: Cannot open the file!" << endl;
            return MatrixXd();
        }

        int magic_num = 0;
        int dimension_num = 0;
        int items_num = 0;
        int img_rows = 0;
        int img_cols = 0;

        if (infs.read(reinterpret_cast<char*>(&magic_num), 4)) {
            auto pmn = reinterpret_cast<char*>(&magic_num);
            dimension_num = static_cast<int>(pmn[3]);
        } else {
            cout << __FUNCTION__ << ":: Cannot read the magic number!" << endl;
            return MatrixXd();
        }

        if (!infs.read(reinterpret_cast<char*>(&items_num), 4)) {
            cout << __FUNCTION__ << ":: Cannot read the images number!" << endl;
            return MatrixXd();
        }
        if (is_image && !infs.read(reinterpret_cast<char*>(&img_rows), 4)) {
            cout << __FUNCTION__ << ":: Cannot read the rows number!" << endl;
            return MatrixXd();
        }
        if (is_image && !infs.read(reinterpret_cast<char*>(&img_cols), 4)) {
            cout << __FUNCTION__ << ":: Cannot read the cols number!" << endl;
            return MatrixXd();
        }

        items_num = swap_endian(items_num);
        if (is_image) {
            img_rows = swap_endian(img_rows);
            img_cols = swap_endian(img_cols);
        }

        size_t len_pixels = is_image ? (items_num * img_rows * img_cols) : items_num;
        vector<double> _data(len_pixels, 0);
		uint8_t ubyte_rd = 0;
		size_t rd_ind = 0;
		while (infs.read(reinterpret_cast<char*>(&ubyte_rd), 1))
			_data[rd_ind++] = static_cast<double>(ubyte_rd);

		cout << __FUNCTION__ << ":: " << "finish. " << endl;
        return MatrixXd::Map(&_data[0], is_image ? img_rows * img_cols : items_num, is_image ? items_num : 1);
    }

	std::pair<ImageVolume, Eigen::MatrixXd> read_mnist(
		const std::string& file_path_examples,
		const std::string& file_path_labels)
	{
		using namespace std;
		using namespace Eigen;
		cout << __FUNCTION__ << ":: " << "loading examples: " << file_path_examples << endl;
		cout << __FUNCTION__ << ":: " << "loading labels: " << file_path_labels << endl;

		ifstream infs_examples(file_path_examples, std::ios::binary);
		ifstream infs_labels(file_path_labels, std::ios::binary);
		if (!infs_examples.is_open() || !infs_labels.is_open()) {
			cout << __FUNCTION__ << ":: Cannot open the file!" << endl;
			return pair<ImageVolume, Eigen::MatrixXd>();
		}

		int magic_num_examples = 0, magic_num_labels = 0;
		int dimension_num_examples = 0;
		int items_num_examples = 0, items_num_labels = 0;
		int img_rows = 0;
		int img_cols = 0;

		if (infs_examples.read(reinterpret_cast<char*>(&magic_num_examples), 4) &&
			infs_labels.read(reinterpret_cast<char*>(&magic_num_labels), 4)) {
			auto pmn = reinterpret_cast<char*>(&magic_num_examples);
			dimension_num_examples = static_cast<int>(pmn[3]);
		} else {
			cout << __FUNCTION__ << ":: Cannot read the magic number!" << endl;
			return pair<ImageVolume, Eigen::MatrixXd>();
		}

		if (!infs_examples.read(reinterpret_cast<char*>(&items_num_examples), 4) ||
			!infs_labels.read(reinterpret_cast<char*>(&items_num_labels), 4)) {
			cout << __FUNCTION__ << ":: Cannot read the images number!" << endl;
			return pair<ImageVolume, Eigen::MatrixXd>();
		}
		if (!infs_examples.read(reinterpret_cast<char*>(&img_rows), 4)) {
			cout << __FUNCTION__ << ":: Cannot read the rows number!" << endl;
			return pair<ImageVolume, Eigen::MatrixXd>();
		}
		if (!infs_examples.read(reinterpret_cast<char*>(&img_cols), 4)) {
			cout << __FUNCTION__ << ":: Cannot read the cols number!" << endl;
			return pair<ImageVolume, Eigen::MatrixXd>();
		}

		items_num_examples = swap_endian(items_num_examples);
		items_num_labels = swap_endian(items_num_labels);
		img_rows = swap_endian(img_rows);
		img_cols = swap_endian(img_cols);

		size_t len_pixels_examples = items_num_examples * img_rows * img_cols;
		size_t len_pixels_labels = items_num_labels;
		size_t len_image = img_rows * img_cols;
		ImageVolume examples;
		MatrixXd labels;

		vector<uint8_t> ubytes_rd(len_image, 0);
		while (infs_examples.read(reinterpret_cast<char*>(&ubytes_rd[0]), len_image)) {
			vector<double> _data(ubytes_rd.begin(), ubytes_rd.end());
			auto img = MatrixXd::Map(&_data[0], img_rows, img_cols);
			examples.push_back(vector<MatrixXd>({ img }));
		}

		ubytes_rd = vector<uint8_t>(len_pixels_labels, 0);
		if (!infs_labels.read(reinterpret_cast<char*>(&ubytes_rd[0]), len_pixels_labels)) {
			return std::pair<ImageVolume, Eigen::MatrixXd>();
		}

		vector<double> labels_data(ubytes_rd.begin(), ubytes_rd.end());
		labels = MatrixXd::Map(&labels_data[0], items_num_labels, 1);

		cout << __FUNCTION__ << ":: " << "finish. " << endl;

		return make_pair(examples, labels);
	}

	uint32_t swap_endian(uint32_t val) {
		val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
		return (val << 16) | (val >> 16);
	}

	Eigen::VectorXd argmax(const Eigen::MatrixXd& m, int axis) {
		using namespace Eigen;

		const bool _rowwise = (axis == 0);

		size_t _len = _rowwise ? m.rows() : m.cols();
		VectorXd _indices(_len);

		Index ind = -1;
		for (int i = 0; i < _len; i++) {
			if (_rowwise)
				m.row(i).maxCoeff(&ind);
			else
				m.col(i).maxCoeff(&ind);
			_indices[i] = static_cast<double>(ind);
		}

		return _indices;
	}

	double f_relu(double value) {
		return value <= 0 ? 0.0 : value; 
	}

	double df_relu(double value) {
		return value <= 0 ? 0.0 : 1;
	}
}