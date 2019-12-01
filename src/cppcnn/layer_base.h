#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include "cppcnn.h"
#include <memory>

class LayerBase {
public:
	enum class LayerType {
		UNKNOWN,
		CONV,		// Convolution layer
		POOL,		// Pooling layer
		FC			// Fully-connected layer
	};

public:
	LayerBase() {}
	LayerBase(int p, int s, LayerType t, const std::weak_ptr<CppCNN>& p_cnn)
	: _padding(p)
	, _stride(s)
	, _type(t)
	, _cnn(p_cnn){}

	virtual ~LayerBase() {}

	virtual inline int padding() const { return _padding; }
	virtual inline int stride() const { return _stride; }
	virtual inline LayerType type() const { return _type; }

protected:
	int _padding = 0;
	int _stride = 1;
	LayerType _type = LayerType::UNKNOWN;
	std::weak_ptr<CppCNN> _cnn;
};

#endif // LAYER_BASE_H