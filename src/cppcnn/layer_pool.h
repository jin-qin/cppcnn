#ifndef LAYER_POOL_H
#define LAYER_POOL_H

#include "layer_base.h"
#include "cnn_defines.h"

class LayerPool : public LayerBase {
public:
	enum class PoolMethod {
		UNKNOWN,
		MAX,
		AVERAGE
	};

public:
	LayerPool(const std::weak_ptr<CppCNN>& p_cnn, int f, PoolMethod pm = PoolMethod::MAX, int s = 1)
	: LayerBase(0, s, LayerBase::LayerType::POOL, p_cnn)
	, __pool_method(pm)
	, __filter_size(f) {}
	~LayerPool() {}

	ImageVolume pool(const ImageVolume& input);
	ImageVolume unpool(const ImageVolume& input);

private:
	ImageVolume max_pool(const ImageVolume& input);
	ImageVolume avg_pool(const ImageVolume& input);

public:
	PoolMethod __pool_method = PoolMethod::UNKNOWN;
	int __filter_size = 0;
};

#endif // LAYER_POOL_H