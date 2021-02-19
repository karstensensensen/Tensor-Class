#pragma once

#include "Tensor.h"

namespace TSlib
{
	template<Device device>
	class Tensor<bool, device>: public Tensor<char, device>
	{

		size_t bool_dim;
		using Tensor<char, device>::m_shape;

	public:
		Tensor();
		Tensor(const std::vector<size_t>& sizes, bool pad_val = False);
		Tensor(const std::vector<size_t>& sizes, std::function<bool()> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<bool(const size_t&)> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<bool(const std::vector<size_t>&)> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<bool(const std::vector<size_t>&, const size_t&)> generator);
		Tensor(const TensorSlice<bool, device>& slicee);

		Tensor(const Tensor<bool, device>& other);

		Tensor<bool, device>& Resize(const std::vector<size_t>& sizes, bool pad_val = false);



	};
}

#include "TensorBool.ipp"
