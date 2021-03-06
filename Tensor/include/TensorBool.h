#pragma once

#include "Tensor.h"

namespace TSlib
{
	// in addition to adding some extra functions, this specialization also stores the bool values as one bit per bool instead of one byte per bool
	template<Device device>
	class Tensor<bool, device>: public Tensor<unsigned char, device>
	{
	protected:
		std::vector<size_t> m_bool_shape;

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

		bool Get(const std::vector<size_t>& coords) const;
		template<typename ... Args>
		bool Get(const Args& ... coords) const;

		bool At(size_t indx) const;

		void SetIdx(bool val, size_t indx);
		void Set(bool val, const std::vector<size_t>& coords);
		template<typename ... Args>
		void Set(bool val, const Args& ... coords);

		void ToggleIdx(size_t indx);
		void Toggle(const std::vector<size_t>& coords);
		template<typename ... Args>
		void Toggle(const Args& ... coords);

		bool* Data() = delete;


	};
}

#include "TensorBool.ipp"
