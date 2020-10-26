#pragma once

#include <stdint.h>
#include "TSliceWrapper.h"

namespace TSlib
{
	

	template<typename T, Mode device>
	class Tensor;

	template<typename T, Mode device>
	class TensorSlice
	{
		Tensor<T, device>* source;
		std::vector<TSlice> m_slice_shape;
		std::vector<size_t> m_real_shape;
		size_t m_offset;

		#ifdef _DEBUG

		template<typename First, typename ... Args>
		void bounds_check(size_t& i, First first, Args ... remaining);

		template<typename First>
		void bounds_check(size_t& i, First first);

		#endif

		void calc_offset();

		class iterator;

		T copy_generator(const size_t& index);

	public:



		TensorSlice(Tensor<T, device>* source, const std::vector<TSlice>& slices);

		void update();

		size_t size() const;

		size_t Dims() const;

		size_t get_dim_size(const size_t& index) const;

		size_t map_index(size_t index) const;

		const std::vector<size_t>& Shape() const;

		template<typename ... Args>
		T& Get(Args ... coords);
		template<typename ... Args>
		T Get(Args ... coords) const;

		T& At(size_t index);
		T At(size_t index) const;

		template<typename RT, Mode return_device = device>
		Tensor<RT, return_device> asVector();

		T operator[](size_t index) const;
		T& operator[](size_t index);

		template<typename RT, Mode return_device = device>
		operator Tensor<RT, return_device>();

		iterator begin();

		iterator end();
	};

}
