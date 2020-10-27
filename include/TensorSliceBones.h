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

		template<typename T, Mode device>
		void add(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void add(const TensorSlice<T, device>& other);
		template<typename T>
		void add(const T& other);

		template<typename T, Mode device>
		void subtract(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void subtract(const TensorSlice<T, device>& other);
		template<typename T>
		void subtract(const T& other);

		template<typename T, Mode device>
		void multiply(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void multiply(const TensorSlice<T, device>& other);
		template<typename T>
		void multiply(const T& other);

		template<typename T, Mode device>
		void dot(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void dot(const TensorSlice<T, device>& other);
		template<typename T>
		void dot(const T& other);

		template<typename T, Mode device>
		void divide(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void divide(const TensorSlice<T, device>& other);
		template<typename T>
		void divide(const T& other);

		template<typename T, Mode device>
		void compare(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void compare(const TensorSlice<T, device>& other);
		template<typename T>
		void compare(const T& other);

		template<typename T, Mode device>
		void lessThan(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void lessThan(const TensorSlice<T, device>& other);
		template<typename T>
		void lessThan(const T& other);

		template<typename T, Mode device>
		void greaterThan(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void greaterThan(const TensorSlice<T, device>& other);
		template<typename T>
		void greaterThan(const T& other);

		template<typename T, Mode device>
		void lessThanEqual(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void lessThanEqual(const TensorSlice<T, device>& other);
		template<typename T>
		void lessThanEqual(const T& other);

		template<typename T, Mode device>
		void greaterThanEqual(const Tensor<T, device>& other);
		template<typename T, Mode device>
		void greaterThanEqual(const TensorSlice<T, device>& other);
		template<typename T>
		void greaterThanEqual(const T& other);
	};

}
