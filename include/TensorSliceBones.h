#pragma once

#include <stdint.h>
#include "TensorEnums.h"

namespace TSlib
{
	struct TSlice
	{
		intmax_t from;
		intmax_t to;


		uint32_t from_max = 0;
		uint32_t to_max = 0;

		TSlice(const intmax_t& from, const intmax_t& to, const uint32_t& from_max = 0, const uint32_t& to_max = 0);

		template<typename T>
		TSlice(const std::initializer_list<T>& val)
			:from(0), to(0), from_max(0), to_max(0)
		{
			MEASURE();

			from = *val.begin();
			to = *(val.begin() + 1);

			#ifdef _DEBUG

			if (from < 0 && to < 0)
			{
				assert((to > from, "negative from value must be more than negative to value"));
			}
			else
			{
				assert((to < from, "from value must be less than to value"));
			}
			#endif
		}

		TSlice();

		bool contains(intmax_t val) const;

		uint32_t get_from() const;

		uint32_t get_to() const;

		uint32_t width() const;

		bool operator()(intmax_t val, size_t from_max = 0, size_t to_max = 0) const;

		class iterator;

		iterator begin();

		iterator end();
	};

	template<typename T, Mode device>
	class Tensor;

	template<typename T, Mode device>
	class TensorSlice
	{
		Tensor<T, device>* source;
		std::vector<TSlice> m_slice_shape;
		size_t m_offset;

		#ifdef _DEBUG

		template<typename First, typename ... Args>
		void bounds_check(size_t& i, First first, Args ... remaining);

		template<typename First>
		void bounds_check(size_t& i, First first);

		#endif

		void calc_offset();

		class iterator;

	public:



		TensorSlice(Tensor<T, device>* source, const std::vector<TSlice>& slices);

		void update();

		size_t size() const;

		size_t Dims() const;

		size_t get_dim_size(const size_t& index) const;

		size_t map_index(size_t index) const;

		const std::vector<TSlice>& DimSizes() const;

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
