#pragma once

#include "TensorSliceBones.h"

namespace TSlib
{
	TSlice::TSlice(const intmax_t& from, const intmax_t& to, const uint32_t& from_max, const uint32_t& to_max)
		:from(from), to(to), from_max(from_max), to_max(to_max)
	{
		MEASURE();
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

	TSlice::TSlice()
		: from(NULL), to(NULL), from_max(NULL), to_max(NULL)
	{
		MEASURE();
	}

	bool TSlice::contains(intmax_t val) const
	{
		MEASURE();
		return (get_from() <= val &&
			get_to() >= val);
	}

	uint32_t TSlice::get_from() const
	{
		MEASURE();
		return (uint32_t)std::abs(int((from < 0) * from_max + from));
	}

	uint32_t TSlice::get_to() const
	{
		MEASURE();
		return (uint32_t)std::abs(int((to < 0) * (to_max + 1) + to));
	}

	inline uint32_t TSlice::width() const
	{
		MEASURE();
		return get_to() - get_from();
	}

	bool TSlice::operator()(intmax_t val, size_t from_max, size_t to_max) const
	{
		MEASURE();
		return contains(val);
	}

	class TSlice::iterator
	{
		uint32_t num;
		TSlice& slice;

	public:
		iterator(uint32_t start, TSlice& slice)
			: num(start), slice(slice)
		{
			MEASURE();
		}

		iterator& operator++()
		{
			num++;
			return *this;
		}

		iterator operator++(int)
		{
			iterator retval = *this;
			++(*this);
			return retval;
		}

		bool operator==(const iterator& other) const
		{
			return num == other.num;
		}

		bool operator!=(const iterator& other) const
		{
			return num != other.num;
		}

		uint32_t operator*()
		{
			return num;
		}

		using diffrence_type = intmax_t;
		using value_type = intmax_t;
		using pointer = const intmax_t*;
		using refrence = const intmax_t&;
		using iterator_category = std::forward_iterator_tag;
	};

	TSlice::iterator TSlice::begin()
	{
		MEASURE();
		return { get_from(), *this };
	}

	TSlice::iterator TSlice::end()
	{
		MEASURE();
		return { get_to() + 1, *this };
	}

	#ifdef _DEBUG
	template<typename T, Mode device>
	template<typename First, typename ...Args>
	inline void TensorSlice<T, device>::bounds_check(size_t& i, First first, Args ...remaining)
	{
		MEASURE();
		bounds_check(i, first);
		bounds_check(i, remaining...);
	}

	template<typename T, Mode device>
	template<typename First>
	inline void TensorSlice<T, device>::bounds_check(size_t& i, First first)
	{
		MEASURE();
		assert(m_slice_shape[i].contains(first));

		i++;
	}
	#endif

	template<typename T, Mode device>
	inline void TensorSlice<T, device>::calc_offset()
	{
		MEASURE();
		size_t tmp_multiply = source->size();
		m_offset = 0;

		for (size_t i = m_slice_shape.size() - 1; i != SIZE_MAX; i--)
		{

			tmp_multiply /= m_slice_shape[i].to_max;

			m_offset += m_slice_shape[i].get_from() * tmp_multiply;
		}
	}

	template<typename T, Mode device>
	TensorSlice<T, device>::TensorSlice(Tensor<T, device>* source, const std::vector<TSlice>& slices)
		: source(source), m_slice_shape(slices)
	{
		MEASURE();
		#ifdef _DEBUG
		assert((slices.size() <= source->Dims() - 1, "There must be the same amount or less slices as dimensions in the tensor"));
		#endif

		update();
	}

	template<typename T, Mode device>
	template<typename ... Args>
	T& TensorSlice<T, device>::Get(Args ... coords)
	{
		MEASURE();
		#ifdef _DEBUG
		size_t i = 0;
		bounds_check(i, coords...);
		#endif

		return source->Get(coords...);
	}

	template<typename T, Mode device>
	template<typename ... Args>
	T TensorSlice<T, device>::Get(Args ... coords) const
	{
		MEASURE();
		#ifdef _DEBUG
		size_t i = 0;
		bounds_check(i, coords...);
		#endif

		return source->Get(coords...);
	}


	template<typename T, Mode device>
	T& TensorSlice<T, device>::At(size_t index)
	{
		MEASURE();
		return source->At(map_index(index));
	}

	template<typename T, Mode device>
	T TensorSlice<T, device>::At(size_t index) const
	{
		MEASURE();
		return source->At(map_index(index));
	}

	template<typename T, Mode device>
	template<typename RT, Mode return_device>
	Tensor<RT, return_device> TensorSlice<T, device>::asVector()
	{
		return Tensor<RT, return_device>(*this);
	}

	template<typename T, Mode device>
	inline void TensorSlice<T, device>::update()
	{
		MEASURE();
		#ifdef _DEBUG
		assert((m_slice_shape.size() <= source->Dims() - 1, "There must be the same amount or less slices as dimensions in the tensor"));
		#endif

		for (size_t i = 0; i < source->Dims() - 1; i++)
		{
			m_slice_shape[i].to_max = (uint32_t)source->Shape()[i];
		}

		calc_offset();
	}

	template<typename T, Mode device>
	size_t TensorSlice<T, device>::size() const
	{
		MEASURE();
		size_t size = m_slice_shape[0].get_to() - m_slice_shape[0].get_from();
		for (size_t i = 1; i < m_slice_shape.size(); i++)
		{
			size *= (m_slice_shape[i].get_to() - m_slice_shape[i].get_from());
		}

		return size;
	}

	template<typename T, Mode device>
	inline size_t TensorSlice<T, device>::Dims() const
	{
		MEASURE();
		return m_slice_shape.size();
	}

	template<typename T, Mode device>
	inline size_t TensorSlice<T, device>::get_dim_size(const size_t& index) const
	{
		MEASURE();
		size_t size = m_slice_shape[0].get_to() - m_slice_shape[0].get_from();

		for (size_t i = 1; i <= index; i++)
		{
			size *= m_slice_shape[i].get_to() - m_slice_shape[i].get_from();
		}

		return size;
	}

	template<typename T, Mode device>
	const std::vector<TSlice>& TensorSlice<T, device>::Shape() const
	{
		MEASURE();
		return m_slice_shape;
	}

	template<typename T, Mode device>
	size_t TensorSlice<T, device>::map_index(size_t index) const
	{
		MEASURE();
		size_t tmp_multiply = size();
		size_t new_index = 0;

		for (size_t i = m_slice_shape.size() - 1; i != SIZE_MAX; i--)
		{
			size_t rows = index / tmp_multiply;
			index -= tmp_multiply * rows;

			tmp_multiply /= m_slice_shape[i].width();

			new_index += rows * source->get_dim_size(i);

		}

		new_index += index + m_offset;

		return new_index;
	}

	template<typename T, Mode device>
	class TensorSlice<T, device>::iterator
	{
		size_t num;
		TensorSlice<T, device>& slice;

	public:
		iterator(size_t start, TensorSlice<T, device>& slice)
			: num(start), slice(slice)
		{}

		iterator& operator++()
		{
			num++;
			return *this;
		}

		iterator operator++(int)
		{
			iterator retval = *this;
			++(*this);
			return retval;
		}

		bool operator==(const iterator& other) const
		{
			return num == other.num;
		}

		bool operator!=(const iterator& other) const
		{
			return num != other.num;
		}

		T& operator*()
		{
			return slice.At(num);
		}

		using diffrence_type = intmax_t;
		using value_type = intmax_t;
		using pointer = const intmax_t*;
		using refrence = const intmax_t&;
		using iterator_category = std::forward_iterator_tag;
	};

	template<typename T, Mode device>
	typename TensorSlice<T, device>::iterator TensorSlice<T, device>::begin()
	{
		MEASURE();
		return { 0 ,*this };
	}

	template<typename T, Mode device>
	typename TensorSlice<T, device>::iterator TensorSlice<T, device>::end()
	{
		MEASURE();
		return { size() ,*this };
	}

	template<typename T, Mode device>
	T TensorSlice<T, device>::operator[](size_t index) const
	{
		MEASURE();
		return At(index);
	}

	template<typename T, Mode device>
	T& TensorSlice<T, device>::operator[](size_t index)
	{
		MEASURE();
		return At(index);
	}
}
