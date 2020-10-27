#pragma once

#include "TensorSliceBones.h"
#include "TensorExceptions.h"

namespace TSlib
{
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

		if (!m_slice_shape[i].contains(first))
		{
			throw OutOfBounds(Shape(), "Index was out of bounds", i, first);
		}

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
		: source(source), m_slice_shape(slices), m_real_shape(m_slice_shape.size())
	{
		MEASURE();

		#ifdef _DEBUG

		if (slices.size() > (source->Dims() - 1))
		{
			throw BadShape(source, "There must be the same amount or less slices as dimensions in the slice", slices);
		}
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

		if (m_slice_shape.size() > source->Dims() - 1)
		{
			throw BadShape(source, "There must be the same amount or less slices as dimensions in the tensor", Shape());
		}
		#endif

		for (size_t i = 0; i < Dims(); i++)
		{
			m_slice_shape[i].to_max = (uint32_t)source->Shape()[i];
			m_real_shape[i] = m_slice_shape[i].width();
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
	const std::vector<size_t>& TensorSlice<T, device>::Shape() const
	{
		MEASURE();
		return m_real_shape;
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
	T TSlib::TensorSlice<T, device>::copy_generator(const size_t& index)
	{
		return At(index);
	}

	template<typename T, Mode device>
	template<typename RT, Mode return_device>
	TensorSlice<T, device>::operator Tensor<RT, return_device>()
	{
		return Tensor<T, device>(*this, false);
	}

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
