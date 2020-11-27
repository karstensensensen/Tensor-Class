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
	template<typename OT, Mode device_other>
	void TensorSlice<T, device>::Fill(const Tensor<OT, device_other>& other)
	{
		#ifdef DEBUG
		
		if (Dims() != other.Dims())
		{
			throw BadShape(Shape(), "There must be the same amount dimensions", other.Shape());
		}

		for(size_t i = 0; i < Dims(); i++)
		{
			if(Shape()[i] != other.Shape()[i])
			{
				throw BadShape(Shape(), "The shapes passed must be the same", other.Shape());
			{
		}
		#endif

		for (size_t i = 0; i < size(); i++)
		{
			At(i) = other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode device_other>
	void TSlib::TensorSlice<T, device>::Fill(const TensorSlice<OT, device_other>& other)
	{
		#ifdef DEBUG

		if (Dims() != other.Dims())
		{
			throw BadShape(Shape(), "There must be the same amount dimensions", other.Shape());
		}

		for (size_t i = 0; i < Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(Shape(), "The shapes passed must be the same", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < size(); i++)
		{
			At(i) = other[i];
		}
	}

	template<typename T, Mode device>
	void TensorSlice<T, device>::Fill(const T& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) = other;
		}
	}

	template<typename T, Mode device>
	inline void TensorSlice<T, device>::Fill(std::function<T(const size_t&)> generator)
	{
		MEASURE();
		for (size_t i = 0; i < size(); i++)
		{
			At(i) = generator(i);
		}
	}

	template<typename T, Mode device>
	inline void TensorSlice<T, device>::Fill(std::function<T(const std::vector<size_t>&)> generator)
	{
		MEASURE();
		std::vector<size_t> indexes(Dims());

		for (size_t i = 0; i < size(); i++)
		{
			indexes[0] = (i % source->get_real_size(0));
			for (size_t j = 1; j < Dims(); j++)
			{
				indexes[j] = (i / source->get_real_size(j - 1)) % source->get_real_size(j - 1);
			}
			At(i) = generator(indexes);
		}
	}

	template<typename T, Mode device>
	inline void TensorSlice<T, device>::Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator)
	{
		MEASURE();
		std::vector<size_t> indexes(Dims());

		for (size_t i = 0; i < size(); i++)
		{
			indexes[0] = (i % source->get_real_size(0));
			for (size_t j = 1; j < Dims(); j++)
			{
				indexes[j] = (i / source->get_real_size(j - 1)) % source->get_real_size(j - 1);
			}
			At(i) = generator(indexes, i);
		}
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
	template<typename RT>
	inline RT TSlib::TensorSlice<T, device>::sum()
	{
		RT sum = At(0);

		for (size_t i = 1; i < size(); i++)
		{
			sum += At(i);
		}

		return sum;
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
	T TensorSlice<T, device>::copy_generator(const size_t& index)
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
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::add(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) + other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::add(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) + other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::add(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) + other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::addAsgmt(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);
		
		for (size_t i = 0; i < size(); i++)
		{
			At(i) += other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::addAsgmt(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);
		
		for (size_t i = 0; i < size(); i++)
		{
			At(i) += other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	void TensorSlice<T, device>::addAsgmt(const OT& other)
	{
		Tensor<T, device> r_val(*this);
		
		for (size_t i = 0; i < size(); i++)
		{
			At(i) += other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::subtract(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) - other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::subtract(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) - other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::subtract(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) - other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::subtractAsgmt(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);
		
		for (size_t i = 0; i < size(); i++)
		{
			At(i) -= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::subtractAsgmt(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);
		
		for (size_t i = 0; i < size(); i++)
		{
			At(i) -= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	void TensorSlice<T, device>::subtractAsgmt(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) -= other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::multiply(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) * other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::multiply(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) * other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::multiply(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) * other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::multiplyAsgmt(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) *= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::multiplyAsgmt(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) *= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	void TensorSlice<T, device>::multiplyAsgmt(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) *= other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::divide(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) / other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::divide(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) / other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::divide(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) / other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::divideAsgmt(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) /= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::divideAsgmt(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) /= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	void TensorSlice<T, device>::divideAsgmt(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) /= other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::modulou(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) % other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	Tensor<T, device> TensorSlice<T, device>::modulou(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) % other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::modulou(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) % other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::modulouAsgmt(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) %= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	void TensorSlice<T, device>::modulouAsgmt(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) %= other[i];
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename OT>
	void TensorSlice<T, device>::modulouAsgmt(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) %= other;
		}

		return r_val;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> TensorSlice<T, device>::compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", Shape(), other.Shape());
		}

		for (size_t i = 0; i < Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", Shape(), other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(Shape());

		for (size_t i = 0; i < size(); i++)
		{
			result[i] = comp_func(At(i), other[i]);
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> TensorSlice<T, device>::compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", Shape(), other.Shape());
		}

		for (size_t i = 0; i < Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", Shape(), other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(Shape());

		for (size_t i = 0; i < size(); i++)
		{
			result[i] = comp_func(At(i), other[i]);
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> TensorSlice<T, device>::compareSingle(const OT& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();

		Tensor<RT, device> result(this->Shape());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other);
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator+(const Tensor<OT, other_device>& other)
	{
		return add(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TSlib::TensorSlice<T, device>::operator+(const TensorSlice<OT, other_device>& other)
	{
		return add(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator+(const OT& other)
	{
		return add(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator-(const Tensor<OT, other_device>& other)
	{
		return subtract(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator-(const TensorSlice<OT, other_device>& other)
	{
		return subtract(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator-(const OT& other)
	{
		return subtract(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TSlib::TensorSlice<T, device>::operator*(const Tensor<OT, other_device>& other)
	{
		return multiply(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator*(const TensorSlice<OT, other_device>& other)
	{
		return multiply(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator*(const OT& other)
	{
		return multiply(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator/(const Tensor<OT, other_device>& other)
	{
		return divide(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator/(const TensorSlice<OT, other_device>& other)
	{
		return divide(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator/(const OT& other)
	{
		return divide(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator%(const Tensor<OT, other_device>& other)
	{
		return modulou(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator%(const TensorSlice<OT, other_device>& other)
	{
		return modulou(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator%(const OT& other)
	{
		return modulou(other);
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator==(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other).template sum<size_t>();
		#else
		return compare(other).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator==(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other).template sum<size_t>();
		#else
		return compare(other).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator==(const OT& other)
	{
		#ifdef __clang__
		return compareSingle(other).template sum<size_t>();
		#else
		return compareSingle(other).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator!=(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, NotEqual).template sum<size_t>();
		#else
		return compare(other, NotEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator!=(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, NotEqual).template sum<size_t>();
		#else
		return compare(other, NotEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator!=(const OT& other)
	{
		#ifdef __clang__
		return compareSingle(other, NotEqual).template sum<size_t>();
		#else
		return compareSingle(other, NotEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator<(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, LessThan).template sum<size_t>();
		#else
		return compare(other, LessThan).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator<(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, LessThan).template sum<size_t>();
		#else
		return compare(other, LessThan).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator<(const OT& other)
	{
		#ifdef __clang__
		return compareSingle(other, LessThan).template sum<size_t>();
		#else
		return compareSingle(other, LessThan).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator>(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, GreaterThan).template sum<size_t>();
		#else
		return compare(other, GreaterThan).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator>(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, GreaterThan).template sum<size_t>();
		#else
		return compare(other, GreaterThan).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator>(const OT& other)
	{
		#ifdef __clang__
		return compareSingle(other, GreaterThan).template sum<size_t>();
		#else
		return compareSingle(other, GreaterThan).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator<=(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, LessThanEqual).template sum<size_t>();
		#else
		return compare(other, LessThanEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator<=(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, LessThanEqual).template sum<size_t>();
		#else
		return compare(other, LessThanEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator<=(const OT& other)
	{
		#ifdef __clang__
		return compareSingle(other, LessThanEqual).template sum<size_t>();
		#else
		return compareSingle(other, LessThanEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator>=(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, GreaterThanEqual).template sum<size_t>();
		#else
		return compare(other, GreaterThanEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT, Mode other_device>
	inline bool TensorSlice<T, device>::operator>=(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return compare(other, GreaterThanEqual).template sum<size_t>();
		#else
		return compare(other, GreaterThanEqual).sum<size_t>();
		#endif
	}

	template<typename T, Mode device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator>=(const OT& other)
	{
		#ifdef __clang__
		return compareSingle(other, GreaterThanEqual).template sum<size_t>();
		#else
		return compareSingle(other, GreaterThanEqual).sum<size_t>();
		#endif
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
