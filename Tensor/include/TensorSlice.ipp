#pragma once

#include "TensorSlice.h"

namespace TSlib
{
	#ifdef _TS_DEBUG
	template<typename T, Device device>
	template<typename First, typename ...Args>
	inline void TensorSlice<T, device>::bounds_check(size_t& i, First first, Args ...remaining)
	{
		
		bounds_check(i, first);
		bounds_check(i, remaining...);
	}

	template<typename T, Device device>
	template<typename First>
	inline void TensorSlice<T, device>::bounds_check(size_t& i, First first)
	{
		

		if (!m_slice_shape[i].contains(first))
		{
			throw OutOfBounds(Shape(), "Index was out of bounds", i, first);
		}

		i++;
	}
	#endif

	template<typename T, Device device>
	inline void TensorSlice<T, device>::calc_offset()
	{
		
		size_t tmp_multiply = source->size();
		m_offset = 0;

		for (size_t i = 0; i < m_slice_shape.size(); i++)
		{
			tmp_multiply /= m_slice_shape[i].to_max;

			m_offset += m_slice_shape[i].get_from() * tmp_multiply;
		}
	}

	template<typename T, Device device>
	TensorSlice<T, device>::TensorSlice(Tensor<T, device>* source, const std::vector<TSlice>& slices)
		: source(source), m_slice_shape(slices), m_real_shape(source->Dims()), m_shape(m_real_shape)
	{
		

		#ifdef _TS_DEBUG

		if (slices.size() > (source->Dims()))
		{
			throw BadShape(source, "There must be the same amount or less slices as dimensions in the slice", slices);
		}
		#endif

		update();
	}

	template<typename T, Device device>
	template<typename First>
	void TensorSlice<T, device>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord)
	{
		

		#ifdef _TS_DEBUG
		if (m_shape[iter] <= coord)
			throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", iter, coord);
		#endif

		tmp_multiply /= m_shape[iter];
		indx += coord * tmp_multiply;

		iter++;
	}

	template<typename T, Device device>
	template<typename First, typename... Args>
	void TensorSlice<T, device>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining)
	{
		

		get_indx(indx, iter, tmp_multiply, coord);
		get_indx(indx, iter, tmp_multiply, remaining...);
	}

	template<typename T, Device device>
	template<typename ... Args>
	T& TensorSlice<T, device>::Get(Args ... coords)
	{
		

		#ifdef _TS_DEBUG
		if (Dims() != sizeof...(coords))
		{
			throw BadValue("Exception was thrown, because there were not the same nuumber of coordinates given as the number of dimensions in the Tensor", ExceptValue("Coords", sizeof...(coords)), ExceptValue("Dimensions", Dims()));
		}
		#endif

		#ifdef _TS_DEBUG
		size_t i_d = 0;
		bounds_check(i_d, coords...);
		#endif

		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);
		size_t i = 0;

		get_indx(index, i, tmp_multiply, coords...);

		return At(index);
	}

	template<typename T, Device device>
	inline T& TensorSlice<T, device>::Get(const std::vector<size_t>& coords)
	{
		#ifdef _TS_DEBUG
		if (Dims() != coords.size())
		{
			throw BadValue("Exception was thrown, because there were not the same nuumber of coordinates given as the number of dimensions in the Tensor", ExceptValue("Coords", coords.size()), ExceptValue("Dimensions", Dims()));
		}
		#endif

		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);

		for (size_t i = 0; i < Dims(); i++)
		{
			#ifdef _TS_DEBUG
			if (Shape()[i] <= coords[i])
				throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", i, coords[i]);
			#endif

			tmp_multiply /= Shape()[i];
			index += coords[i] * tmp_multiply;
		}

		return At(index);
	}

	template<typename T, Device device>
	template<typename ... Args>
	T TensorSlice<T, device>::Get(Args ... coords) const
	{
		
		#ifdef _TS_DEBUG
		size_t i_d = 0;
		bounds_check(i_d, coords...);
		#endif

		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);
		size_t i = 0;

		get_indx(index, i, tmp_multiply, coords...);

		return At(index);
	}

	template<typename T, Device device>
	inline T TensorSlice<T, device>::Get(const std::vector<size_t>& coords) const
	{
		#ifdef _TS_DEBUG
		if (Dims() != coords.size())
		{
			throw BadValue("Exception was thrown, because there were not the same nuumber of coordinates given as the number of dimensions in the Tensor", ExceptValue("Coords", coords.size()), ExceptValue("Dimensions", Dims()));
		}

		#endif

		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);

		for (size_t i = 0; i < Dims(); i++)
		{
			#ifdef _TS_DEBUG
			if (Shape()[i] <= coords[i])
				throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", i, coords[i]);
			#endif

			tmp_multiply /= Shape()[i];
			index += coords[i] * tmp_multiply;
		}

		return At(index);
	}

	template<typename T, Device device>
	T& TensorSlice<T, device>::At(size_t index)
	{
		
		return source->At(MapIndex(index));
	}

	template<typename T, Device device>
	T TensorSlice<T, device>::At(size_t index) const
	{
		
		return source->At(MapIndex(index));
	}

	template<typename T, Device device>
	template<typename RT, Device return_device>
	Tensor<RT, return_device> TensorSlice<T, device>::asVector()
	{
		return Tensor<RT, return_device>(*this);
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::update()
	{
		
		#ifdef _TS_DEBUG

		if (m_slice_shape.size() > source->Dims())
		{
			throw BadShape(source, "There must be the same amount or less slices as dimensions in the tensor", Shape());
		}
		#endif

		m_slice_shape.resize(source->Dims(), All);
		m_shape.resize(source->Dims());

		for (size_t i = 0; i < Dims(); i++)
		{
			m_slice_shape[i].to_max = (uint32_t)source->Shape()[i];
			m_real_shape[i] = m_slice_shape[i].width();
			m_shape[i] = m_real_shape[i];
		}

		calc_offset();
	}

	template<typename T, Device device>
	template<typename OT, Device device_other>
	void TensorSlice<T, device>::Fill(const Tensor<OT, device_other>& other)
	{
		#ifdef _TS_DEBUG

		if (Dims() != other.Dims())
		{
			throw BadShape("There must be the same amount dimensions", other.Shape(), Shape());
		}

		for (size_t i = 0; i < Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("The shapes passed must be the same", other.Shape(), Shape());
			}
		}
		#endif

		Compute([&](T& elem, const size_t& index) {elem = other[index]; });
	}

	template<typename T, Device device>
	template<typename OT, Device device_other>
	void TensorSlice<T, device>::Fill(const TensorSlice<OT, device_other>& other)
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

		Compute([&](T& elem, const size_t& index) {elem = other[index]; });
	}

	template<typename T, Device device>
	void TensorSlice<T, device>::Fill(const T& val)
	{
		Compute([&](T& elem) {elem = val; });
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Fill(std::function<T(const size_t&)> generator)
	{
		
		Compute([&](T& elem, const size_t& index) {elem = generator(index); });
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Fill(std::function<T(const std::vector<size_t>&)> generator)
	{
		
		Compute([&](T& elem, const std::vector<size_t>& dimensions) {elem = generator(dimensions); });
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator)
	{
		Compute([&](T& elem, const std::vector<size_t>& dimensions, const size_t& index) {elem = generator(dimensions, index); });
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Fill(const std::vector<T>& vals)
	{
		#ifdef _TS_DEBUG
		if (vals.size() != size())
		{
			throw BadShape("Vector must have the same size as the target tensor slice", m_slice_shape, std::vector<size_t>{ vals.size() });
		}
		#endif

		Compute([&](T& elem, const size_t& index) {elem = vals[index]; });
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Compute(std::function<void(T&)> compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(At(index));
		}
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Compute(std::function<void(T&, const size_t&)> compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(At(index), index);
		}
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Compute(std::function<void(T&, const std::vector<size_t>&)> compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			std::vector<size_t> coords(Dims());

			size_t tmp_indx = index;

			for (size_t j = 0; j < Dims(); j++)
			{
				coords[Dims() - j - 1] = tmp_indx % Shape()[Dims() - j - 1];
				tmp_indx /= Shape()[Dims() - j - 1];
			}

			compute_func(At(index), coords);
		}
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Compute(std::function<void(T&, const std::vector<size_t>&, const size_t&)> compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			std::vector<size_t> coords(Dims());

			size_t tmp_indx = index;

			for (size_t j = 0; j < Dims(); j++)
			{
				coords[Dims() - j - 1] = tmp_indx % Shape()[Dims() - j - 1];
				tmp_indx /= Shape()[Dims() - j - 1];
			}

			compute_func(At(index), coords, index);
		}
	}

	template<typename T, Device device>
	inline Tensor<T, device> TensorSlice<T, device>::Compute(std::function<void(T&, const T&)> compute_func, size_t axis, T pad_val, bool keepDims) const
	{
		std::vector<size_t> return_shape(Shape());

		return_shape[axis] = 1;

		Tensor<T, device> result(return_shape, pad_val);

		result.Compute([&](T& elem, const std::vector<size_t>& coords)
			{
				std::vector<size_t> new_coords = coords;
				new_coords[axis] = 0;

				for (size_t i = 0; i < Shape()[axis]; i++)
				{
					compute_func(elem, Get(new_coords));
					new_coords[axis]++;
				}
			});

		if (!keepDims)
		{
			return_shape.resize(result.Dims() - 1);

			for (size_t i = 0; i < return_shape.size(); i++)
			{
				return_shape[i] = Shape()[i] * (i < axis) + Shape()[i + 1] * (i >= axis);
			}

			result.Reshape(return_shape);
		}

		return result;
	}

	template<typename T, Device device>
	inline Tensor<T, device> TensorSlice<T, device>::Compute(std::function<void(T&, const T&, const std::vector<size_t>&)> compute_func, size_t axis, T pad_val, bool keepDims) const
	{
		std::vector<size_t> return_shape(Shape());

		return_shape[axis] = 1;

		Tensor<T, device> result(return_shape, pad_val);

		result.Compute([&](T& elem, const std::vector<size_t>& coords, const size_t& index)
			{
				std::vector<size_t> new_coords = coords;
				new_coords[axis] = 0;

				for (size_t i = 0; i < Shape()[axis]; i++)
				{
					compute_func(elem, Get(new_coords), index);
					new_coords[axis]++;
				}
			});

		if (!keepDims)
		{
			return_shape.resize(result.Dims() - 1);

			for (size_t i = 0; i < return_shape.size(); i++)
			{
				return_shape[i] = Shape()[i] * (i < axis) + Shape()[i + 1] * (i >= axis);
			}

			result.Reshape(return_shape);
		}

		return result;
	}

	template<typename T, Device device>
	inline Tensor<T, device> TensorSlice<T, device>::Compute(std::function<void(T&, const T&, const size_t&)> compute_func, size_t axis, T pad_val, bool keepDims) const
	{
		std::vector<size_t> return_shape(Shape());

		return_shape[axis] = 1;

		Tensor<T, device> result(return_shape, pad_val);

		result.Compute([&](T& elem, const std::vector<size_t>& coords, const size_t& index)
			{
				std::vector<size_t> new_coords = coords;
				new_coords[axis] = 0;

				for (size_t i = 0; i < Shape()[axis]; i++)
				{
					compute_func(elem, Get(new_coords), index);
					new_coords[axis]++;
				}
			});

		if (!keepDims)
		{
			return_shape.resize(result.Dims() - 1);

			for (size_t i = 0; i < return_shape.size(); i++)
			{
				return_shape[i] = Shape()[i] * (i < axis) + Shape()[i + 1] * (i >= axis);
			}

			result.Reshape(return_shape);
		}

		return result;
	}

	template<typename T, Device device>
	inline Tensor<T, device> TensorSlice<T, device>::Compute(std::function<void(T&, const T&, const std::vector<size_t>&, const size_t&)> compute_func, size_t axis, T pad_val, bool keepDims) const
	{
		std::vector<size_t> return_shape(Shape());

		return_shape[axis] = 1;

		Tensor<T, device> result(return_shape, pad_val);

		result.Compute([&](T& elem, const std::vector<size_t>& coords, const size_t& index)
			{
				std::vector<size_t> new_coords = coords;
				new_coords[axis] = 0;

				for (size_t i = 0; i < Shape()[axis]; i++)
				{
					compute_func(elem, Get(new_coords), new_coords, index);
					new_coords[axis]++;
				}
			});

		if (!keepDims)
		{
			return_shape.resize(result.Dims() - 1);

			for (size_t i = 0; i < return_shape.size(); i++)
			{
				return_shape[i] = Shape()[i] * (i < axis) + Shape()[i + 1] * (i >= axis);
			}

			result.Reshape(return_shape);
		}

		return result;
	}

	template<typename T, Device device>
	inline void TensorSlice<T, device>::Replace(const T& target, const T& value)
	{
		Compute([&](T& elem) {if (target == elem) elem = value; });
	}

	template<typename T, Device device>
	size_t TensorSlice<T, device>::size() const
	{
		
		size_t size = m_real_shape[0];
		for (size_t i = 1; i < m_real_shape.size(); i++)
		{
			size *= m_real_shape[i];
		}

		return size;
	}

	template<typename T, Device device>
	inline size_t TensorSlice<T, device>::Dims() const
	{
		
		return m_shape.size();
	}

	template<typename T, Device device>
	inline size_t TensorSlice<T, device>::get_real_size(const size_t& index) const
	{
		size_t r_size = 1;

		for (size_t i = 0; i <= index; i++)
		{
			r_size *= m_shape[Dims() - i - 1];
		}

		return r_size;
	}

	template<typename T, Device device>
	size_t TensorSlice<T, device>::get_dim_length(const size_t& index) const
	{
		

		size_t r_size = 1;

		for (size_t i = index; i < m_shape.size(); i++)
		{
			r_size *= m_shape[Dims() - i - 1];
		}

		return r_size;
	}

	template<typename T, Device device>
	const std::vector<size_t>& TensorSlice<T, device>::Shape() const
	{
		
		return m_shape;
	}

	template<typename T, Device device>
	const std::vector<TSlice>& TensorSlice<T, device>::TSliceShape() const
	{
		
		return m_slice_shape;
	}

	template<typename T, Device device>
	TensorSlice<T, device>& TensorSlice<T, device>::Reshape(const std::vector<long long>& shape)
	{
		
		#ifdef _TS_DEBUG
		long long new_shape_size = 1;
		size_t unknown = 0;
		for (const long long& elem : shape)
		{
			if (elem < 0)
				unknown++;
			new_shape_size *= elem;
		}

		if (unknown > 1)
			throw BadValue("There cannot be passed more than 1 unknown dimension when reshapeing a Tensor", unknown);
		else if (size() != new_shape_size && unknown == 0)
		{
			throw BadShape("New shape does not fit the current tensor's dimensions", Shape(), shape);
		}
		#endif

		long long unknown_pos = -1;
		size_t shape_product = 1;

		for (size_t i = 0; i < shape.size(); i++)
		{
			unknown_pos = i * (shape[i] < 0) + unknown_pos * (shape[i] >= 0);
			shape_product *= shape[i] * (shape[i] >= 0) + (shape[i] < 0);
		}

		size_t unknown_value = size() / shape_product;

		#ifdef _TS_DEBUG
		if (double(unknown_value) != round(double(size()) / double(shape_product), 1000))
		{
			throw BadShape("The unknown dimension is impossible to fit with the given shape", Shape(), shape);
		}
		#endif

		m_shape.clear();
		m_shape.reserve(shape.size());

		for (size_t i = 0; i < shape.size(); i++)
		{
			m_shape.push_back(shape[i] * (i != unknown_pos) + unknown_value * (i == unknown_pos));
		}

		return *this;
	}

	template<typename T, Device device>
	size_t TensorSlice<T, device>::MapIndex(size_t index) const
	{
		
		size_t tmp_multiply = m_slice_shape[Dims() - 1].width();
		size_t new_index = 0;

		for (size_t i = 0; i < m_slice_shape.size(); i++)
		{
			size_t rows = index / tmp_multiply;
			index -= tmp_multiply * rows;

			tmp_multiply *= m_real_shape[i];

			new_index += rows * source->get_real_size(i);
		}

		new_index += index + m_offset;

		return new_index;
	}

	template<typename T, Device device>
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

	template<typename T, Device device>
	T TensorSlice<T, device>::copy_generator(const size_t& index)
	{
		return At(index);
	}

	template<typename T, Device device>
	template<typename RT, Device return_device>
	TensorSlice<T, device>::operator Tensor<RT, return_device>()
	{
		return Tensor<T, device>(*this);
	}

	template<typename T, Device device>
	typename TensorSlice<T, device>::iterator TensorSlice<T, device>::begin()
	{
		
		return { 0 ,*this };
	}

	template<typename T, Device device>
	typename TensorSlice<T, device>::iterator TensorSlice<T, device>::end()
	{
		
		return { size() ,*this };
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Add(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) + other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Add(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) + other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::Add(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) + other;
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::AddAsgmt(const Tensor<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) += other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::AddAsgmt(const TensorSlice<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) += other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void TensorSlice<T, device>::AddAsgmt(const OT& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) += other;
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Subtract(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) - other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Subtract(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) - other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::Subtract(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) - other;
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::SubtractAsgmt(const Tensor<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) -= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::SubtractAsgmt(const TensorSlice<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) -= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void TensorSlice<T, device>::SubtractAsgmt(const OT& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) -= other;
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Multiply(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) * other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Multiply(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) * other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::Multiply(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) * other;
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::MultiplyAsgmt(const Tensor<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) *= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::MultiplyAsgmt(const TensorSlice<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) *= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void TensorSlice<T, device>::MultiplyAsgmt(const OT& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) *= other;
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Divide(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) / other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::Divide(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) / other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> TensorSlice<T, device>::Divide(const OT& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) / other;
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::divideAsgmt(const Tensor<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) /= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::divideAsgmt(const TensorSlice<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) /= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void TensorSlice<T, device>::divideAsgmt(const OT& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) /= other;
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::modulou(const Tensor<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) % other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	Tensor<T, device> TensorSlice<T, device>::modulou(const TensorSlice<OT, other_device>& other)
	{
		Tensor<T, device> r_val(*this);

		for (size_t i = 0; i < size(); i++)
		{
			r_val[i] = At(i) % other[i];
		}

		return r_val;
	}

	template<typename T, Device device>
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

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::ModulouAsgmt(const Tensor<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) %= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	void TensorSlice<T, device>::ModulouAsgmt(const TensorSlice<OT, other_device>& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) %= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void TensorSlice<T, device>::ModulouAsgmt(const OT& other)
	{
		for (size_t i = 0; i < size(); i++)
		{
			At(i) %= other;
		}
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> TensorSlice<T, device>::Compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		
		#ifdef _TS_DEBUG
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

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> TensorSlice<T, device>::Compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		
		#ifdef _TS_DEBUG
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

	template<typename T, Device device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> TensorSlice<T, device>::Compare(const OT& other, bool(*comp_func)(const T&, const OT&))
	{
		

		Tensor<RT, device> result(this->Shape());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other);
		}

		return result;
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator+(const Tensor<OT, other_device>& other)
	{
		return Add(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator+(const TensorSlice<OT, other_device>& other)
	{
		return Add(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator+(const OT& other)
	{
		return Add(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator-(const Tensor<OT, other_device>& other)
	{
		return Subtract(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator-(const TensorSlice<OT, other_device>& other)
	{
		return Subtract(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator-(const OT& other)
	{
		return Subtract(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator*(const Tensor<OT, other_device>& other)
	{
		return Multiply(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator*(const TensorSlice<OT, other_device>& other)
	{
		return Multiply(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator*(const OT& other)
	{
		return Multiply(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator/(const Tensor<OT, other_device>& other)
	{
		return Divide(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator/(const TensorSlice<OT, other_device>& other)
	{
		return Divide(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline Tensor<T, device> TensorSlice<T, device>::operator/(const OT& other)
	{
		return Divide(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator%(const Tensor<OT, other_device>& other)
	{
		return modulou(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline Tensor<T, device> TensorSlice<T, device>::operator%(const TensorSlice<OT, other_device>& other)
	{
		return modulou(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator+=(const Tensor<OT, other_device>& other)
	{
		AddAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator+=(const TensorSlice<OT, other_device>& other)
	{
		AddAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline void TensorSlice<T, device>::operator+=(const OT& other)
	{
		AddAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator-=(const Tensor<OT, other_device>& other)
	{
		SubtractAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator-=(const TensorSlice<OT, other_device>& other)
	{
		SubtractAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline void TensorSlice<T, device>::operator-=(const OT& other)
	{
		SubtractAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator*=(const Tensor<OT, other_device>& other)
	{
		MultiplyAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator*=(const TensorSlice<OT, other_device>& other)
	{
		MultiplyAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline void TensorSlice<T, device>::operator*=(const OT& other)
	{
		MultiplyAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator/=(const Tensor<OT, other_device>& other)
	{
		divideAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator/=(const TensorSlice<OT, other_device>& other)
	{
		divideAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline void TensorSlice<T, device>::operator/=(const OT& other)
	{
		divideAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator%=(const Tensor<OT, other_device>& other)
	{
		ModulouAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline void TensorSlice<T, device>::operator%=(const TensorSlice<OT, other_device>& other)
	{
		ModulouAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	inline void TensorSlice<T, device>::operator%=(const OT& other)
	{
		ModulouAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator==(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other).template Sum<size_t>() == other.size();
		#else
		return Compare(other).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator==(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other).template Sum<size_t>() == other.size();
		#else
		return Compare(other).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator==(const OT& other)
	{
		#ifdef __clang__
		return Compare(other).template Sum<size_t>() == size();
		#else
		return Compare(other).Sum<size_t>() == size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator!=(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, NotEqual).template Sum<size_t>() == other.size();
		#else
		return Compare(other, NotEqual).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator!=(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, NotEqual).template Sum<size_t>() == other.size();
		#else
		return Compare(other, NotEqual).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator!=(const OT& other)
	{
		#ifdef __clang__
		return Compare(other, NotEqual).template Sum<size_t>() == size();
		#else
		return Compare(other, NotEqual).Sum<size_t>() == size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator<(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, LessThan).template Sum<size_t>() == other.size();
		#else
		return Compare(other, LessThan).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator<(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, LessThan).template Sum<size_t>() == other.size();
		#else
		return Compare(other, LessThan).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator<(const OT& other)
	{
		#ifdef __clang__
		return Compare(other, LessThan).template Sum<size_t>() == size();
		#else
		return Compare(other, LessThan).Sum<size_t>() == size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator>(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, GreaterThan).template Sum<size_t>() == other.size();
		#else
		return Compare(other, GreaterThan).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator>(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, GreaterThan).template Sum<size_t>() == other.size();
		#else
		return Compare(other, GreaterThan).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator>(const OT& other)
	{
		#ifdef __clang__
		return Compare(other, GreaterThan).template Sum<size_t>() == size();
		#else
		return Compare(other, GreaterThan).Sum<size_t>() == size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator<=(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, LessThanEqual).template Sum<size_t>() == other.size();
		#else
		return Compare(other, LessThanEqual).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator<=(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, LessThanEqual).template Sum<size_t>() == other.size();
		#else
		return Compare(other, LessThanEqual).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator<=(const OT& other)
	{
		#ifdef __clang__
		return Compare(other, LessThanEqual).template Sum<size_t>() == size();
		#else
		return Compare(other, LessThanEqual).Sum<size_t>() == size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator>=(const Tensor<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, GreaterThanEqual).template Sum<size_t>() == other.size();
		#else
		return Compare(other, GreaterThanEqual).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT, Device other_device>
	inline bool TensorSlice<T, device>::operator>=(const TensorSlice<OT, other_device>& other)
	{
		#ifdef __clang__
		return Compare(other, GreaterThanEqual).template Sum<size_t>() == other.size();
		#else
		return Compare(other, GreaterThanEqual).Sum<size_t>() == other.size();
		#endif
	}

	template<typename T, Device device>
	template<typename OT>
	inline bool TensorSlice<T, device>::operator>=(const OT& other)
	{
		#ifdef __clang__
		return Compare(other, GreaterThanEqual).template Sum<size_t>() == size();
		#else
		return Compare(other, GreaterThanEqual).Sum<size_t>() == size();
		#endif
	}

	template<typename T, Device device>
	T TensorSlice<T, device>::operator[](size_t index) const
	{
		
		return At(index);
	}

	template<typename T, Device device>
	T& TensorSlice<T, device>::operator[](size_t index)
	{
		
		return At(index);
	}

	template<typename T, Device device>
	inline std::string TensorSlice<T, device>::printable() const
	{
		size_t max_length = 0;
		std::stringstream stream;

		for (size_t i = 0; i < size(); i++)
		{
			max_length = std::max(std::to_string(At(i)).size(), max_length);
		}

		for (size_t i = 0; i < m_shape[Dims() - 1]; i++)
		{
			stream << std::to_string(At(i));

			size_t str_len = std::to_string(At(i)).size();

			for (size_t j = 0; j < max_length - str_len; j++)
			{
				stream << ' ';
			}

			stream << ',';

			if (i % m_shape[Dims() - 1] == m_shape[Dims() - 1] - 1)
			{
				stream << '\n';
			}
		}

		for (size_t dim = 1; dim < Dims(); dim++)
		{
			stream << "\n";
			for (size_t i = get_real_size(dim - 1); i < get_real_size(dim); i++)
			{
				stream << std::to_string(At(i));

				size_t str_len = std::to_string(At(i)).size();

				for (size_t j = 0; j < max_length - str_len; j++)
				{
					stream << ' ';
				}

				stream << ',';

				if (i % m_shape[Dims() - 1] == m_shape[Dims() - 1] - 1)
				{
					stream << '\n';
				}
			}
		}

		return stream.str();
	}
}