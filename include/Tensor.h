#pragma once

#ifdef __clang__
typedef double double_t;
#ifdef DEBUG
#define _TS_DEBUG
#endif
#endif

#include <stdlib.h>

#ifdef TENSOR_PROFILING
#include <Profiler.h>
#else
#define MEASURE()
#endif

#ifdef _CUDA

#include "TensorCuda.cuh"

#else

#include "TensorBones.h"

#endif

#include <sstream>

#include "TensorOperators.h"
#include "TensorSlice.h"
#include <algorithm>
#include <math.h>
#include <cstdarg>
#include <numeric>
#include <tuple>

namespace TSlib
{
	/// Tensor private functions

	template<typename T, Mode device>
	size_t Tensor<T, device>::get_real_size(const size_t& index) const
	{
		MEASURE();

		size_t r_size = 1;

		for (size_t i = 0; i <= index; i++)
		{
			r_size *= m_shape[Dims() - i - 1];
		}

		return r_size;
	}

	template<typename T, Mode device>
	size_t Tensor<T, device>::get_dim_length(const size_t& index) const
	{
		MEASURE();

		size_t r_size = 1;

		for (size_t i = index; i < m_shape.size(); i++)
		{
			r_size *= m_shape[Dims() - i - 1];
		}

		return r_size;
	}

	template<typename T, Mode device>
	std::vector<size_t> Tensor<T, device>::FlattenDims(size_t dims) const
	{
		MEASURE();

		std::vector<size_t> new_dim(dims, 1);

		size_t i;
		for (i = 0; i < std::min(dims, Dims()); i++)
		{
			new_dim[i] = m_shape[i];
		}
		i--;

		for (size_t j = i + 1; j < m_shape.size(); j++)
		{
			new_dim[i] *= m_shape[j];
		}

		return new_dim;
	}

	template<typename T, Mode device>
	size_t Tensor<T, device>::FlattenDims() const
	{
		MEASURE();

		size_t new_dim = m_shape[0];

		for (size_t j = 1; j < m_shape.size(); j++)
		{
			new_dim *= m_shape[j];
		}

		return new_dim;
	}

	template<typename T, Mode device>
	size_t Tensor<T, device>::get_dim_offset(const size_t& index) const
	{
		MEASURE();
		size_t result = 0;
		for (size_t i = 0; i < index; i++)
		{
			result += m_shape[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename First>
	void Tensor<T, device>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord)
	{
		MEASURE();

		#ifdef _TS_DEBUG
		if (m_shape[iter] <= coord)
			throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", iter, coord);
		#endif

		tmp_multiply /= m_shape[iter];
		indx += coord * tmp_multiply;

		iter++;
	}

	template<typename T, Mode device>
	template<typename First, typename... Args>
	void Tensor<T, device>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining)
	{
		MEASURE();

		get_indx(indx, iter, tmp_multiply, coord);
		get_indx(indx, iter, tmp_multiply, remaining...);
	}

	template<typename T, Mode device>
	template<typename First>
	void Tensor<T, device>::to_vector(std::vector<TSlice>& vec, const std::initializer_list<First>& first)
	{
		vec.push_back(TSlice(first));
	}

	template<typename T, Mode device>
	template<typename ... Args, typename First>
	void Tensor<T, device>::to_vector(std::vector<TSlice>& vec, const std::initializer_list<First>& first, const std::initializer_list<Args>& ... args)
	{
		#ifdef _TS_DEBUG
		std::cout << std::is_integral<First>::value << '\n' << !std::is_integral<First>::value << '\n';
		if (!std::is_integral<First>::value)
		{
			#ifdef __clang__
			throw BadType("Integral", typeid(First).name());
			#else
			throw BadType::BadType("Integral", typeid(First).name());
			#endif
		}
		#endif

		to_vector(vec, first);
		to_vector(vec, args...);
	}

	template<typename T, Mode device>
	std::string Tensor<T, device>::printable() const
	{
		size_t max_length = 0;
		std::stringstream stream;

		for (const T& elem : *this)
		{
			max_length = std::max(std::to_string(elem).size(), max_length);
		}

		for (size_t i = 0; i < Shape()[Dims() - 1]; i++)
		{
			stream << std::to_string(At(i));

			size_t str_len = std::to_string(At(i)).size();

			for (size_t j = 0; j < max_length - str_len; j++)
			{
				stream << ' ';
			}

			stream << ',';

			if (i % Shape()[Dims() - 1] == Shape()[Dims() - 1] - 1)
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

				if (i % Shape()[Dims() - 1] == Shape()[Dims() - 1] - 1)
				{
					stream << '\n';
				}
			}
		}

		return stream.str();
	}

	/// This is a helper struct, that acts as the sort function, for the std::sort function in Tensor::based_sort function.

	namespace
	{
		struct sorter
		{
			const std::vector<size_t>& vec;

			sorter(const std::vector<size_t>& vec)
				:vec(vec)
			{}

			bool operator()(size_t a, size_t b)
			{
				return vec[a] < vec[b];
			}
		};

		struct sorterSlice
		{
			const std::vector<TSlice>& vec;

			sorterSlice(const std::vector<TSlice>& vec)
				:vec(vec)
			{}

			bool operator()(size_t a, size_t b)
			{
				return vec[a].width() < vec[b].width();
			}
		};
	}

	template<typename T, Mode device>
	std::vector<size_t> Tensor<T, device>::based_sort(const std::vector<size_t>& target)
	{
		std::vector<size_t> new_indexes(target.size());
		//
		std::generate(new_indexes.begin(), new_indexes.end(), [n = target.size() - 1]() mutable {return n--; });

		std::sort(new_indexes.begin(), new_indexes.end(), sorter(target));

		return new_indexes;
	}

	template<typename T, Mode device>
	std::vector<size_t> Tensor<T, device>::based_sort(const std::vector<TSlice>& target)
	{
		std::vector<size_t> new_indexes(target.size());
		std::iota(new_indexes.begin(), new_indexes.end(), 0);

		std::sort(new_indexes.begin(), new_indexes.end(), sorterSlice(target));

		return new_indexes;
	}

	/// Tensor constructors

	template<typename T, Mode device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, const T& pad_val)
		: m_shape(sizes.size())
	{
		MEASURE();
		Resize(sizes, pad_val);
	}

	//generator functions take an index / coordinate as parameters and returns a value with the containter type

	template<typename T, Mode device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, std::function<T(const size_t&)> generator)

		: m_shape(sizes.size())
	{
		Resize(sizes);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) = generator({ i });
		}
	}

	template<typename T, Mode device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&)> generator)
		: m_shape(sizes.size())
	{
		Resize(sizes);

		std::vector<size_t> indexes(Dims());

		for (size_t i = 0; i < size(); i++)
		{
			indexes[0] = (i % get_real_size(0));
			for (size_t j = 1; j < Dims(); j++)
			{
				indexes[j] = (i / get_real_size(j - 1)) % get_real_size(j - 1);
			}
			At(i) = generator(indexes);
		}
	}

	template<typename T, Mode device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&, const size_t&)> generator)
		: m_shape(sizes.size())
	{
		Resize(sizes);

		std::vector<size_t> indexes(Dims());

		for (size_t i = 0; i < size(); i++)
		{
			indexes[0] = (i % get_real_size(0));
			for (size_t j = 1; j < Dims(); j++)
			{
				indexes[j] = (i / get_real_size(j - 1)) % get_real_size(j - 1);
			}
			At(i) = generator(indexes, i);
		}
	}

	template<typename T, Mode device>
	Tensor<T, device>::Tensor(const TensorSlice<T, device>& slice)
		: m_shape(slice.Shape().size())
	{
		MEASURE();

		Resize(slice.Shape());

		for (size_t i = 0; i < slice.size(); i++)
		{
			At(i) = slice[i];
		}
	}

	template<typename T, Mode device>
	Tensor<T, device>::Tensor(const Tensor<T, device>& other)
		: m_vector(other.asVector()), m_shape(other.Shape())
	{
		MEASURE();

		#ifdef _CUDA

		if (other.isAllocated())
		{
			allocate();
			cudaMemcpy(gpu_mem, other.gpu_mem, sizeof(T) * size(), cudaMemcpyDeviceToDevice);
		}

		#endif
	}

	template<typename T, Mode device>
	Tensor<T, device>::~Tensor()
	{
		#ifdef _CUDA
		if (isAllocated())
		{
			deallocate();
		}
		#endif
	}

	/// Tensor public functions

	template<typename T, Mode device>
	void Tensor<T, device>::Fill(const T& val)
	{
		MEASURE();
		for (T& elem : *this)
		{
			elem = val;
		}
	}

	template<typename T, Mode device>
	void Tensor<T, device>::Fill(const size_t& dim, const T& val, const size_t& index)
	{
		MEASURE();
		for (size_t i = 0; i < get_real_size(dim); i++)
		{
			At(i + get_dim_offset(dim) * index) = val;
		}
	}

	template<typename T, Mode device>
	inline void TSlib::Tensor<T, device>::Fill(std::function<T(const size_t&)> generator)
	{
		MEASURE();
		for (size_t i = 0; i < size(); i++)
		{
			At(i) = generator(i);
		}
	}

	template<typename T, Mode device>
	inline void Tensor<T, device>::Fill(std::function<T(const std::vector<size_t>&)> generator)
	{
		MEASURE();
		std::vector<size_t> indexes(Dims());

		for (size_t i = 0; i < size(); i++)
		{
			indexes[0] = (i % get_real_size(0));
			for (size_t j = 1; j < Dims(); j++)
			{
				indexes[j] = (i / get_real_size(j - 1)) % get_real_size(j - 1);
			}
			At(i) = generator(indexes);
		}
	}

	template<typename T, Mode device>
	inline void Tensor<T, device>::Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator)
	{
		MEASURE();
		std::vector<size_t> indexes(Dims());

		for (size_t i = 0; i < size(); i++)
		{
			indexes[0] = (i % get_real_size(0));
			for (size_t j = 1; j < Dims(); j++)
			{
				indexes[j] = (i / get_real_size(j - 1)) % get_real_size(j - 1);
			}
			At(i) = generator(indexes, i);
		}
	}

	template<typename T, Mode device>
	inline void Tensor<T, device>::Fill(std::vector<T> vals)
	{
		#ifdef _TS_DEBUG
		if (vals.size() != size())
		{
			throw BadShape(this, "Vector must have the same size as the target tensor", std::vector<size_t>{ vals.size() });
		}
		#endif

		memcpy(Data(), vals.data(), size()*sizeof(T));

	}

	template<typename T, Mode device>
	inline void Tensor<T, device>::Replace(const T& target, const T& value)
	{
		for (size_t i = 0; i < size(); i++)
		{
			if (target == At(i))
			{
				At(i) = value;
			}
		}
	}

	/// resize functions

	template<typename T, Mode device>
	void Tensor<T, device>::upscale_dim(const size_t& index, const size_t& row_size, const size_t& amount, const T& pad_val)
	{
		MEASURE();
		if (0 < get_dim_length(index + 1))
		{
			size_t insert_length = amount / get_dim_length(index + 1);
			if (!insert_length)
				return;
			for (size_t j = 0; j < get_dim_length(index + 1); j++)
			{
				m_vector.insert(begin() + (row_size + get_real_size(index) * j), insert_length, pad_val);
			}
		}
	}

	template<typename T, Mode device>
	void Tensor<T, device>::downscale_dim(const size_t& index, const size_t& row_size, const size_t& amount)
	{
		MEASURE();
		if (0 < get_dim_length(index + 1))
		{
			size_t erase_length = amount / get_dim_length(index + 1);
			if (!erase_length)
				return;
			for (size_t j = 0; j < get_dim_length(index + 1); j++)
			{
				size_t offset = row_size + get_real_size(index) * j;
				m_vector.erase(begin() + offset - erase_length, begin() + offset);
			}
		}
	}

	template<typename T, Mode device>
	void Tensor<T, device>::ResizeDim(const size_t& dim, const size_t& amount, const T& pad_val)
	{
		MEASURE();
		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (isAllocated())
			deallocate();
		#endif

		//Reserve size

		size_t new_amount = amount;
		size_t tmp_size = m_vector.size();
		size_t tmp_row_size = get_real_size(dim);

		m_shape[dim] = amount;

		if (dim != 0)
			new_amount = get_real_size(dim - 1) * get_dim_length(dim);
		else

			new_amount = get_dim_length(dim);

		if (new_amount > tmp_size)
		{
			m_vector.reserve(new_amount);
			new_amount -= tmp_size;

			//Resize dimension
			upscale_dim(dim, tmp_row_size, new_amount, pad_val);
		}
		else
		{
			new_amount = tmp_size - new_amount;
			downscale_dim(dim, tmp_row_size, new_amount);
		}
	}

	template<typename T, Mode device>
	inline size_t Tensor<T, device>::calc_new_size(const std::initializer_list<size_t>& sizes)
	{
		MEASURE();
		size_t new_size = 1;
		size_t index = 0;
		for (const size_t& size : sizes)
		{
			new_size *= size * (size >= m_shape[index]) +
				m_shape[index] * (size < m_shape[index]);
			index++;
		}
		return new_size;
	}

	template<typename T, Mode device>
	inline size_t Tensor<T, device>::calc_new_size(const std::vector<size_t>& sizes)
	{
		MEASURE();
		size_t new_size = 1;
		for (const size_t& elem_size : sizes)
		{
			new_size *= elem_size;
		}
		return new_size;
	}

	template<typename T, Mode device>
	void Tensor<T, device>::Resize(const std::vector<size_t>& sizes, const T& pad_val)
	{
		MEASURE();
		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (isAllocated())
			deallocate();
		#endif

		#ifdef _TS_DEBUG
		if (sizes.size() == 0)
		{
			throw BadShape("New shape must not be of length 0");
		}
		#endif

		SetDims(sizes.size());

		size_t current_size = get_real_size(m_shape.size() - 1);
		size_t new_size = calc_new_size(sizes);

		if (current_size < new_size)
			m_vector.reserve(new_size - current_size);

		std::vector<size_t> dimensions = based_sort(sizes);

		for (size_t i = 0; i < sizes.size(); i++)
		{
			size_t target_size = sizes.size() - dimensions[i] - 1;
			size_t new_amount = NULL;
			size_t tmp_size = size();
			size_t tmp_row_size = get_real_size(dimensions[i]);

			m_shape[target_size] = sizes[target_size];

			if (dimensions[i] != 0)
				new_amount = get_real_size(dimensions[i] - 1) * get_dim_length(dimensions[i]);
			else
				new_amount = get_dim_length(dimensions[i]);

			if (new_amount > tmp_size)
			{
				new_amount -= tmp_size;

				//Resize dimension
				upscale_dim(dimensions[i], tmp_row_size, new_amount, pad_val);
			}
			else
			{
				new_amount = tmp_size - new_amount;

				downscale_dim(dimensions[i], tmp_row_size, new_amount);
			}
		}

		m_vector.shrink_to_fit();
	}

	template<typename T, Mode device>
	void Tensor<T, device>::Reshape(const std::initializer_list<size_t>& shape)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		size_t new_shape_size = 1;
		for (const size_t& elem : shape)
		{
			new_shape_size *= elem;
		}

		if (size() != new_shape_size)
			throw BadShape(this, "New shape does not fit the current tensor's dimensions", shape);
		#endif

		m_shape.erase(m_shape.begin(), m_shape.end());

		for (const size_t& elem : shape)
		{
			m_shape.push_back(elem);
		}
	}

	template<typename T, Mode device>
	void Tensor<T, device>::Reshape(const std::vector<size_t>& shape)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		size_t new_shape_size = 1;
		for (const size_t& elem : shape)
		{
			new_shape_size *= elem;
		}

		if (size() != new_shape_size)
		{
			throw BadShape(this, "New shape does not fit the current tensor's dimensions", shape);
		}
		#endif

		m_shape.erase(m_shape.begin(), m_shape.end());

		for (const size_t& elem : shape)
		{
			m_shape.push_back(elem);
		}
	}

	template<typename T, Mode device>
	void Tensor<T, device>::SetDims(const size_t& dims)
	{
		MEASURE();

		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (isAllocated())
			deallocate();
		#endif
		if (dims > Dims())
		{
			AddDims(dims - Dims());
		}
		else if ( dims < Dims() )
		{
			RemoveDims(Dims() - dims);
		}
	}

	template<typename T, Mode device>
	void Tensor<T, device>::AddDims(const size_t& dims)
	{
		MEASURE();

		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (isAllocated())
			deallocate();
		#endif

		m_shape.resize(m_shape.size() + dims, 1);
	}

	template<typename T, Mode device>
	void Tensor<T, device>::RemoveDims(const size_t& dims)
	{
		MEASURE();

		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (isAllocated())
			deallocate();
		#endif

		#ifdef _TS_DEBUG

		if (dims > Dims() - 1)
		{
			throw BadValue("Cannot Remove more dims than the amount of dims in Tensor", dims);
		}

		#endif

		for (unsigned int i = 0; i < dims; i++)
		{
			ResizeDim(Dims() - i - 1, 1);
		}

		m_shape.resize(m_shape.size() - dims);
	}

	template<typename T, Mode device>
	inline TensorSlice<T, device> Tensor<T, device>::Slice(const std::vector<TSlice>& slices)
	{
		MEASURE();
		return TensorSlice<T, device>(this, slices);
	}

	//template<typename T, Mode device>
	//template<typename ... Args>
	//TensorSlice<T, device> Tensor<T, device>::Slice(const std::initializer_list<Args>& ... slices)
	//{
	//	std::vector<TSlice> slice_vec;
	//	slice_vec.reserve(Dims());
	//
	//	to_vector(slice_vec, slices...);
	//
	//	#ifdef _TS_DEBUG
	//
	//	if (slice_vec.size() > Dims())
	//	{
	//		throw BadValue("There cannot be passed more TSlices than dimensions", ExceptValue<size_t>("Got", slice_vec.size()), ExceptValue<size_t>("Expected", size()));
	//	}
	//	#endif
	//
	//	return TensorSlice<T, device>(this, slice_vec);
	//}

	/// Element access functions

	#ifdef _CUDA
	template<typename T, Mode device>
	Tensor<T, device>::operator T* ()
	{
		MEASURE();
		return gpu_mem;
	}
	#endif

	template<typename T, Mode device>
	template<typename ... Args>
	inline T& Tensor<T, device>::Get(const Args& ... coords)
	{
		MEASURE();
		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);
		size_t i = 0;

		get_indx(index, i, tmp_multiply, coords...);

		return m_vector.at(index);
	}

	template<typename T, Mode device>
	inline T& Tensor<T, device>::At(size_t indx)
	{
		return m_vector[indx];
	}

	template<typename T, Mode device>
	inline T Tensor<T, device>::At(size_t indx) const
	{
		return m_vector[indx];
	}

	template<typename T, Mode device>
	inline const T* Tensor<T, device>::Data() const
	{
		return m_vector.data();
	}

	template<typename T, Mode device>
	inline T* Tensor<T, device>::Data()
	{
		return m_vector.data();
	}

	/// Tensor info / get functions

	template<typename T, Mode device>
	inline size_t Tensor<T, device>::Dims() const
	{
		return m_shape.size();
	}

	template<typename T, Mode device>
	inline const std::vector<size_t>& Tensor<T, device>::Shape() const
	{
		return m_shape;
	}

	template<typename T, Mode device>
	inline size_t Tensor<T, device>::size() const
	{
		return m_vector.size();
	}

	/// Tensor iterator funtions

	template<typename T, Mode device>
	inline typename std::vector<T>::const_iterator Tensor<T, device>::begin() const
	{
		return m_vector.begin();
	}

	template<typename T, Mode device>
	inline typename std::vector<T>::iterator Tensor<T, device>::begin()
	{
		return m_vector.begin();
	}

	template<typename T, Mode device>
	inline typename std::vector<T>::const_iterator Tensor<T, device>::end() const
	{
		return m_vector.end();
	}

	template<typename T, Mode device>
	inline typename std::vector<T>::iterator Tensor<T, device>::end()
	{
		return m_vector.end();
	}

	template<typename T, Mode device>
	inline const std::vector<T>& Tensor<T, device>::asVector() const
	{
		return m_vector;
	}

	/// Tensor access operators

	template<typename T, Mode device>
	template<typename ... Args>
	inline T& Tensor<T, device>::operator()(Args ... coords)
	{
		MEASURE();
		return Get(coords...);
	}

	template<typename T, Mode device>
	inline T& Tensor<T, device>::operator[](size_t indx)
	{
		return At(indx);
	}

	template<typename T, Mode device>
	inline T Tensor<T, device>::operator[](size_t indx) const
	{
		return At(indx);
	}

	template<typename T, Mode device>
	Tensor<T>& Tensor<T, device>::operator=(const std::vector<T>& other)
	{
		Fill(other);
		return *this;
	}

	#if defined(_CUDA) && defined(_AMP)
	#pragma message("warning: Cannot use both cuda and amp at the same time defaulting to cuda")
	#endif

	/// Tensor math functions

	template<typename T, Mode device>
	template<typename RT>
	RT Tensor<T, device>::sum()
	{
		MEASURE();
		RT result = RT();
		for (size_t i = 0; i < size(); i++)
			result += (RT)At(i);
		return result;
	}

	/*
	 * CPU operators
	 */

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::add(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG

		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) + other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::add(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) + other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::addSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < size(); i++)
		{
			result[i] = At(i) + other;
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::subtract(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) - other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::subtract(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) - other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::subtractSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) - other;
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::multiply(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) * other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::multiply(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) * other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::multiplySingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) * other;
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::divide(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) - other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::divide(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) / other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::divideSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) - other;
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::modulous(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) % other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::modulous(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) % other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::modulousSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = this->At(i) % other;
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::additionAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) += other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::additionAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) += other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::additionAssignmentSingle(const OT& other)
	{
		MEASURE();
		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) += other;
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::subtractionAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) -= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::subtractionAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) -= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::subtractionAssignmentSingle(const OT& other)
	{
		MEASURE();
		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) -= other;
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::multiplicationAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) *= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::multiplicationAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) *= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::multiplicationAssignmentSingle(const OT& other)
	{
		MEASURE();
		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) *= other;
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::divisionAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) /= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::divisionAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) /= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::divisionAssignmentSingle(const OT& other)
	{
		MEASURE();
		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) /= other;
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::modulouAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) %= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::modulouAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) %= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::modulouAssignmentSingle(const OT& other)
	{
		MEASURE();
		for (size_t i = 0; i < this->size(); i++)
		{
			this->At(i) %= other;
		}
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> TSlib::Tensor<T, device>::compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other[i]);
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> TSlib::Tensor<T, device>::compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other[i]);
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> TSlib::Tensor<T, device>::compareSingle(const OT& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();

		Tensor<RT, device> result(this->Shape());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other);
		}

		return result;
	}
}
