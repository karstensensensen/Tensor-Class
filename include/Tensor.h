#pragma once

#ifdef __APPLE__
typedef double double_t;
#ifdef DEBUG
#define _DEBUG
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
#include <sstream>

#endif


#include "TensorOperators.h"
#include "TensorSlice.h"
#include <cstdarg>
#include <numeric>
#include <tuple>

namespace TSlib
{
/// <summary>
/// Tensor private functions
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="index"></param>
/// <returns></returns>
template<typename T, Mode device>
size_t Tensor<T, device>::get_real_size(const size_t& index) const
{
	MEASURE();
	
	size_t r_size = 1;

	for (size_t i = 0; i <= index; i++)
	{
		r_size *= m_dim_size[i];
	}



	return r_size;
}

template<typename T, Mode device>
size_t Tensor<T, device>::get_dim_length(const size_t& index) const
{
	MEASURE();
	
	size_t r_size = 1;

	for (size_t i = index; i < m_dim_size.size(); i++)
	{
		r_size *= m_dim_size[i];
	}

	return r_size;
}

template<typename T, Mode device>
std::vector<size_t> Tensor<T, device>::FlattenDims(size_t dims) const
{
	MEASURE();
	
	std::vector<size_t> new_dim(dims, 0);

	size_t i;
	for (i = 0; i < std::min(dims, Dims()); i++)
	{
		new_dim[i] = m_dim_size[i];
	}
	i--;

	for (size_t j = i+1; j < m_dim_size.size(); j++)
	{
		new_dim[i] *= m_dim_size[j];
	}

	return new_dim;
}

template<typename T, Mode device>
size_t Tensor<T, device>::FlattenDims() const
{
	MEASURE();
	
	size_t new_dim = m_dim_size[0];

	for (size_t j = 1; j < m_dim_size.size(); j++)
	{
		new_dim *= m_dim_size[j];
	}

	return new_dim;
}


template<typename T, Mode device>
size_t  Tensor<T, device>::get_dim_size(const size_t& index) const
{
	
	MEASURE();
	
	size_t size = 1;
	for (size_t i = 0; i <= index; i++)
	{
		size *= m_dim_size[i];
	}

	return size;
}

template<typename T, Mode device>
size_t Tensor<T, device>::get_dim_offset(const size_t& index) const
{
	MEASURE();
	size_t result = 0;
	for (size_t i = 0; i < index; i++)
	{
		result += m_dim_size[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename First>
void Tensor<T, device>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord)
{
	MEASURE();

	#ifdef _DEBUG
	if (m_dim_size[iter] <= coord)
		throw OutOfBounds(*this, "Exception was thrown, because an element outside the Tensor bounds was accsessed", iter, coord);
	#endif

	indx += coord * tmp_multiply;
	tmp_multiply *= m_dim_size[iter];
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

	#ifdef _DEBUG
	if(std::is_integral<First>::value)
	{
		#ifdef __APPLE__
		throw BadType("Integral", typeid(First).name());
		#else
		throw BadType::BadType("Integral", typeid(First).name());
		#endif
	}
	#endif

	to_vector(vec, first);
	to_vector(vec, args...);
}

/// <summary>
/// This is a helper struct that acts as the sort function for the std::sort function in Tensor::based_sort function.
/// </summary>
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
}

template<typename T, Mode device>
std::vector<size_t> Tensor<T, device>::based_sort(const std::vector<size_t>& target)
{
	std::vector<size_t> new_indexes(target.size());
	std::iota(new_indexes.begin(), new_indexes.end(), 0);

	std::sort(new_indexes.begin(), new_indexes.end(), sorter(target));

	return new_indexes;
}


/// <summary>
/// Tensor constructors
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="dims"></param>
template<typename T, Mode device>
Tensor<T, device>::Tensor(const int& dims)
	: m_dim_size(dims + 1)
{
	MEASURE();
	m_dim_size[dims] = 1;
}

//template<typename T, Mode device>
//Tensor<T, device>::Tensor(const std::initializer_list<size_t>& sizes, const T& pad_val, const bool& add_extra_dim)
//	: m_dim_size(sizes.size() + 1 * add_extra_dim)
//{
//	MEASURE();
//
//	if(add_extra_dim)
//		m_dim_size[sizes.size()] = 1;
//	MultiResize(sizes, pad_val);
//}

template<typename T, Mode device>
Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, const T& pad_val, const bool& add_extra_dim)
	: m_dim_size(sizes.size() + 1 * add_extra_dim)
{
	MEASURE();
	if(add_extra_dim)
		m_dim_size[sizes.size()] = 1;
	MultiResize(sizes, pad_val);
}

template<typename T, Mode device>
Tensor<T, device>::Tensor(const std::vector<TSlice>& sizes, const T& pad_val, const bool& add_extra_dim)
	: m_dim_size(sizes.size() + 1 * add_extra_dim)
{
	MEASURE();
	if (add_extra_dim)
		m_dim_size[sizes.size()] = 1;
	MultiResize(sizes, pad_val);
}

template<typename T, Mode device>
Tensor<T, device>::Tensor(const TensorSlice<T, device>& slice, const T& pad_val, const bool& add_extra_dim)
	: m_dim_size(slice.DimSizes().size() + 1 * add_extra_dim)
{
	MEASURE();
	if (add_extra_dim)
		m_dim_size[slice.DimSizes().size()] = 1;
	MultiResize(slice.DimSizes(), pad_val);

	for (size_t i = 0; i < slice.size(); i++)
	{
		At(i) = slice[i];
	}
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

template<typename T, Mode device>
TSlib::Tensor<T, device>::Tensor(const Tensor<T, device>& other)
	: m_dim_size(other.DimSizes().size()), m_vector(other.m_vector)
{
	MEASURE();

	#ifdef _CUDA
	if (other.isAllocated())
	{
		allocate();
		CER(cudaMemcpy(gpu_mem, other.getGPU(), sizeof(T) * size(), cudaMemcpyDeviceToDevice));
	}
	#endif
}

/// <summary>
//// Tensor public functions
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="val"></param>

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
	for (size_t i = 0; i < get_dim_size(dim); i++)
	{
		At(i + get_dim_offset(dim) * index) = val;
	}
}

/// <summary>
/// resize functions
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="index"></param>
/// <param name="row_size"></param>
/// <param name="amount"></param>
/// <param name="pad_val"></param>

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
void Tensor<T, device>::ResizeDim(const size_t& index, const size_t& amount, const T& pad_val)
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
	size_t tmp_row_size = get_real_size(index);


	m_dim_size[index] = amount;

	if (index != 0)
		new_amount = get_real_size(index - 1) * get_dim_length(index);
	else

		new_amount = get_dim_length(index);

	if (new_amount > tmp_size)
	{
		m_vector.reserve(new_amount);
		new_amount -= tmp_size;

		//Resize dimention
		upscale_dim(index, tmp_row_size, new_amount, pad_val);
	}
	else
	{
		new_amount = tmp_size - new_amount;
		downscale_dim(index, tmp_row_size, new_amount);
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
		new_size *= size * (size >= m_dim_size[index]) +
					m_dim_size[index] * (size < m_dim_size[index]);
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
inline size_t Tensor<T, device>::calc_new_size(const std::vector<TSlice>& sizes)
{
	MEASURE();
	size_t new_size = 1;
	for (const TSlice& elem_size : sizes)
	{
		new_size *= elem_size.width();
	}
	return new_size;
}



#if 0
template<typename T, Mode device>
void Tensor<T, device>::MultiResize(const std::initializer_list<size_t>& sizes, const T& pad_val)
{
	MEASURE();
	#ifdef _CUDA
	//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
	if (isAllocated())
		deallocate();
	#endif

	#ifdef _DEBUG
	assert(sizes.size() != 0);
	#endif

	size_t current_size = get_real_size(m_dim_size.size() - 1);
	size_t new_size = calc_new_size(sizes);

	if(current_size < new_size)
		m_vector.reserve(new_size - current_size);
	size_t index = 0;
	for (const size_t& size : sizes)
	{
		size_t new_amount = size;
		size_t tmp_size = m_vector.size();
		size_t tmp_row_size = get_real_size(index);

		m_dim_size[index] = size;

		if (index != 0)
			new_amount = get_real_size(index - 1) * get_dim_length(index);
		else
			new_amount = get_dim_length(index);

		if (new_amount > tmp_size)
		{
			new_amount -= tmp_size;

			//Resize dimention
			upscale_dim(index, tmp_row_size, new_amount, pad_val);
		}
		else
		{
			new_amount = tmp_size - new_amount;

			downscale_dim(index, tmp_row_size, new_amount);
		}

		index++;
	}
}
#endif

template<typename T, Mode device>
void Tensor<T, device>::MultiResize(const std::vector<size_t>& sizes, const T& pad_val)
{
	MEASURE();
	#ifdef _CUDA
	//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
	if (isAllocated())
		deallocate();
	#endif

	#ifdef _DEBUG
	assert(sizes.size() != 0);
	#endif

	size_t current_size = get_real_size(m_dim_size.size() - 1);
	size_t new_size = calc_new_size(sizes);

	if (current_size < new_size)
		m_vector.reserve(new_size - current_size);
	size_t iteration = 0;
	std::vector<size_t> indexes = based_sort(sizes);


	for (size_t i = 0; i < sizes.size(); i++)
	{
		size_t new_amount = sizes[indexes[iteration]];
		size_t tmp_size = m_vector.size();
		size_t tmp_row_size = get_real_size(indexes[iteration]);

		m_dim_size[indexes[iteration]] = sizes[indexes[iteration]];

		if (indexes[iteration] != 0)
			new_amount = get_real_size(indexes[iteration] - 1) * get_dim_length(indexes[iteration]);
		else
			new_amount = get_dim_length(indexes[iteration]);

		if (new_amount > tmp_size)
		{
			new_amount -= tmp_size;

			//Resize dimention
			upscale_dim(indexes[iteration], tmp_row_size, new_amount, pad_val);
		}
		else
		{
			new_amount = tmp_size - new_amount;

			downscale_dim(indexes[iteration], tmp_row_size, new_amount);
		}

		iteration++;
	}
	
	m_vector.shrink_to_fit();
}

template<typename T, Mode device>
void Tensor<T, device>::MultiResize(const std::vector<TSlice>& sizes, const T& pad_val)
{
	MEASURE();
	#ifdef _CUDA
	//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
	if (isAllocated())
		deallocate();
	#endif

	#ifdef _DEBUG
	assert(sizes.size() != 0);
	#endif

	size_t current_size = get_real_size(m_dim_size.size() - 1);
	size_t new_size = calc_new_size(sizes);

	if (current_size < new_size)
		m_vector.reserve(new_size - current_size);
	size_t iteration = 0;
	std::vector<size_t> indexes = based_sort(sizes);


	for (size_t i = 0; i < sizes.size(); i++)
	{
		size_t new_amount = sizes[indexes[iteration]].width();
		size_t tmp_size = m_vector.size();
		size_t tmp_row_size = get_real_size(indexes[iteration]);

		m_dim_size[indexes[iteration]] = sizes[indexes[iteration]].width();

		if (indexes[iteration] != 0)
			new_amount = get_real_size(indexes[iteration] - 1) * get_dim_length(indexes[iteration]);
		else
			new_amount = get_dim_length(indexes[iteration]);

		if (new_amount > tmp_size)
		{
			new_amount -= tmp_size;

			//Resize dimention
			upscale_dim(indexes[iteration], tmp_row_size, new_amount, pad_val);
		}
		else
		{
			new_amount = tmp_size - new_amount;

			downscale_dim(indexes[iteration], tmp_row_size, new_amount);
		}

		iteration++;
	}

	m_vector.shrink_to_fit();
}


template<typename T, Mode device>
void Tensor<T, device>::Reshape(const std::initializer_list<size_t>& shape, bool add_extra_dim)
{
	MEASURE();
	#ifdef _DEBUG
	size_t new_shape_size = 1;
	for (const size_t& elem : shape)
	{
		new_shape_size *= elem;
	}
	
	if (size() != new_shape_size)
		throw BadShape(*this, shape);
	#endif

	m_dim_size.erase(m_dim_size.begin(), m_dim_size.end());

	for (const size_t& elem : shape)
	{
		m_dim_size.push_back(elem);
	}

	if (add_extra_dim)
		m_dim_size.push_back(1);
}

template<typename T, Mode device>
void Tensor<T, device>::Reshape(const std::vector<size_t>& shape, bool add_extra_dim)
{
	MEASURE();
	#ifdef _DEBUG
	size_t new_shape_size = 1;
	for (const size_t& elem : shape)
	{
		new_shape_size *= elem;
	}

	if (size() != new_shape_size)
	{
		throw BadShape(*this, shape);
	}
	#endif

	m_dim_size.erase(m_dim_size.begin(), m_dim_size.end());

	for (const size_t& elem : shape)
	{
		m_dim_size.push_back(elem);
	}

	if (add_extra_dim)
		m_dim_size.push_back(1);
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

	m_dim_size.resize(m_dim_size.size() + dims, 1);
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
	
	#ifdef _DEBUG
	
	assert("Cannot Remove more dims than the amount of dims in TensorBase" && dims < m_dim_size.size() - 1);
	#endif

	m_dim_size.resize(m_dim_size.size() - dims);
	m_dim_size[m_dim_size.size() - 1] = 1;
}


template<typename T, Mode device>
inline TensorSlice<T, device> Tensor<T, device>::Slice(const std::vector<TSlice>& slices)
{
	MEASURE();
	return TensorSlice<T, device>(this, slices);
}

template<typename T, Mode device>
template<typename ... Args>
TensorSlice<T, device> Tensor<T, device>::Slice(const std::initializer_list<Args>& ... slices)
{
	std::vector<TSlice> slice_vec;
	slice_vec.reserve(Dims());

	to_vector(slice_vec, slices...);

	#ifdef _DEBUG
	assert((slice_vec.size() <= Dims(), "There cannot be passed more TSlices than dimensions"));
	#endif

	return TensorSlice<T, device>(this, slice_vec);
}


/// <summary>
/// Element access functions
/// </summary>
/// <typeparam name="T"></typeparam>
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
	size_t tmp_multiply = 1;
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

/// <summary>
/// Tensor info / get functions
/// </summary>
/// <typeparam name="T"></typeparam>
/// <returns></returns>

template<typename T, Mode device>
inline size_t Tensor<T, device>::Dims() const
{
	return m_dim_size.size();
}

template<typename T, Mode device>
inline const std::vector<size_t>& Tensor<T, device>::DimSizes() const
{
	return m_dim_size;
}

template<typename T, Mode device>
inline size_t Tensor<T, device>::size() const
{
	return m_vector.size();
}

/// <summary>
/// Tensor iterator funtions
/// </summary>
/// <typeparam name="T"></typeparam>
/// <returns></returns>

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

/// <summary>
/// Tensor access operators
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="...coords"></param>
/// <returns></returns>

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



#if defined(_CUDA) && defined(_AMP)
	#pragma message("warning: Cannot use both cuda and amp at the same time defaulting to cuda")
#endif

/// <summary>
/// Tensor math functions
/// </summary>
/// <typeparam name="T"></typeparam>
/// <returns></returns>
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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims()-1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	Tensor<RT, device> result(this->DimSizes(), RT(), false);

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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
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
	#ifdef _DEBUG
	assert("Slice must have less or the same amount of dimensions as target tensor" && Dims() >= other.Dims());
	for (size_t i = 0; i < Dims() - 1; i++)
	{
		assert((DimSizes()[i] == other.DimSizes()[i].get_to(), "Slice dimension must be less or the same length of the target dimension"));
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
Tensor<RT, device> Tensor<T, device>::compare(const Tensor<OT, o_device>& other) const
{
	MEASURE();
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) == other[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT, Mode o_device>
Tensor<RT, device> Tensor<T, device>::compare(const TensorSlice<OT, o_device>& other) const
{
	MEASURE();
	#ifdef _DEBUG
	for (size_t i = 0; i < other.Dims(); i++)
	{
		assert("Must have less than or equal dimension length in each Tensor and TensorSlice" && DimSizes()[i] >= other.DimSizes()[i].get_to());
	}
	#endif

	Tensor<RT, device> result(DimSizes(), false, false);

	for (size_t i = 0; i < other.size(); i++)
	{
		result[other.map_index(i)] = At(other.map_index(i)) == other[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT>
Tensor<RT, device> Tensor<T, device>::compareSingle(const OT& other) const
{
	MEASURE();
	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) == other;
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT, Mode o_device>
Tensor<RT, device> Tensor<T, device>::lessThan(const Tensor<OT, o_device>& other) const
{
	MEASURE();
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) < other[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT>
Tensor<RT, device> Tensor<T, device>::lessThanSingle(const OT& other) const
{
	MEASURE();
	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) < other;
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT, Mode o_device>
Tensor<RT, device> Tensor<T, device>::moreThan(const Tensor<OT, o_device>& other) const
{
	MEASURE();
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) > other[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT>
Tensor<RT, device> Tensor<T, device>::moreThanSingle(const OT& other) const
{
	MEASURE();
	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) > other;
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT, Mode o_device>
Tensor<RT, device> Tensor<T, device>::lessThanEqual(const Tensor<OT, o_device>& other) const
{
	MEASURE();
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) <= other[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT>
Tensor<RT, device> Tensor<T, device>::lessThanEqualSingle(const OT& other) const
{
	MEASURE();
	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) <= other;
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT, Mode o_device>
Tensor<RT, device> Tensor<T, device>::moreThanEqual(const Tensor<OT, o_device>& other) const
{
	MEASURE();
	#ifdef _DEBUG
	assert("Must have the same number of dimensions in each TensorBase" && this->Dims() == other.Dims());
	for (size_t i = 0; i < this->Dims(); i++)
	{
		assert("Must have same dimension length in each TensorBase" && this->DimSizes()[i] == other.DimSizes()[i]);
	}
	#endif

	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) >= other[i];
	}

	return result;
}

template<typename T, Mode device>
template<typename RT, typename OT>
Tensor<RT, device> Tensor<T, device>::moreThanEqualSingle(const OT& other) const
{
	MEASURE();
	Tensor<RT, device> result(this->DimSizes());

	for (size_t i = 0; i < this->size(); i++)
	{
		result[i] = this->At(i) >= other;
	}

	return result;
}
}

#ifdef __APPLE__
#ifdef DEBUG
#undef _DEBUG
#endif
#endif
