#pragma once

#ifdef __clang__
typedef double double;
#endif

#include <stdlib.h>

#ifdef TENSOR_PROFILING
#include <Profiler.h>
#else
#define MEASURE()
#endif

#include "TensorExceptions.h"

#ifdef _CUDA

#include "TensorCuda.cuh"

#else

#include "Tensor.h"

#endif

#include <sstream>

#include "TensorArithmetic.h"
#include "TensorSlice.h"
#include "TensorTools.h"
#include <filesystem>
#include <functional>
#include <algorithm>
#include <math.h>
#include <cstdarg>
#include <numeric>
#include <fstream>
#include <tuple>

namespace TSlib
{
	/// Tensor private functions

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
	std::vector<size_t> Tensor<T, device>::FlattenDims(size_t dims) const
	{
		MEASURE();

		std::vector<size_t> new_dim(dims, 1);

		size_t i;
		for (i = 0; i < std::min(dims, Dims()); i++)
		{
			new_dim[dims - i - 1] = m_shape[Dims() - i - 1];
		}

		for (size_t j = dims; j < Dims(); j++)
		{
			new_dim[0] *= m_shape[Dims() - j - 1];
		}

		return new_dim;
	}

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
	template<typename First, typename... Args>
	void Tensor<T, device>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining)
	{
		MEASURE();

		get_indx(indx, iter, tmp_multiply, coord);
		get_indx(indx, iter, tmp_multiply, remaining...);
	}

	template<typename T, Device device>
	template<typename First>
	void Tensor<T, device>::to_vector(std::vector<TSlice>& vec, const std::initializer_list<First>& first)
	{
		vec.push_back(TSlice(first));
	}

	template<typename T, Device device>
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

	template<typename T, Device device>
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

		template<size_t n>
		struct staticSorter
		{
			const std::array<size_t, n>& vec;

			staticSorter(const std::array<size_t, n>& vec)
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

	template<typename T, Device device>
	std::vector<size_t> Tensor<T, device>::based_sort(const std::vector<size_t>& target)
	{
		std::vector<size_t> new_indexes(target.size());
		
		std::generate(new_indexes.begin(), new_indexes.end(), [n = target.size() - 1]() mutable {return n--; });

		std::sort(new_indexes.begin(), new_indexes.end(), sorter(target));

		return new_indexes;
	}

	template<typename T, Device device>
	template<size_t n>
	std::array<size_t, n> Tensor<T, device>::based_sort(const std::array<size_t, n>& target)
	{
		std::array<size_t, n> new_indexes;
		
		std::generate(new_indexes.begin(), new_indexes.end(), [i = n - 1]() mutable {return i--; });

		std::sort(new_indexes.begin(), new_indexes.end(), staticSorter<n>(target));

		return new_indexes;
	}

	template<typename T, Device device>
	std::vector<size_t> Tensor<T, device>::based_sort(const std::vector<TSlice>& target)
	{
		std::vector<size_t> new_indexes(target.size());
		std::iota(new_indexes.begin(), new_indexes.end(), 0);

		std::sort(new_indexes.begin(), new_indexes.end(), sorterSlice(target));

		return new_indexes;
	}

	template<typename T, Device device>
	inline Tensor<T, device>::Tensor()
		: m_shape(1)
	{
		Resize({ 0 });
	}

	template<typename T, Device device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, const T& pad_val)
		: m_shape(sizes.size())
	{
		MEASURE();
		Resize(sizes, pad_val);
	}

	//generator functions take an index / coordinate or nothing as parameters and returns a value with the containter type

	template<typename T, Device device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, std::function<T()> generator)

		: m_shape(sizes.size())
	{
		Resize(sizes);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) = generator();
		}
	}

	template<typename T, Device device>
	Tensor<T, device>::Tensor(const std::vector<size_t>& sizes, std::function<T(const size_t&)> generator)

		: m_shape(sizes.size())
	{
		Resize(sizes);

		for (size_t i = 0; i < size(); i++)
		{
			At(i) = generator({ i });
		}
	}

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
	Tensor<T, device>::Tensor(const Tensor<T, device>& other)
		: m_vector(other.asVector()), m_shape(other.Shape())
	{
		MEASURE();

		#ifdef _CUDA

		if (other.IsAllocated())
		{
			Allocate();
			cudaMemcpy(gpu_mem, other.gpu_mem, sizeof(T) * size(), cudaMemcpyDeviceToDevice);
		}

		#endif
	}

	template<typename T, Device device>
	Tensor<T, device>::Tensor(Tensor<T, device>&& other)
		: m_vector(std::move(other.m_vector)), m_shape(std::move(other.m_shape))
	{
		#ifdef _CUDA
		gpu_mem = other.gpu_mem;
		allocated = other.allocated;
		m_threads = other.m_threads;

		other.gpu_mem = nullptr;
		#endif
	}


	template<typename T, Device device>
	inline void Tensor<T, device>::Save(std::string dir) const
	{
		// create directories
		dir += ".tnsr";
		std::filesystem::path path(dir);
		std::filesystem::create_directories(path.parent_path());

		std::ofstream out_file(path, std::ios::binary);

		size_t dims = Dims();

		out_file.write((char*)&dims, sizeof(dims));

		out_file.write((const char*)Shape().data(), sizeof(size_t) * dims);

		size_t data_size = sizeof(T);

		out_file.write((char*)&data_size, sizeof(data_size));

		out_file.write((char*)Data(), sizeof(T) * size());

		out_file.close();
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::Load(std::string dir)
	{
		dir += ".tnsr";
		std::filesystem::path path(dir);

		std::ifstream in_file(path, std::ios::binary);

		size_t dims;
		in_file.read((char*)&dims, sizeof(dims));

		std::vector<size_t> loaded_shape(dims);
		in_file.read((char*)loaded_shape.data(), sizeof(size_t) * dims);
		Resize(loaded_shape);

		#ifdef _TS_DEBUG

		size_t data_size;
		in_file.read((char*)&data_size, sizeof(data_size));

		if (data_size != sizeof(T))
		{
			throw BadValue("The data size stored in the file is not the same as the Tensor data type", ExceptValue("Tensor data size", sizeof(T)), ExceptValue("File data size", data_size), ExceptValue("File", dir));
		}

		#else
		in_file.ignore(sizeof(size_t));
		#endif

		in_file.read((char*)Data(), sizeof(T) * size());

		in_file.close();

		return *this;
	}

	template<typename T, Device device>
	Tensor<T, device>::~Tensor()
	{
		#ifdef _CUDA
		if (IsAllocated())
		{
			Deallocate();
		}
		#endif
	}

	/// Tensor public functions

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::Fill(const T& val)
	{
		MEASURE();
		Compute([val](T& elem) {elem = val; });

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Fill(std::function<T(const size_t&)> generator)
	{
		MEASURE();

		Compute([generator](T& elem, const size_t& index) {elem = generator(index); });

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Fill(std::function<T(const std::vector<size_t>&)> generator)
	{
		MEASURE();
		Compute([generator](T& elem, const std::vector<size_t>& dimensions) {elem = generator(dimensions); });

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator)
	{
		MEASURE();

		Compute([generator](T& elem, const std::vector<size_t>& dimensions, const size_t& index) {elem = generator(dimensions, index); });

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Fill(const std::vector<T>& vals)
	{
		#ifdef _TS_DEBUG
		if (vals.size() != size())
		{
			throw BadShape(this, "Vector must have the same size as the target tensor", std::vector<size_t>{ vals.size() });
		}
		#endif

		memcpy(Data(), vals.data(), size() * sizeof(T));

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Compute(std::function<void(T&)> compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(At(index));
		}

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Compute(std::function<void(T&, const size_t&) > compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(At(index), index);
		}

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Compute(std::function<void(T&, const std::vector<size_t>&)> compute_func)
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

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device>& Tensor<T, device>::Compute(std::function<void(T&, const std::vector<size_t>&, const size_t&)> compute_func)
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

		return *this;
	}

	template<typename T, Device device>
	void Tensor<T, device>::Compute(std::function<void(const T&)> compute_func) const
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(At(index));
		}
	}

	template<typename T, Device device>
	void Tensor<T, device>::Compute(std::function<void(const T&, const size_t&)> compute_func) const
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(At(index), index);
		}
	}

	template<typename T, Device device>
	void Tensor<T, device>::Compute(std::function<void(const T&, const std::vector<size_t>&)> compute_func) const
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
	void Tensor<T, device>::Compute(std::function<void(const T&, const std::vector<size_t>&, const size_t&)> compute_func) const
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

		return *this;
	}

	template<typename T, Device device>
	inline Tensor<T, device> Tensor<T, device>::Compute(std::function<void(T&, const T&) > compute_func, size_t axis, T pad_val, bool keepDims) const
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
	inline Tensor<T, device> Tensor<T, device>::Compute(std::function<void(T&, const T&, const size_t&)> compute_func, size_t axis, T pad_val, bool keepDims) const
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
	inline Tensor<T, device> Tensor<T, device>::Compute(std::function<void(T&, const T&, const std::vector<size_t>&)> compute_func, size_t axis, T pad_val, bool keepDims) const
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
					compute_func(elem, Get(new_coords), new_coords);
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
	inline Tensor<T, device> Tensor<T, device>::Compute(std::function<void(T&, const T&, const std::vector<size_t>&, const size_t&)> compute_func, size_t axis, T pad_val, bool keepDims) const
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
	inline Tensor<T, device>& Tensor<T, device>::Replace(const T& target, const T& value)
	{
		for (size_t i = 0; i < size(); i++)
		{
			if (target == At(i))
			{
				At(i) = value;
			}
		}

		return *this;
	}

	/// resize functions

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::ResizeDim(const size_t& dim, const size_t& amount, const T& pad_val)
	{
		MEASURE();
		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (IsAllocated())
			Deallocate();
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

		return *this;
	}

	template<typename T, Device device>
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

	template<typename T, Device device>
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

	template<typename T, Device device>
	template<size_t n>
	inline size_t Tensor<T, device>::calc_new_size(const std::array<size_t, n>& sizes)
	{
		MEASURE();
		size_t new_size = 1;
		for (const size_t& elem_size : sizes)
		{
			new_size *= elem_size;
		}
		return new_size;
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::Resize(const std::vector<size_t>& sizes, const T& pad_val)
	{
		MEASURE();
		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (IsAllocated())
			Deallocate();
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

		return *this;
	}

	template<typename T, Device device>
	template<size_t n>
	inline Tensor<T, device>& Tensor<T, device>::Resize(const std::array<size_t, n>& sizes, const T& pad_val)
	{
		MEASURE();
		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (IsAllocated())
			Deallocate();
		#endif

		#ifdef _TS_DEBUG
		if (n == 0)
		{
			throw BadShape("New shape must not be of length 0");
		}
		#endif

		SetDims(n);

		size_t current_size = get_real_size(m_shape.size() - 1);
		size_t new_size = calc_new_size<n>(sizes);

		if (current_size < new_size)
			m_vector.reserve(new_size - current_size);

		std::array<size_t, n> dimensions = based_sort(sizes);

		for (size_t i = 0; i < n; i++)
		{
			size_t target_size = n - dimensions[i] - 1;
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

		return *this;
	}

	template<typename T, Device device>
	template<typename Ts, std::enable_if_t<std::is_integral<Ts>::value, int>>
	Tensor<T, device>& Tensor<T, device>::Reshape(const std::vector<Ts>& shape)
	{
		MEASURE();
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
		if constexpr (std::is_unsigned<T>::value)
		{
			for (size_t i = 0; i < shape.size(); i++)
			{
				unknown_pos = i * (shape[i] < 0) + unknown_pos * (shape[i] >= 0);
				shape_product *= shape[i] * (shape[i] >= 0) + (shape[i] < 0);
			}
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
	inline TensorSlice<T, device> Tensor<T, device>::AsShape(const std::vector<long long>& shape)
	{
		TensorSlice<T, device> slice = Slice();
		slice.Reshape(shape);

		return slice;
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::SetDims(const size_t& dims)
	{
		MEASURE();

		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (IsAllocated())
			Deallocate();
		#endif
		if (dims > Dims())
		{
			AddDims(dims - Dims());
		}
		else if (dims < Dims())
		{
			RemoveDims(Dims() - dims);
		}

		return *this;
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::AddDims(const size_t& dims)
	{
		MEASURE();

		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (IsAllocated())
			Deallocate();
		#endif

		m_shape.resize(m_shape.size() + dims, 1);

		return *this;
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::RemoveDims(const size_t& dims)
	{
		MEASURE();

		#ifdef _CUDA
		//deallocate gpu memory if allocated to make sure there isnt accidentally copied too much or little memory to cpu or gpu
		if (IsAllocated())
			Deallocate();
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

		return *this;
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline Tensor<T, device>& Tensor<T, device>::Append(const Tensor<OT, o_device>& other, const size_t& dimension)
	{
		#ifdef _TS_DEBUG

		if (dimension >= Dims())
		{
			throw BadValue("The target dimension must be less than or equal to the total dimensions in the target tensors", { "Target dimension", dimension }, { "Tensor dimensions", Dims() });
		}
		else if (other.Dims() != Dims())
		{
			throw BadShape("The source Tensor must have the same amount of dimensions as the destination Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i] && i != dimension)
			{
				throw BadShape("The source Tensor must match the destination Tensors Shape appart from the dimensions that is getting appended to", other.Shape(), Shape());
			}
		}
		#endif

		std::vector<size_t> new_shape = Shape();

		new_shape[dimension] += other.Shape()[dimension];

		Resize(new_shape);

		std::vector<TSlice> append_slice(Dims(), All);

		append_slice[dimension] = TSlice(Shape()[dimension] - other.Shape()[dimension], -1);

		Slice(append_slice).Fill(other);

		return *this;
	}

	template<typename T, Device device>
	inline TensorSlice<T, device> Tensor<T, device>::Slice(const std::vector<TSlice>& slices)
	{
		MEASURE();

		TensorSlice<T, device> slice(this, slices);

		#ifdef _TS_DEBUG
		for (size_t i = 0; i < Dims(); i++)
		{
			if (slice.TSliceShape()[i].get_from() + slice.Shape()[i] > Shape()[i])
			{
				throw BadShape("Slice must be within Tensor borders", slice.Shape(), Shape());
			}
		}
		#endif

		return slice;
	}

	//template<typename T, Device device>
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
	template<typename T, Device device>
	Tensor<T, device>::operator T* ()
	{
		MEASURE();
		return gpu_mem;
	}

	template<typename T, Device device>
	Tensor<T, device>::operator const T* () const
	{
		MEASURE();
		return gpu_mem;
	}
	#endif

	template<typename T, Device device>
	inline T& Tensor<T, device>::Get(const std::vector<size_t>& coords)
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
	T& Tensor<T, device>::Get(const Args& ... coords)
	{
		MEASURE();

		#ifdef _TS_DEBUG
		if (Dims() != sizeof...(coords))
		{
			throw BadValue("Exception was thrown, because there were not the same nuumber of coordinates given as the number of dimensions in the Tensor", ExceptValue("Coords", sizeof...(coords)), ExceptValue("Dimensions", Dims()));
		}
		#endif

		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);
		size_t i = 0;

		get_indx(index, i, tmp_multiply, coords...);

		return m_vector.at(index);
	}

	template<typename T, Device device>
	T Tensor<T, device>::Get(const std::vector<size_t>& coords) const
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
	T Tensor<T, device>::Get(const Args& ... coords) const
	{
		MEASURE();
		size_t index = 0;
		size_t tmp_multiply = get_real_size(Dims() - 1);
		size_t i = 0;

		get_indx(index, i, tmp_multiply, coords...);

		return m_vector.at(index);
	}

	template<typename T, Device device>
	inline T& Tensor<T, device>::At(size_t indx)
	{
		return m_vector[indx];
	}

	template<typename T, Device device>
	inline T Tensor<T, device>::At(size_t indx) const
	{
		return m_vector[indx];
	}

	template<typename T, Device device>
	inline const T* Tensor<T, device>::Data() const
	{
		return m_vector.data();
	}

	template<typename T, Device device>
	inline T* Tensor<T, device>::Data()
	{
		return m_vector.data();
	}

	/// Tensor info / get functions

	template<typename T, Device device>
	inline size_t Tensor<T, device>::Dims() const
	{
		return m_shape.size();
	}

	template<typename T, Device device>
	inline const std::vector<size_t>& Tensor<T, device>::Shape() const
	{
		return m_shape;
	}

	template<typename T, Device device>
	inline size_t Tensor<T, device>::size() const
	{
		return m_vector.size();
	}

	/// Tensor iterator funtions

	template<typename T, Device device>
	inline typename std::vector<T>::const_iterator Tensor<T, device>::begin() const
	{
		return m_vector.begin();
	}

	template<typename T, Device device>
	inline typename std::vector<T>::iterator Tensor<T, device>::begin()
	{
		return m_vector.begin();
	}

	template<typename T, Device device>
	inline typename std::vector<T>::const_iterator Tensor<T, device>::end() const
	{
		return m_vector.end();
	}

	template<typename T, Device device>
	inline typename std::vector<T>::iterator Tensor<T, device>::end()
	{
		return m_vector.end();
	}

	template<typename T, Device device>
	inline const std::vector<T>& Tensor<T, device>::asVector() const
	{
		return m_vector;
	}

	/// Tensor access operators

	template<typename T, Device device>
	template<typename ... Args>
	T& Tensor<T, device>::operator()(Args ... coords)
	{
		MEASURE();
		return Get(coords...);
	}

	template<typename T, Device device>
	template<typename ... Args>
	T Tensor<T, device>::operator()(Args ... coords) const
	{
		return Get(coords...);
	}

	template<typename T, Device device>
	inline T& Tensor<T, device>::operator[](size_t indx)
	{
		return At(indx);
	}

	template<typename T, Device device>
	inline T Tensor<T, device>::operator[](size_t indx) const
	{
		return At(indx);
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::operator=(const std::vector<T>& other)
	{
		Fill(other);
		return *this;
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::operator=(const Tensor<T, device>& other)
	{
		Fill(other.m_vector);
		return *this;
	}

	template<typename T, Device device>
	Tensor<T, device>& Tensor<T, device>::operator=(Tensor<T, device>&& other)
	{
		m_vector = std::move(other.m_vector);
		m_shape = std::move(other.m_shape);

		#ifdef _CUDA
		if (allocated)
		{
			Deallocate();
		}

		gpu_mem = other.gpu_mem;
		allocated = other.allocated;
		m_threads = other.m_threads;

		other.gpu_mem = nullptr;
		#endif

		return *this;
	}

	#if defined(_CUDA) && defined(_AMP)
	#pragma message("warning: Cannot use both cuda and amp at the same time defaulting to cuda")
	#endif
}

#ifdef _CUDA
#include "TensorCuda.ipp"
#endif
#include "TensorArithmeticOperators.ipp"
#include "TensorMath.ipp"
#include "TensorSlice.ipp"