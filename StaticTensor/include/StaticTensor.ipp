#include <string>

namespace TSlib
{

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	StaticTensor<T, dimensions, var_shape>::StaticTensor(T init_val)
	{
		m_array.fill(init_val);
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<typename Tother, size_t dim_other, const std::array<size_t, dim_other>& shape_other>
	StaticTensor<T, dimensions, var_shape>::StaticTensor(const StaticTensor<Tother, dim_other, shape_other>& other)
	{
		static_assert(size() == other.size(), "A tensor can only be copied, if it has the same size as the source");
		for (size_t i = 0; i < size(); i++)
		{
			at(i) = other[i];
		}
	}


	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	T& StaticTensor<T, dimensions, var_shape>::at(size_t index)
	{
		return m_array[index];
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	T StaticTensor<T, dimensions, var_shape>::at(size_t index) const
	{
		return m_array[index];
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<typename ...Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int>>
	T& StaticTensor<T, dimensions, var_shape>::get(Tcoords ... var_coords)
	{
		std::array<size_t, dimensions> coords = { (size_t)var_coords... };

		size_t index = 0;
		size_t tmp_multiply = get_real_size<dimensions - 1>();

		for (size_t i = 0; i < dimensions; i++)
		{
			#ifdef _TS_DEBUG
			if (var_shape[i] <= coords[i])
				assert(false);
				/*throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", i, coords[i]);*/
			#endif

			tmp_multiply /= var_shape[i];
			index += coords[i] * tmp_multiply;
		}

		return at(index);
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<typename ...Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int>>
	inline T StaticTensor<T, dimensions, var_shape>::get(Tcoords ... var_coords) const
	{
		std::array<size_t, dimensions> coords = { var_coords... };

		size_t index = 0;
		size_t tmp_multiply = get_real_size<dimensions-1>();

		for (size_t i = 0; i < Dims(); i++)
		{
			#ifdef _TS_DEBUG
			if (var_shape[i] <= coords[i])
				assert(false);
				/*throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", i, coords[i]);*/
			#endif

			tmp_multiply /= var_shape[i];
			index += coords[i] * tmp_multiply;
		}

		return at(index);
	}


	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<size_t axis>
	constexpr size_t StaticTensor<T, dimensions, var_shape>::get_real_size()
	{
		size_t r_size = 1;

		for (size_t i = 0; i <= axis; i++)
		{
			r_size *= var_shape[dimensions - i - 1];
		}

		return r_size;
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	size_t StaticTensor<T, dimensions, var_shape>::get_real_size(size_t axis)
	{
		size_t r_size = 1;

		for (size_t i = 0; i <= axis; i++)
		{
			r_size *= var_shape[dimensions - i - 1];
		}

		return r_size;
	}


	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	constexpr const size_t StaticTensor<T, dimensions, var_shape>::size()
	{
		size_t prod = 1;
		for (size_t i = 0; i < dimensions; i++)
		{
			prod *= var_shape[i];
		}

		return prod;
	}
	
	template<typename T, size_t axis, size_t n, const std::array<T, n>& arr>
	static constexpr std::array<T, n - 1> flatten()
	{
		std::array<size_t, dimensions - 1> new_arr;

		for (int i = 0; i < dimensions - 1; i++)
		{
			if constexpr (i >= axis)
			{
				new_arr[i] = old_arr[i + 1];
			}
			else
			{
				new_arr[i] = old_arr[i];
			}
		}

		return new_arr;
	}


	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	StaticTensor<T, dimensions, var_shape>& StaticTensor<T, dimensions, var_shape>::Compute(std::function<void(T&)> compute_func)
	{
		#pragma omp parallel for
		for (long long index = 0; (size_t)index < size(); index++)
		{
			compute_func(at(index));
		}

		return *this;
	}


	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	T& StaticTensor<T, dimensions, var_shape>::operator[](size_t index)
	{
		return at(index);
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	T StaticTensor<T, dimensions, var_shape>::operator[](size_t index) const
	{
		return at(index);
	}
	
	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<typename ...Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int>>
	T& StaticTensor<T, dimensions, var_shape>::operator()(Tcoords ...coords)
	{
		return get(coords...);
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<typename ...Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int>>
	T StaticTensor<T, dimensions, var_shape>::operator()(Tcoords ...coords) const
	{
		return get(coords...);
	}


	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	std::ostream& StaticTensor<T, dimensions, var_shape>::printable(std::ostream& stream) const
	{
		size_t max_length = 0;

		for (size_t i = 0; i < size(); i++)
		{
			max_length = std::max(std::to_string(at(i)).size(), max_length);
		}

		for (size_t i = 0; i < var_shape[dimensions - 1]; i++)
		{
			stream << at(i);

			size_t str_len = std::to_string(at(i)).size();

			for (size_t j = 0; j < max_length - str_len; j++)
			{
				stream << ' ';
			}

			stream << ',';

			if (i % var_shape[dimensions - 1] == var_shape[dimensions - 1] - 1)
			{
				stream << '\n';
			}
		}

		for (size_t dim = 1; dim < dimensions; dim++)
		{
			stream << "\n";
			for (size_t i = get_real_size(dim - 1); i < get_real_size(dim); i++)
			{
				stream << at(i);

				size_t str_len = std::to_string(at(i)).size();

				for (size_t j = 0; j < max_length - str_len; j++)
				{
					stream << ' ';
				}

				stream << ',';

				if (i % var_shape[dimensions - 1] == var_shape[dimensions - 1] - 1)
				{
					stream << '\n';
				}
			}
		}

		return stream;
	}
}


