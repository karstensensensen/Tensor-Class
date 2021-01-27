namespace TSlib
{

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	StaticTensor<T, dimensions, var_shape>::StaticTensor(T init_val)
	{
		m_array.fill(init_val);
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
		std::array<size_t, dimensions> coords = { var_coords... };

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
				throw OutOfBounds(Shape(), "Exception was thrown, because an element outside the Tensor bounds was accsessed", i, coords[i]);
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
	inline constexpr const size_t StaticTensor<T, dimensions, var_shape>::size()
	{
		size_t prod = 1;
		for (size_t i = 0; i < dimensions; i++)
		{
			prod *= var_shape[i];
		}

		return prod;
	}

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	template<size_t axis>
	inline constexpr std::array<size_t, dimensions - 1> StaticTensor<T, dimensions, var_shape>::flatten()
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
}