namespace TSlib
{

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	StaticTensor<T, dimensions, var_shape>::StaticTensor(T init_val)
		: m_array(init_val)
	{}

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

	

	return At(index);

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	inline constexpr const size_t StaticTensor<T, dimensions, var_shape>::size()
	{
		size_t prod = 1;
		for (size_t i = 0; i < n; i++)
		{
			prod *= arr[i];
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