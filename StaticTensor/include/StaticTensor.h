#pragma once
#include "TensorExceptions.h"
#include <array>
#include <functional>

namespace TSlib
{

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	class StaticTensor
	{
		std::array<T, size> m_array;

	public:
		StaticTensor(T init_val = T());

		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int> = 0>
		T& get(Tcoords ... coords);
		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int> = 0>
		T get(Tcoords ... coords) const;

		T& at(size_t index);
		T at(size_t index) const;

		template<size_t axis, std::enable_if_t<axis < dimensions, int> = 0>
		StaticTensor<T, n - 1, flatten<axis, n, var_shape>()>& Compute(std::function<void(T&)> compute_func);

		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoods) == dimensions, int> = 0>
		T& operator()(Tcoords ... coords);

		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoods) == dimensions, int> = 0>
		T operator()(Tcoords ... coords) const;

		T& operator[](size_t index);
		
		T operator[](size_t index) const;

	private:
		static constexpr const size_t size();

		template<size_t axis>
		static constexpr std::array<size_t, dimensions - 1> flatten();
	};

	template<size_t ... shape>
	static constexpr std::array<size_t, sizeof...(shape)> make_shape = { shape... };


	template<typename T, size_t ... coords>
	using make_tensor = StaticTensor<T, sizeof...(coords), make_shape<coords...>>;
}

#include "StaticTensor.ipp"
