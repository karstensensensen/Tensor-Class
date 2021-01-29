#pragma once
#include <array>
#include <functional>

namespace TSlib
{
	template<typename T, size_t axis, size_t n, const std::array<T, n>& arr>
	static constexpr std::array<T, n - 1> flatten();

	template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
	class StaticTensor
	{
	public:
		static constexpr const size_t size();

	private:
		std::array<T, size()> m_array;
		static constexpr std::array<size_t, dimensions> m_shape = var_shape;

	public:
		
		
		StaticTensor(T init_val = T());
		
		template<typename Tother, size_t dim_other, const std::array<size_t, dim_other>& shape_other>
		StaticTensor(const StaticTensor<Tother, dim_other, shape_other>& other);


		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int> = 0>
		T& get(Tcoords ... coords);
		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int> = 0>
		T get(Tcoords ... coords) const;

		T& at(size_t index);
		T at(size_t index) const;

		StaticTensor<T, dimensions, var_shape>& Compute(std::function<void(T&)> compute_func);

		template<size_t axis, std::enable_if_t<axis < dimensions, int> = 0>
		StaticTensor<T, dimensions - 1, flatten<size_t, axis, var_shape>()> Compute(std::function<void(T&)> compute_func);

		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int> = 0>
		T& operator()(Tcoords ... coords);

		template<typename ... Tcoords, std::enable_if_t<sizeof...(Tcoords) == dimensions, int> = 0>
		T operator()(Tcoords ... coords) const;

		T& operator[](size_t index);
		
		T operator[](size_t index) const;

		std::ostream& printable(std::ostream& stream) const;

	private:
		template<size_t dimension>
		static constexpr size_t get_real_size();
		static size_t get_real_size(size_t dimension);
	};

	template<size_t ... shape>
	static constexpr std::array<size_t, sizeof...(shape)> make_shape = { shape... };


	template<typename T, size_t ... coords>
	using make_tensor = StaticTensor<T, sizeof...(coords), make_shape<coords...>>;
}

#include "StaticTensor.ipp"
