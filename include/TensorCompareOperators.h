#pragma once

namespace TSlib
{
	template<typename Ta, typename Tb>
	inline static bool Equal(const Ta& a, const Tb& b)
	{
		return a == b;
	}

	template<typename Ta, typename Tb>
	inline static bool NotEqual(const Ta& a, const Tb& b)
	{
		return a != b;
	}

	template<typename Ta, typename Tb>
	inline static bool LessThan(const Ta& a, const Tb& b)
	{
		return a < b;
	}

	template<typename Ta, typename Tb>
	inline static bool GreaterThan(const Ta& a, const Tb& b)
	{
		return a > b;
	}

	template<typename Ta, typename Tb>
	inline static bool LessThanEqual(const Ta& a, const Tb& b)
	{
		return a <= b;
	}

	template<typename Ta, typename Tb>
	inline static bool GreaterThanEqual(const Ta& a, const Tb& b)
	{
		return a >= b;
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator==(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other, Equal).sum<size_t>();
		}

		return (bool)compare(other, Equal).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator==(const OT& other)
	{
		return (bool)compare(other, Equal).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator!=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return !(bool)Ccompare(other, Equal).sum<size_t>();
		}

		return !(bool)compare(other, Equal).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator!=(const OT& other)
	{
		return !(bool)compare(other, Equal).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator<(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other, LessThan).sum<size_t>();
		}

		return (bool)compare(other, LessThan).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator<(const OT& other)
	{
		return (bool)compare(other, LessThan).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator>(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other, GreaterThan).sum<size_t>();
		}

		return (bool)compare(other, GreaterThan).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator>(const OT& other)
	{
		return (bool)compare(other, GreaterThan).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator<=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other, LessThanEqual).sum<size_t>();
		}

		return (bool)compare(other, LessThanEqual).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator<=(const OT& other)
	{
		return (bool)compare(other, LessThanEqual).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator>=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other, GreaterThanEqual).sum<size_t>();
		}

		return (bool)compare(other, GreaterThanEqual).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator>=(const OT& other)
	{
		return (bool)compare(other, GreaterThanEqual).sum<size_t>();
	}
}