#pragma once
#include "TensorBones.h"
#include "TensorArithmetic.h"

namespace TSlib
{
	

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