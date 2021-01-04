#pragma once
#include "TensorBones.h"

namespace TSlib
{

	namespace Consts
	{
		static constexpr double Euler = 2.71828182845904523536;
	}

	// takes every element of the tensor and sets e to a power of that element
	// elem = e^elem
	template<typename T, Mode device>
	Tensor<T, device>& Tensor<T, device>::exp()
	{
		Compute([](T& elem) {elem = std::pow(Euler, elem)};);

		return *this;
	}

	template<typename T, Mode device>
	TensorSlice<T, device>& TensorSlice<T, device>::exp()
	{
		Compute([](T& elem) {elem = std::pow(Euler, elem)};);

		return *this;
	}

	// takes the sum of the tensor and divides each element with it.
	// elem = elem/sum(tensor)

	template<typename T, Mode device>
	Tensor<T, device>& Tensor<T, device>::normalize()
	{
		T tensor_sum = sum();
		Compute([=](T& elem) {elem = elem / tensor_sum});

		return *this;
	}

	template<typename T, Mode device>
	TensorSlice<T, device>& TensorSlice<T, device>::normalize()
	{
		T tensor_sum = sum();
		Compute([=](T& elem) {elem = elem / tensor_sum});

		return *this;
	}
}
