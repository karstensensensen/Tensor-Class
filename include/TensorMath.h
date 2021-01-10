#pragma once
#include "TensorBones.h"
#include "TensorToolsBones.h"

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
		Compute([](T& elem) {elem = T(std::pow(Consts::Euler, double(elem))); });

		return *this;
	}

	template<typename T, Mode device>
	TensorSlice<T, device>& TensorSlice<T, device>::exp()
	{
		Compute([](T& elem) {elem = std::pow(T(Consts::Euler), double(elem)); });

		return *this;
	}

	template<typename T, Tools::enable_if_tensor<T>>
	Tensor<typename T::Type, T::Device> Tools::exp(const T& source)
	{
		Tensor<typename T::Type, T::Device> result = source;

		result.exp();

		return result;
	}

	template<typename T, Tools::enable_if_tensor<T>>
	Tensor<typename T::Type, T::Device> Tools::exp(const T& source, size_t axis, bool KeepDims)
	{
		std::vector<size_t> return_shape(source.Shape());

		return_shape[axis] = 1;

		Tensor<typename T::Type, T::Device> result(return_shape, 0);

		result.Compute([&](typename T::Type& elem, const std::vector<size_t>& coords)
			{
				std::vector<size_t> new_coords = coords;
				new_coords[axis] = 0;

				double new_elem = 0;
				for (size_t i = 0; i < source.Shape()[axis]; i++)
				{
					new_elem += std::pow(Consts::Euler, double(source.Get(new_coords)));
					new_coords[axis]++;
				}
				elem = typename T::Type(new_elem);
			});

		if (!KeepDims)
		{
			return_shape.resize(result.Dims() - 1);

			for (size_t i = 0; i < return_shape.size(); i++)
			{
				return_shape[i] = source.Shape()[i] * (i < axis - 1) + source.Shape()[i + 1] * (i >= axis - 1);
			}

			result.Reshape(return_shape);
		}

		return result;
	}

	// takes the sum of the tensor and divides each element with it.
	// elem = elem/sum(tensor)

	template<typename T, Mode device>
	Tensor<T, device>& Tensor<T, device>::normalize()
	{
		T tensor_sum = sum();
		Compute([=](T& elem) {elem = elem / tensor_sum; });

		return *this;
	}

	template<typename T, Mode device>
	TensorSlice<T, device>& TensorSlice<T, device>::normalize()
	{
		T tensor_sum = sum();
		Compute([=](T& elem) {elem = elem / tensor_sum; });

		return *this;
	}

	template<typename T, Tools::enable_if_tensor<T>>
	Tensor<typename T::Type, T::Device> Tools::normalize(const T& source)
	{
		Tensor<typename T::Type, T::Device> result = source;
		result.normalize();

		return result;
	}

	// takes the max element of the tensor
	// largest elem = max(largest elem, current elem)

	template<typename T, Mode device>
	T Tensor<T, device>::max() const
	{
		T max_elem = At(0);

		for size_t i = 1; i < size(), i++)
		{
		max_elem = std::max(max_elem, At(i));
		}

		return max_elem;
	}

	template<typename T, Mode device>
	T TensorSlice<T, device>::max() const
	{
		T max_elem = At(0);

		for (size_t i = 1; i < size(), i++)
		{
			max_elem = std::max(max_elem, At(i));
		}

		return max_elem;
	}

	template<typename T, Tools::enable_if_tensor<T>>
	T Tools::max(const T& source, size_t axis, bool KeepDims)
	{
		T result = source;
		result.Compute([](typename T::Type& new_elem, const typename T::Type& elem) {new_elem = std::max(new_elem, elem); }, axis, KeepDims);
		return result;
	}

	// takes the min element of the tensor
	// largest elem = mix(largest elem, current elem)

	template<typename T, Mode device>
	T Tensor<T, device>::min() const
	{
		T min_elem = At(0);

		for (size_t i = 1; i < size(), i++)
		{
			min_elem = std::min(min_elem, At(i));
		}

		return min_elem;
	}

	template<typename T, Mode device>
	T TensorSlice<T, device>::min() const
	{
		T min_elem = At(0);

		for (size_t i = 1; i < size(), i++)
		{
			max_elem = std::min(min_elem, At(i));
		}

		return min_elem;
	}

	template<typename T, Tools::enable_if_tensor<T>>
	T Tools::min(const T& source, size_t axis, bool KeepDims)
	{
		T result = source;
		result.Compute([](typename T::Type& new_elem, const typename T::Type& elem) {new_elem = std::min(new_elem, elem); }, axis, KeepDims);
		return result;
	}
}
