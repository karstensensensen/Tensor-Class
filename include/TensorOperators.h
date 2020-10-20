#pragma once

#ifdef _CUDA
#include "TensorCuda.cuh"
#else
#include "TensorBones.h"
#endif
#include <tuple>

template<typename T, TSlib::Mode device>
std::ostream& operator<< (std::ostream& stream, const TSlib::TensorSlice<T, device>& slice)
{
	MEASURE();
	for (size_t i = 0; i < slice.get_dim_size(0); i++)
	{
		stream << slice[i] << ',';
		if (i % slice.DimSizes()[0].width() == slice.DimSizes()[0].width() - 1)
		{
			stream << '\n';
		}
	}

	for (unsigned int dim = 1; dim < slice.Dims(); dim++)
	{
		stream << "\n";
		for (size_t i = slice.get_dim_size(dim - 1); i < slice.get_dim_size(dim); i++)
		{
			stream << slice[i] << ',';
			if (i % slice.DimSizes()[0].width() == slice.DimSizes()[0].width() - 1)
			{
				stream << '\n';
			}
		}
	}

	return stream;
}

template<typename T, TSlib::Mode device>
std::ostream& operator<< (std::ostream& stream, const TSlib::Tensor<T, device>& Tensor)
{

	size_t max_length = 0;
	for (const T& elem : Tensor)
	{

		max_length = std::max(std::to_string(elem).size(), max_length);
	}

	for (size_t i = 0; i < Tensor.get_dim_size(0); i++)
	{
		stream << Tensor[i] << ',';

		size_t str_len = std::to_string(Tensor[i]).size();

		for (size_t j = 0; j < max_length - str_len; j++)
		{
			std::cout << ' ';
		}

		if (i % Tensor.DimSizes()[0] == Tensor.DimSizes()[0] - 1)
		{
			stream << '\n';
		}
	}

	for (unsigned int dim = 1; dim < Tensor.Dims(); dim++)
	{
		stream << "\n";
		for (size_t i = Tensor.get_dim_size(dim - 1); i < Tensor.get_dim_size(dim); i++)
		{
			stream << Tensor[i] << ',';

			size_t str_len = std::to_string(Tensor[i]).size();

			for (size_t j = 0; j < max_length - str_len; j++)
			{
				std::cout << ' ';
			}

			if (i % Tensor.DimSizes()[0] == Tensor.DimSizes()[0] - 1)
			{
				stream << '\n';
			}
		}
	}
	return stream;
}
