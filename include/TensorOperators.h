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
		if (i % slice.Shape()[0].width() == slice.Shape()[0].width() - 1)
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
			if (i % slice.Shape()[0].width() == slice.Shape()[0].width() - 1)
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
	stream << Tensor.printable();
	return stream;
}
