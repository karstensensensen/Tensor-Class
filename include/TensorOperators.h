#pragma once

#ifdef _CUDA
#include "TensorCudaBones.cuh"
#else
#include "TensorBones.h"
#endif
#include <tuple>

namespace TSlib
{

}

template<typename T, TSlib::Mode device>
std::ostream& operator<< (std::ostream& stream, const TSlib::TensorSlice<T, device>& slice)
{
	stream << slice.printable();
	return stream;
}

template<typename T, TSlib::Mode device>
std::ostream& operator<< (std::ostream& stream, const TSlib::Tensor<T, device>& Tensor)
{
	stream << Tensor.printable();
	return stream;
}
