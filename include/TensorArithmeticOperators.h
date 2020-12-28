#pragma once

#ifdef _CUDA
#include "TensorCudaBones.cuh"
#else
#include "TensorBones.h"
#endif
#include <tuple>

namespace TSlib
{
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cadd(other);
		}

		return add(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+(const OT& other)
	{
		return add(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Csubtract(other);
		}

		return subtract(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-(const OT& other)
	{
		return subtract(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cmultiply(other);
		}

		return multiply(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*(const OT& other)
	{
		return multiply(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cdivide(other);
		}

		return divide(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/(const OT& other)
	{
		return divide(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cmodulous(other);
		}

		return modulous(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%(const OT& other)
	{
		return modulous(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CaddAsgmt(other);
		}

		return addAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+=(const OT& other)
	{
		return addAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CsubtractAsgmt(other);
		}

		return subtractAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-=(const OT& other)
	{
		return subtractAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CmultiplyAsgmt(other);
		}

		return multiplyAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*=(const OT& other)
	{
		return multiplyAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CdivideAsgmt(other);
		}

		return divideAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/=(const OT& other)
	{
		return divideAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CmodulousAsgmt(other);
		}

		return modulousAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%=(const OT& other)
	{
		return modulousAsgmt(other);
	}
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
