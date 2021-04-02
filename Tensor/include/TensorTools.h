#pragma once
#include "Tensor.h"
#include "SaveSequence.h"

namespace TSlib
{
	namespace Tools
	{
		template<typename ... Args>
		using enable_if_tensor = std::enable_if_t<(... && is_tensor_type<Args>::value), int>;

		template<typename T, Device device, typename OT, Device o_device>
		Tensor<T, device> merge(Tensor<T, device> tensor1, Tensor<OT, o_device> tensor2, const size_t& dimension);

		template<typename T1, typename T2, enable_if_tensor<T1, T2> = 0>
		bool fits(const T1& tensor1, const T2& tensor2);

		template<typename T1, typename T2, enable_if_tensor<T1, T2> = 0>
		void exceptFit(const T1& tensor1, const T2& tensor2);

		template<typename T, enable_if_tensor<T> = 0>
		typename T::Type Sum(const T& source);
		template<typename T, enable_if_tensor<T> = 0>
		T Sum(const T& source, size_t axis, bool keepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		typename T::Type Prod(const T& source);
		template<typename T, enable_if_tensor<T> = 0>
		T Prod(const T& source, size_t axis, bool keepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		T Exp(const T& source);
		template<typename T, enable_if_tensor<T> = 0>
		T Exp(const T& source, size_t axis, bool keepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		T Normalize(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T Max(const T& source, size_t axis, bool keepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		T Min(const T& source, size_t axis, bool keepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		T Avg(const T& source, size_t axis, bool keepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		T sin(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T cos(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T tan(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T arcsin(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T arccos(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T arctan(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T convDeg(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T convRad(const T& source);
	}
}

#include "TensorTools.ipp"
