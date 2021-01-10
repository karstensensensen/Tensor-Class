#pragma once
#include "TensorBones.h"

namespace TSlib
{
	namespace Tools
	{
		template<typename ... Args>
		using enable_if_tensor = std::enable_if_t<(... && is_tensor_type<Args>::value), int>;

		template<typename T, Mode device, typename OT, Mode o_device>
		Tensor<T, device> merge(Tensor<T, device> tensor1, Tensor<OT, o_device> tensor2, const size_t& dimension);

		template<typename T1, typename T2, enable_if_tensor<T1, T2> = 0>
		bool fits(const T1& tensor1, const T2& tensor2);

		template<typename T1, typename T2, enable_if_tensor<T1, T2> = 0>
		void exceptFit(const T1& tensor1, const T2& tensor2);

		template<typename T, enable_if_tensor<T> = 0>
		Tensor<typename T::Type, T::Device> exp(const T& source);
		template<typename T, enable_if_tensor<T> = 0>
		Tensor<typename T::Type, T::Device> exp(const T& source, size_t axis, bool KeepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		Tensor<typename T::Type, T::Device> normalize(const T& source);

		template<typename T, enable_if_tensor<T> = 0>
		T max(const T& source, size_t axis, bool KeepDims = false);

		template<typename T, enable_if_tensor<T> = 0>
		T min(const T& source, size_t axis, bool KeepDims = false);
	}
}
