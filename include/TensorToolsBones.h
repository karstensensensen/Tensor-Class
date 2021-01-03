#pragma once
#include "TensorBones.h"

namespace TSlib
{
	namespace Tools
	{
		template<typename T, Mode device, typename OT, Mode o_device>
		Tensor<T, device> merge(Tensor<T, device> tensor1, Tensor<OT, o_device> tensor2, const size_t& dimension);

		template<typename T1, typename T2, std::enable_if_t<is_tensor_type<T1>::value && is_tensor_type<T2>::value, int> = 0>
		bool fits(const T1& tensor1, const T2& tensor2);

		template<typename T1, typename T2, std::enable_if_t<is_tensor_type<T1>::value && is_tensor_type<T2>::value, int> = 0>
		void exceptFit(const T1& tensor1, const T2& tensor2);
	}
}
