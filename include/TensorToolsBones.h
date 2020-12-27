#pragma once
#include "Tensor.h"

namespace TSlib
{
	namespace Tools
	{
		template<typename T, Mode device, typename OT, Mode o_device>
		Tensor<T, device> merge(Tensor<T, device> tensor1, Tensor<OT, o_device> tensor2, const size_t& dimension);
	}
}
