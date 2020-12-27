#pragma once
#include "TensorToolsBones.h"

namespace TSlib
{
	namespace Tools
	{
		template<typename T, Mode device, typename OT, Mode o_device>
		Tensor<T, device> merge(Tensor<T, device> tensor1, Tensor<OT, o_device> tensor2, const size_t& dimension)
		{
			#ifdef _TS_DEBUG
			
			if (dimension >= tensor1.Dims())
			{
				throw BadValue("The target dimension must be less than or equal to the total dimensions in the target tensors", ExceptValue{ "Target dimension", dimension }, ExceptValue{ "Tensor dimension", tensor1.Dims() });
			}
			else if (tensor2.Dims() != tensor1.Dims())
			{
				throw BadShape("The source Tensor must have the same amount of dimensions as the destination Tensor", tensor2.Shape(), tensor1.Shape());
			}

			for (size_t i = 0; i < tensor1.Dims(); i++)
			{
				if (tensor1.Shape()[i] != tensor2.Shape()[i] && i != dimension)
				{
					throw BadShape("The source Tensor must match the destination Tensors Shape appart from the dimensions that is getting appended to", tensor2.Shape(), tensor1.Shape());
				}
			}
			#endif

			std::vector<size_t> new_shape = tensor1.Shape();

			new_shape[dimension] += tensor2.Shape()[dimension];

			Tensor<T, device> result(new_shape);

			std::vector<TSlice> append_slice(tensor1.Dims(), All);

			append_slice[dimension] = TSlice(result.Shape()[dimension] - tensor2.Shape()[dimension], -1);

			result.Slice(append_slice).Fill(tensor2);

			append_slice[dimension] = TSlice(0, result.Shape()[dimension] - tensor2.Shape()[dimension]);

			result.Slice(append_slice).Fill(tensor1);

			return result;
		}
	}
}
