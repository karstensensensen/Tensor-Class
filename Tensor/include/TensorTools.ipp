#pragma once
#include "TensorTools.h"

namespace TSlib
{
	namespace Tools
	{
		template<typename T, Device device, typename OT, Device o_device>
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

		template<typename T1, typename T2, enable_if_tensor<T1, T2>>
		bool fits(const T1& tensor1, const T2& tensor2)
		{
			if (tensor1.Dims() != tensor2.Dims())
				return false;

			for (size_t dim = 0; dim < tensor1.Dims(); dim++)
			{
				size_t dim1 = tensor1.Shape()[dim];
				size_t dim2 = tensor2.Shape()[dim];

				if (dim1 != dim2)
				{
					return false;
				}
			}

			return true;
		}

		template<typename T1, typename T2, enable_if_tensor<T1, T2>>
		void exceptFit(const T1& tensor1, const T2& tensor2)
		{
			if (!fits(tensor1, tensor2))
			{
				throw BadShape("The two Tensors could not fit into each other. They must have the same shape", tensor1.Shape(), tensor2.Shape());
			}
		}
	}
}