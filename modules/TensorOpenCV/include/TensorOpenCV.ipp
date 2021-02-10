#pragma once

namespace TSlib {
	namespace Tools {

		template<typename T, Mode device>
		Tensor<T, device> MatToTensor(const cv::Mat& source)
		{
			Tensor<T, device> return_tensor({ source.cols, source.rows, source.channels() });
			memcpy(return_tensor.Data(), source.data, return_tensor.size());

			return return_tensor;
		}
	}
}