#pragma once
#include <Tensor.h>
#include <opencv2\opencv.hpp>

namespace TSlib {
namespace Tools {

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<unsigned char, device>& source);

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<signed char, device>& source);

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<unsigned short, device>& source);

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<short, device>& source);

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<int, device>& source);

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<float, device>& source);

	template<Mode device>
	cv::Mat TensorToMat(const Tensor<double, device>& source);

	template<typename T, Mode device = default_device>
	Tensor<T, device> MatToTensor(const cv::Mat& source);

}
}
