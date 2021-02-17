#include <iostream>
#include "TensorOpenCV.h"
#include <random>

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned char, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_8UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(unsigned char));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned char, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_8UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(unsigned char));

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<signed char, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_8SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(signed char));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<signed char, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_8SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(signed char));

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned short, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_16UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(unsigned short));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned short, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_16UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(unsigned short));

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<short, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_16SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(short));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<short, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_16SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(short));

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<int, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_32SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(int));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<int, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_32SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(int));

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<float, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_32FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(float));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<float, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_32FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(float));

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<double, Device::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_64FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(double));

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<double, Device::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (height, width, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[1]), int(shape[0])), CV_64FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size() * sizeof(double));

	return return_mat;
}