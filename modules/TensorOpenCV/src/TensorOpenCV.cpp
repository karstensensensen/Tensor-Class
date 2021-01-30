#include <iostream>
#include "TensorOpenCV.h"
#include <random>

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned char, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_8UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned char, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_8UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<signed char, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_8SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<signed char, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_8SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned short, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_16UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<unsigned short, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_16UC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<short, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_16SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<short, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_16SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<int, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_32SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<int, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_32SC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<float, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_32FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<float, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_32FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}

template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<double, Mode::CPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_64FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
template<>
cv::Mat TSlib::Tools::TensorToMat(const Tensor<double, Mode::GPU>& source)
{
	if (source.Dims() != 3)
	{
		throw TSlib::BadShape("Shape must have excactly 3 dimensions (width, height, channels)", source.Shape());
	}
	else if (source.Shape()[2] > 4)
	{
		throw TSlib::BadShape("There cannot be more than 4 channels (A, R, G, B)", source.Shape());
	}

	const std::vector<size_t>& shape = source.Shape();

	cv::Mat return_mat(cv::Size(int(shape[0]), int(shape[1])), CV_64FC(int(shape[2])));
	memcpy(return_mat.data, source.Data(), source.size());

	return return_mat;
}
