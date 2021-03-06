#include "TensorBool.h"

namespace TSlib
{
	template<Device device>
	Tensor<bool, device>::Tensor()
	{
		Resize({ 0 });
	}

	template<Device device>
	Tensor<bool, device>::Tensor(const std::vector<size_t>& sizes, bool pad_val)
		: Tensor<unsigned char, device>(), m_bool_shape(sizes)
	{
		Resize(sizes);
	}

	template<Device device>
	Tensor<bool, device>& Tensor<bool, device>::Resize(const std::vector<size_t>& sizes, bool pad_val)
	{
		// the first dimension of the real shape must be 1/8 of the real size rounded up
		size_t tmp_dim = sizes[0];
		m_bool_shape = sizes;
		m_bool_shape[0] = (tmp_dim + 8) / 8;

		Tensor<unsigned char, device>::Resize(m_bool_shape, pad_val ? 0 : UCHAR_MAX);

		m_bool_shape[0] = tmp_dim;
	}

	template<Device device>
	bool Tensor<bool, device>::At(size_t indx) const
	{

	}

	/*template<Device device>
	Tensor<bool, device>& TSlib::Tensor<bool, device>::Resize(const std::vector<size_t>& sizes, bool pad_val)
	{
		m_shape
	}*/
}
