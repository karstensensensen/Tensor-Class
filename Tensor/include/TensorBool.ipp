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

		Tensor<unsigned char, device>::Resize(m_bool_shape, pad_val ? UCHAR_MAX : 0);

		m_bool_shape[0] = tmp_dim;

		return *this;
	}

	template<Device device>
	bool Tensor<bool, device>::At(size_t indx) const
	{
		unsigned char bit = indx % 8;
		indx = indx / 8;

		return (this->m_vector[indx] >> bit) & 1;
	}

	template<Device device>
	void Tensor<bool, device>::SetIdx(bool val, size_t indx)
	{

		/*#ifdef _TS_DEBUG
		if(size_t indx > )
		#endif*/

		unsigned char bit = indx % 8;
		indx = indx / 8;

		this->m_vector[indx] = (this->m_vector[indx] & ~(1U << bit)) | (val << bit);
	}

	/*template<Device device>
	Tensor<bool, device>& TSlib::Tensor<bool, device>::Resize(const std::vector<size_t>& sizes, bool pad_val)
	{
		m_shape
	}*/
}
