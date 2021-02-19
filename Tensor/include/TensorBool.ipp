#include "TensorBool.h"

namespace TSlib
{
	template<Device device>
	Tensor<bool, device>::Tensor()
	{
		Tensor<char, device>::Resize({ 0 });
	}

	/*template<Device device>
	Tensor<bool, device>& TSlib::Tensor<bool, device>::Resize(const std::vector<size_t>& sizes, bool pad_val)
	{
		m_shape
	}*/
}
