#pragma once

#ifdef _CUDA
#include "TensorCuda.cuh"
#include "TensorOperatorKernels.cuh"
#else
#include "Tensor.h"
#endif
#include <tuple>

namespace TSlib
{
	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cadd(other);
		}

		return add(other);
	}
	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cadd(other);
		}

		return Add(other);
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Csubtract(other);
		}

		return Subtract(other);
	}
	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Csubtract(other);
		}
		return Subtract(other);
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cmultiply(other);
		}

		return Multiply(other);
	}
	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cmultiply(other);
		}
		return Multiply(other);
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cdivide(other);
		}

		return Divide(other);
	}
	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cdivide(other);
		}
		return Divide(other);
	}

	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cmodulous(other);
		}

		return Modulous(other);
	}
	template<typename T, Device device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return Cmodulous(other);
		}
		return Modulous(other);
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator+=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CadditionAsgmt(other);
		}

		AdditionAsgmt(other);
	}
	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator+=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CadditionAsgmt(other);
		}
		AdditionAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator-=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CsubtractionAsgmt(other);
		}

		SubtractionAsgmt(other);
	}
	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator-=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CsubtractionAsgmt(other);
		}
		SubtractionAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator*=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CmultiplicationAsgmt(other);
		}

		MultiplicationAsgmt(other);
	}
	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator*=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CmultiplicationAsgmt(other);
		}
		MultiplicationAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator/=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CdivisionAsgmt(other);
		}

		DivisionAsgmt(other);
	}
	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator/=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CdivisionAsgmt(other);
		}
		DivisionAsgmt(other);
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator%=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CmodulouAsgmt(other);
		}

		ModulouAsgmt(other);
	}
	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::operator%=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			CmodulouAsgmt(other);
		}
		ModulouAsgmt(other);
	}

	#ifdef _CUDA
	/// <summary>
	/// Tensor class cuda specific operator call functions
	/// </summary>

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cadd(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaAdd<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cadd(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaAdd<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cadd(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaAdd<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Csubtract(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaSubtract<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Csubtract(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaSubtract<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Csubtract(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaSubtract<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cmultiply(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaMultiply<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}
	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cmultiply(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaMultiply<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}
	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cmultiply(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaMultiply<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cdivide(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaDivide<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}
	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cdivide(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaDivide<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}
	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cdivide(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaDivide<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cmodulous(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaModulous<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}
	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Cmodulous(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaModulous<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cmodulous(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaModulous<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CadditionAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaAdditionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CadditionAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaAdditionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::CadditionAsgmt(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, OT)>
			(Layout3D(), CudaAdditionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CsubtractionAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaSubtractionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CsubtractionAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaSubtractionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}
	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::CsubtractionAsgmt(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaSubtractionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CmultiplicationAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaMultiplicationAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CmultiplicationAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaMultiplicationAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::CmultiplicationAsgmt(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaMultiplicationAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CdivisionAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaDivisionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CdivisionAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaDivisionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::CdivisionAsgmt(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaDivisionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CmodulouAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaModulouAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	void Tensor<T, device>::CmodulouAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaModulouAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename OT>
	void Tensor<T, device>::CmodulouAsgmt(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaModulouAsgmt<T, OT>, other);

		if (this_alloc)
		{
			Pull();
		}
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Ccompare(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaCompare<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Ccompare(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaCompare<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::ClessThan(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaLessThan<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::ClessThan(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaLessThan<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::CgreaterThan(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaGreaterThan<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::CgreaterThan(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaGreaterThan<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::ClessThanEqual(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaLessThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::ClessThanEqual(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaLessThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::CgreaterThanEqual(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = IsDeallocated();
		bool other_alloc = other.IsDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		else if (other_alloc)
		{
			throw BadValue("A const Tensor must have already allocated memory on the gpu");
		}
		#endif

		if (this_alloc)
		{
			Push();
		}

		if (other_alloc)
		{
			other.Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaGreaterThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		if (other_alloc)
		{
			other.Pull();
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::CgreaterThanEqual(const OT& other)
	{
		bool this_alloc = IsDeallocated();

		if (this_alloc)
		{
			Push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(const CUDATensor3D<T>, CUDATensor3D<RT>, const OT), RT>
			(Layout3D(), CudaGreaterThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			Pull();
		}

		return result;
	}
	#endif

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator==(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)Ccompare(other).template Sum<size_t>();
		}

		return (bool)Compare(other).template Sum<size_t>();
	}
	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator==(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)Ccompare(other).template Sum<size_t>();
		}
		return (bool)Compare(other).template Sum<size_t>();
	}

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator!=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return !(bool)Ccompare(other).template Sum<size_t>();
		}

		return !(bool)Compare(other).template Sum<size_t>();
	}
	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator!=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return !(bool)Ccompare(other).template Sum<size_t>();
		}
		return !(bool)Compare(other).template Sum<size_t>();
	}

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator<(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)ClessThan(other).template Sum<size_t>();
		}

		return (bool)Compare(other, LessThan).template Sum<size_t>();
	}

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator<(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)ClessThan(other).template Sum<size_t>();
		}
		return (bool)Compare(other, LessThan).template Sum<size_t>();
	}

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator>(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)CgreaterThan(other).template Sum<size_t>();
		}

		return (bool)Compare(other, GreaterThan).template Sum<size_t>();
	}
	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator>(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)CgreaterThan(other).template Sum<size_t>();
		}
		return (bool)Compare(other, GreaterThan).template Sum<size_t>();
	}

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator<=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)ClessThanEqual(other).template Sum<size_t>();
		}

		return (bool)Compare(other, LessThanEqual).template Sum<size_t>();
	}
	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator<=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)ClessThanEqual(other).template Sum<size_t>();
		}
		return (bool)Compare(other, LessThanEqual).template Sum<size_t>();
	}

	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator>=(OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)CgreaterThanEqual(other).template Sum<size_t>();
		}

		return (bool)Compare(other, GreaterThanEqual).template Sum<size_t>();
	}
	template<typename T, Device device>
	template<typename OT>
	bool Tensor<T, device>::operator>=(const OT& other)
	{
		if constexpr (device == Device::GPU)
		{
			return (bool)CgreaterThanEqual(other).template Sum<size_t>();
		}
		return (bool)Compare(other, GreaterThanEqual).template Sum<size_t>();
	}
}

template<typename T, TSlib::Device device>
std::ostream& operator<< (std::ostream& stream, const TSlib::TensorSlice<T, device>& slice)
{
	size_t max_length = 0;

	for (size_t i = 0; i < slice.size(); i++)
	{
		max_length = std::max(std::to_string(slice.At(i)).size(), max_length);
	}

	for (size_t i = 0; i < slice.Shape()[slice.Dims() - 1]; i++)
	{
		stream << std::to_string(slice.At(i));

		size_t str_len = std::to_string(slice.At(i)).size();

		for (size_t j = 0; j < max_length - str_len; j++)
		{
			stream << ' ';
		}

		stream << ',';

		if (i % slice.Shape()[slice.Dims() - 1] == slice.Shape()[slice.Dims() - 1] - 1)
		{
			stream << '\n';
		}
	}

	for (size_t dim = 1; dim < slice.Dims(); dim++)
	{
		stream << "\n";
		for (size_t i = slice.get_real_size(dim - 1); i < slice.get_real_size(dim); i++)
		{
			stream << std::to_string(At(i));

			size_t str_len = std::to_string(At(i)).size();

			for (size_t j = 0; j < max_length - str_len; j++)
			{
				stream << ' ';
			}

			stream << ',';

			if (i % slice.Shape()[slice.Dims() - 1] == slice.Shape()[slice.Dims() - 1] - 1)
			{
				stream << '\n';
			}
		}
	}

	return stream;
}
namespace TSlib
{
	template<typename Tprint, Device device_print>
	std::ostream& operator<<(std::ostream& stream, const Tensor<Tprint, device_print>& tensor)
	{
		size_t max_length = 0;

		for (const Tprint& elem : tensor)
		{
			max_length = std::max(to_string(elem).size(), max_length);
		}

		for (size_t i = 0; i < tensor.Shape()[tensor.Dims() - 1]; i++)
		{
			stream << to_string(tensor[i]);

			size_t str_len = to_string(tensor[i]).size();

			for (size_t j = 0; j < max_length - str_len; j++)
			{
				stream << ' ';
			}

			stream << ',';

			if (i % tensor.Shape()[tensor.Dims() - 1] == tensor.Shape()[tensor.Dims() - 1] - 1)
			{
				stream << '\n';
			}
		}

		for (size_t dim = 1; dim < tensor.Dims(); dim++)
		{
			stream << "\n";
			for (size_t i = tensor.get_real_size(dim - 1); i < tensor.get_real_size(dim); i++)
			{
				stream << to_string(tensor[i]);

				size_t str_len = to_string(tensor[i]).size();

				for (size_t j = 0; j < max_length - str_len; j++)
				{
					stream << ' ';
				}

				stream << ',';

				if (i % tensor.Shape()[tensor.Dims() - 1] == tensor.Shape()[tensor.Dims() - 1] - 1)
				{
					stream << '\n';
				}
			}
		}

		return stream;
	}
}
