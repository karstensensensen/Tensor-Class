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
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cadd(other);
		}

		return add(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cadd(other);
		}

		return add(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Csubtract(other);
		}

		return subtract(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Csubtract(other);
		}
		return subtract(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cmultiply(other);
		}

		return multiply(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cmultiply(other);
		}
		return multiply(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cdivide(other);
		}

		return divide(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cdivide(other);
		}
		return divide(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cmodulous(other);
		}

		return modulous(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return Cmodulous(other);
		}
		return modulous(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CaddAsgmt(other);
		}

		return addAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator+=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CaddAsgmt(other);
		}
		return addAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CsubtractAsgmt(other);
		}

		return subtractAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator-=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CsubtractAsgmt(other);
		}
		return subtractAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CmultiplyAsgmt(other);
		}

		return multiplyAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator*=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CmultiplyAsgmt(other);
		}
		return multiplyAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CdivideAsgmt(other);
		}

		return divideAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator/=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CdivideAsgmt(other);
		}
		return divideAsgmt(other);
	}

	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CmodulousAsgmt(other);
		}

		return modulousAsgmt(other);
	}
	template<typename T, Mode device>
	template<typename OT>
	Tensor<T, device> Tensor<T, device>::operator%=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return CmodulousAsgmt(other);
		}
		return modulousAsgmt(other);
	}

	#ifdef _CUDA
	/// <summary>
	/// Tensor class cuda specific operator call functions
	/// </summary>

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cadd(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaAdd<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cadd(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaAdd<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cadd(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaAdd<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Csubtract(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaSubtract<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Csubtract(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaSubtract<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Csubtract(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaSubtract<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cmultiply(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaMultiply<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}
	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cmultiply(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaMultiply<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}
	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cmultiply(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaMultiply<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cdivide(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaDivide<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}
	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cdivide(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaDivide<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}
	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cdivide(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaDivide<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cmodulous(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaModulous<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}
	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::Cmodulous(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaModulous<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Cmodulous(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaModulous<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CadditionAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaAdditionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CadditionAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaAdditionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	void Tensor<T, device>::CadditionAsgmt(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, OT)>
			(Layout3D(), CudaAdditionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CsubtractionAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaSubtractionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CsubtractionAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaSubtractionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}
	template<typename T, Mode device>
	template<typename OT>
	void Tensor<T, device>::CsubtractionAsgmt(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaSubtractionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CmultiplicationAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaMultiplicationAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CmultiplicationAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaMultiplicationAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	void Tensor<T, device>::CmultiplicationAsgmt(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaMultiplicationAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CdivisionAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaDivisionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CdivisionAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaDivisionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	void Tensor<T, device>::CdivisionAsgmt(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaDivisionAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CmodulouAsgmt(Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

		#ifdef _TS_DEBUG

		if (size() != other.size())
		{
			throw BadShape("The source tensor must have the same shape as the destination Tensor", other.Shape(), Shape());
		}
		#endif

		if (this_alloc)
		{
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaModulouAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	void Tensor<T, device>::CmodulouAsgmt(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaModulouAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	void Tensor<T, device>::CmodulouAsgmt(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Kernel3D<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<OT>)>
			(Layout3D(), CudaModulouAsgmt<T, OT>, other);

		if (this_alloc)
		{
			pull();
		}
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::Ccompare(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaCompare<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::Ccompare(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaCompare<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::ClessThan(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaLessThan<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::ClessThan(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaLessThan<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::CgreaterThan(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaGreaterThan<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::CgreaterThan(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaGreaterThan<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::ClessThanEqual(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaLessThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::ClessThanEqual(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, OT), RT>
			(Layout3D(), CudaLessThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	Tensor<RT, device> Tensor<T, device>::CgreaterThanEqual(const Tensor<OT, o_device>& other)
	{
		bool this_alloc = isDeallocated();
		bool other_alloc = other.isDeallocated();

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
			push();
		}

		if (other_alloc)
		{
			other.push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(CUDATensor3D<T>, CUDATensor3D<RT>, CUDATensor3D<OT>), RT>
			(Layout3D(), CudaGreaterThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		if (other_alloc)
		{
			other.pull();
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	Tensor<RT, device> Tensor<T, device>::CgreaterThanEqual(const OT& other)
	{
		bool this_alloc = isDeallocated();

		if (this_alloc)
		{
			push();
		}

		Tensor<RT, device> result = Kernel3DR<Mode::Cube, void(*)(const CUDATensor3D<T>, CUDATensor3D<RT>, const OT), RT>
			(Layout3D(), CudaGreaterThanEqual<T, OT, RT>, other);

		if (this_alloc)
		{
			pull();
		}

		return result;
	}
	#endif

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator==(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other).sum<size_t>();
		}

		return (bool)compare(other).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator==(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)Ccompare(other).sum<size_t>();
		}
		return (bool)compare(other).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator!=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return !(bool)Ccompare(other).sum<size_t>();
		}

		return !(bool)compare(other).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator!=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return !(bool)Ccompare(other).sum<size_t>();
		}
		return !(bool)compare(other).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator<(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)ClessThan(other).sum<size_t>();
		}

		return (bool)compare(other, LessThan).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator<(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)ClessThan(other).sum<size_t>();
		}
		return (bool)compare(other, LessThan).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator>(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)CgreaterThan(other).sum<size_t>();
		}

		return (bool)compare(other, GreaterThan).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator>(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)CgreaterThan(other).sum<size_t>();
		}
		return (bool)compare(other, GreaterThan).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator<=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)ClessThanEqual(other).sum<size_t>();
		}

		return (bool)compare(other, LessThanEqual).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator<=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)ClessThanEqual(other).sum<size_t>();
		}
		return (bool)compare(other, LessThanEqual).sum<size_t>();
	}

	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator>=(OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)CgreaterThanEqual(other).sum<size_t>();
		}

		return (bool)compare(other, GreaterThanEqual).sum<size_t>();
	}
	template<typename T, Mode device>
	template<typename OT>
	bool Tensor<T, device>::operator>=(const OT& other)
	{
		if constexpr (device == Mode::GPU)
		{
			return (bool)CgreaterThanEqual(other).sum<size_t>();
		}
		return (bool)compare(other, GreaterThanEqual).sum<size_t>();
	}
}

template<typename T, TSlib::Mode device>
std::ostream& operator<< (std::ostream& stream, const TSlib::TensorSlice<T, device>& slice)
{
	stream << slice.printable();
	return stream;
}

template<typename T, TSlib::Mode device>
std::ostream& operator<< (std::ostream& stream, const TSlib::Tensor<T, device>& Tensor)
{
	stream << Tensor.printable();
	return stream;
}