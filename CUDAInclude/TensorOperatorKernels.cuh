#pragma once

#ifdef _CUDA
#include <cuda_runtime.h>

namespace TSlib
{
	template <typename T, typename OT, typename RT>
	__kernel__ CudaAdd(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) + RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaAddSingle(CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) + RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaSubtract(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) - RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaSubtractSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) - RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaMultiply(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) * RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaMultiplySingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) * RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaDot(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b);

	template <typename T, typename OT, typename RT>
	__kernel__ CudaDotSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b);

	template <typename T, typename OT, typename RT>
	__kernel__ CudaDivide(const CUDATensor3D<T> a, CUDATensor3D<RT> c, CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) / RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaDivideSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) / RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaModulous(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) % RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaModulousSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) % RT(b);
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaAdditionAssignment(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() += b.At();
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaAdditionAssignmentSingle(CUDATensor3D<T> a, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() += b;
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaSubtractionAssignment(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() -= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaSubtractionAssignmentSingle(CUDATensor3D<T> a, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() -= b;
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaMultiplicationAssignment(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() *= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaMultiplicationAssignmentSingle(CUDATensor3D<T> a, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() *= b;
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaDotAssignment(CUDATensor3D<T> a, OT* b);

	template <typename T, typename OT>
	__kernel__ CudaDivisionAssignment(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() /= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaDivisionAssignmentSingle(CUDATensor3D<T> a, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() /= b;
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaModulouAssignment(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() %= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaModulouAssignmentSingle(CUDATensor3D<T> a, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() %= b;
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaCompare(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() == b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaCompareSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() == b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThan(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() < b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() < b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThan(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() > b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThanSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() > b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanEqual(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() <= b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanEqualSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() <= b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThanEqual(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			a.At() >= b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThanEqualSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.in_bounds())
		{
			a.At() >= b;
		}
	}
	#endif
}