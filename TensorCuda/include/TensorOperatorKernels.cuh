#pragma once

#ifdef _TS_CUDA
#include <cuda_runtime.h>

namespace TSlib
{
	template <typename T, typename OT, typename RT>
	__kernel__ CudaAdd(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) + RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaAdd(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) + RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaSubtract(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) - RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaSubtract(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) - RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaMultiply(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) * RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaMultiply(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) * RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaDot(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b);

	template <typename T, typename OT, typename RT>
	__kernel__ CudaDot(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b);

	template <typename T, typename OT, typename RT>
	__kernel__ CudaDivide(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) / RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaDivide(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) / RT(b);
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaModulous(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) % RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaModulous(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			c.At() = RT(a.At()) % RT(b);
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaAdditionAsgmt(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() += b.At();
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaAdditionAsgmt(CUDATensor3D<T> a, const OT b)
	{
		if (a.InBounds())
		{
			a.At() += b;
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaSubtractionAsgmt(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() -= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaSubtractionAsgmt(CUDATensor3D<T> a, const OT b)
	{
		if (a.InBounds())
		{
			a.At() -= b;
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaMultiplicationAsgmt(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() *= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaMultiplicationAsgmt(CUDATensor3D<T> a, const OT b)
	{
		if (a.InBounds())
		{
			a.At() *= b;
		}
	}

	template <typename T, typename OT>
	__kernel__ CudaDotAsgmt(CUDATensor3D<T> a, OT* b);

	template <typename T, typename OT>
	__kernel__ CudaDivisionAsgmt(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() /= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaDivisionAsgmt(CUDATensor3D<T> a, const OT b)
	{
		if (a.InBounds())
		{
			a.At() /= b;
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaModulouAsgmt(CUDATensor3D<T> a, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() %= b.At();
		}
	}
	template <typename T, typename OT>
	__kernel__ CudaModulouAsgmt(CUDATensor3D<T> a, const OT b)
	{
		if (a.InBounds())
		{
			a.At() %= b;
		}
	}

	template <typename T, typename OT, typename RT>
	__kernel__ CudaCompare(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() == b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaCompare(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			a.At() == b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThan(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() < b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThan(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			a.At() < b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThan(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() > b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThan(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			a.At() > b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanEqual(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() <= b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanEqual(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			a.At() <= b;
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThanEqual(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const CUDATensor3D<OT> b)
	{
		if (a.InBounds())
		{
			a.At() >= b.At();
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaGreaterThanEqual(const CUDATensor3D<T> a, CUDATensor3D<RT> c, const OT b)
	{
		if (a.InBounds())
		{
			a.At() >= b;
		}
	}
	#endif
}