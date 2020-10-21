#pragma once

#ifdef _CUDA
#include <cuda_runtime.h>

namespace TSlib
{
	template <typename T, typename OT, typename RT>
	__kernel__ CudaAdd(const CUDATensor3D<T> a, CUDATensor3D<RT> c, CUDATensor3D<OT> b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) + RT(b.At());
		}
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaAddSingle(const CUDATensor3D<T> a, CUDATensor3D<RT> c, OT b)
	{
		if (a.in_bounds())
		{
			c.At() = RT(a.At()) + RT(b);
		}
	}


	template <typename T, typename OT, typename RT>
	__kernel__ CudaSubtract(const CUDATensor3D<T> a, CUDATensor3D<RT> c, CUDATensor3D<OT> b)
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
	__kernel__ CudaMultiply(const CUDATensor3D<T> a, CUDATensor3D<RT> c, CUDATensor3D<OT> b)
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
	__kernel__ CudaDot(const CUDATensor3D<T> a, CUDATensor3D<RT> c, CUDATensor3D<OT> b);

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
	__kernel__ CudaModulous(const CUDATensor3D<T> a, CUDATensor3D<RT> c, CUDATensor3D<OT> b)
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
	__kernel__ CudaAdditionAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] += b[index];
	}

	template <typename T, typename OT>
	__kernel__ CudaAdditionAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] += (T)b;
	}
	template <typename T, typename OT>
	__kernel__ CudaSubtractionAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] -= b[index];
	}
	template <typename T, typename OT>
	__kernel__ CudaSubtractionAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] -= (T)b;
	}


	template <typename T, typename OT>
	__kernel__ CudaMultiplicationAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] *= b[index];
	}
	template <typename T, typename OT>
	__kernel__ CudaMultiplicationAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] *= (T)b;
	}

	template <typename T, typename OT>
	__kernel__ CudaDotAssignment(size_t width, size_t height, T* a, OT* b);

	template <typename T, typename OT>
	__kernel__ CudaDivisionAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] /= b[index];
	}
	template <typename T, typename OT>
	__kernel__ CudaDivisionAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] /= (T)b;
	}
	template <typename T, typename OT>
	__kernel__ CudaModulouAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] %= b[index];
	}
	template <typename T, typename OT>
	__kernel__ CudaModulouAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] %= (T)b;
	}


	template <typename T, typename OT, typename RT>
	__kernel__ CudaCompare(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] == b[index];
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaCompareSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] == (T)b;
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThan(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] < b[index];
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] < (T)b;
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaMoreThan(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] > b[index];
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaMoreThanSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] > (T)b;
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanEqual(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] <= b[index];
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaLessThanEqualSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] <= (T)b;
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaMoreThanEqual(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] >= b[index];
	}
	template <typename T, typename OT, typename RT>
	__kernel__ CudaMoreThanEqualSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] >= (T)b;
	}
	#endif
}
