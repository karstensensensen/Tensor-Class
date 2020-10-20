#pragma once

#ifdef _CUDA
#include <cuda_runtime.h>

namespace TSlib
{
	template <typename T, typename OT, typename RT>
	__global__ void CudaAdd(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] + (RT)b[index];

	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaAddSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] + (RT)b;

	}


	template <typename T, typename OT, typename RT>
	__global__ void CudaSubtract(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] - (RT)b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaSubtractSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] - (RT)b;
	}


	template <typename T, typename OT, typename RT>
	__global__ void CudaMultiply(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] * (RT)b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaMultiplySingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] * (RT)b;
	}

	template <typename T, typename OT, typename RT>
	__global__ void CudaDot(size_t width, size_t height, const T* a, RT* c, OT* b);
	template <typename T, typename OT, typename RT>
	__global__ void CudaDivide(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] / (RT)b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaDivideSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] / (RT)b;
	}

	template <typename T, typename OT, typename RT>
	__global__ void CudaModulous(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] % (RT)b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaModulousSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = (RT)a[index] % (RT)b;
	}


	template <typename T, typename OT>
	__global__ void CudaAdditionAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] += b[index];
	}

	template <typename T, typename OT>
	__global__ void CudaAdditionAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] += (T)b;
	}
	template <typename T, typename OT>
	__global__ void CudaSubtractionAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] -= b[index];
	}
	template <typename T, typename OT>
	__global__ void CudaSubtractionAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] -= (T)b;
	}


	template <typename T, typename OT>
	__global__ void CudaMultiplicationAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] *= b[index];
	}
	template <typename T, typename OT>
	__global__ void CudaMultiplicationAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] *= (T)b;
	}

	template <typename T, typename OT>
	__global__ void CudaDot_assignment(size_t width, size_t height, T* a, OT* b);
	template <typename T, typename OT>
	__global__ void CudaDivisionAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] /= b[index];
	}
	template <typename T, typename OT>
	__global__ void CudaDivisionAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] /= (T)b;
	}
	template <typename T, typename OT>
	__global__ void CudaModulouAssignment(size_t width, size_t height, T* a, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] %= b[index];
	}
	template <typename T, typename OT>
	__global__ void CudaModulouAssignmentSingle(size_t width, size_t height, T* a, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			a[index] %= (T)b;
	}


	template <typename T, typename OT, typename RT>
	__global__ void CudaCompare(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] == b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaCompareSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] == (T)b;
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaLessThan(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] < b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaLessThanSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] < (T)b;
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaMoreThan(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] > b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaMoreThanSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] > (T)b;
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaLessThanEqual(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] <= b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaLessThanEqualSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] <= (T)b;
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaMoreThanEqual(size_t width, size_t height, const T* a, RT* c, OT* b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] >= b[index];
	}
	template <typename T, typename OT, typename RT>
	__global__ void CudaMoreThanEqualSingle(size_t width, size_t height, const T* a, RT* c, const OT b)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int index = row * width + col;

		if ((col < width) && (row < height))
			c[index] = a[index] >= (T)b;
	}
	#endif
}
