#pragma once
#ifdef _CUDA

#include "TensorCudaBones.cuh"
#include "TensorOperatorKernels.cuh"
#include <algorithm>

namespace TSlib
{
	/// <summary>
	/// CUDATensor implementation
	/// </summary>

	/// <summary>
	/// CUDATensor private funtions
	/// </summary>

	template<typename T>
	template<typename First>
	__device__ void CTBase<T>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord) const
	{
		#ifdef _TS_DEBUG
		assert("Index is out of bounds" && dim_arr[iter] > coord);
		#endif

		tmp_multiply /= dim_arr[iter];
		indx += coord * tmp_multiply;
		iter++;
	}

	template<typename T>
	template<typename First, typename... Args>
	__device__ void CTBase<T>::get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining) const
	{
		get_indx(indx, iter, tmp_multiply, coord);
		get_indx(indx, iter, tmp_multiply, remaining...);
	}

	/// <summary>
	/// CUDATensor base default constructors and destructors
	/// </summary>

	template<typename T>
	CTBase<T>::CTBase(T* gpu_mem, size_t m_size, size_t* dim_arr, size_t dims)
		:gpu_mem(gpu_mem), m_size(m_size), dim_arr(dim_arr), dims(dims)
	{}

	template<typename T>
	CTBase<T>::~CTBase()
	{
		CER(cudaFree(dim_arr));
	}

	/// <summary>
	/// CUDATensor public funtions
	/// </summary>

	template<typename T>
	__device__ T* CTBase<T>::get_gpu()
	{
		return gpu_mem;
	}

	template<typename T>
	__device__ const T* CTBase<T>::get_gpu() const
	{
		return gpu_mem;
	}

	template<typename T>
	__device__ size_t CTBase<T>::size() const
	{
		return m_size;
	}

	template<typename T>
	template<typename ... Args>
	__device__ T& CTBase<T>::Get(Args ... coords)

	{
		size_t index = 0;
		size_t tmp_multiply = size();
		size_t i = 0;

		get_indx(index, i, tmp_multiply, coords...);

		return get_gpu()[index];
	}

	/// <summary>
	/// CUDATensor operator funtions
	/// </summary>

	template<typename T>
	__device__ T& CTBase<T>::operator[](size_t index)
	{
		return get_gpu()[index];
	}

	template<typename T>
	template<typename ... Args>
	__device__ T& CTBase<T>::operator()(Args ... args)
	{
		return Get(args...);
	}

	/// <summary>
	/// CUDATensor children private functions
	/// </summary>

	template<typename T>
	__device__ size_t CUDATensor1D<T>::get_length()
	{
		return m_length;
	}

	template<typename T>
	__device__ size_t CUDATensor2D<T>::get_length()
	{
		return m_length;
	}

	template<typename T>
	__device__ size_t CUDATensor2D<T>::get_width()
	{
		return m_width;
	}

	template<typename T>
	__device__ size_t CUDATensor3D<T>::get_length()
	{
		return m_length;
	}

	template<typename T>
	__device__ size_t CUDATensor3D<T>::get_width()
	{
		return m_width;
	}

	template<typename T>
	__device__ size_t CUDATensor3D<T>::get_height()
	{
		return m_height;
	}

	/// <summary>
	/// CUDATensor children constructors
	/// </summary>

	template<typename T>
	template<Mode device>
	CUDATensor1D<T>::CUDATensor1D(Tensor<T, device>& tensor)
		: CTBase(tensor.gpu_mem, tensor.size(), nullptr, tensor.Dims())
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, tensor.Shape().data(), sizeof(size_t) * dims, cudaMemcpyHostToDevice));

		m_length = tensor.FlattenDims();
	}

	template<typename T>
	template<Mode device>
	CUDATensor1D<T>::CUDATensor1D(const Tensor<T, device>& tensor) const
		: CTBase(tensor.gpu_mem, tensor.size(), nullptr, tensor.Dims())
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, tensor.Shape().data(), sizeof(size_t) * dims, cudaMemcpyHostToDevice));

		m_length = tensor.FlattenDims();
	}

	template<typename T>
	CUDATensor1D<T>::CUDATensor1D(const CUDATensor1D<T>& tensor)
		: CTBase(other.gpu_mem, other.m_size, nullptr, other.dims), m_length(other.m_length)
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, other.dim_arr, sizeof(size_t) * dims, cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	template<Mode device>
	CUDATensor2D<T>::CUDATensor2D(Tensor<T, device>& tensor)
		: CTBase(tensor.gpu_mem, tensor.size(), nullptr, tensor.Dims())
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, tensor.Shape().data(), sizeof(size_t) * dims, cudaMemcpyHostToDevice));

		auto new_dims = tensor.FlattenDims(2);
		m_length = new_dims[1];
		m_width = new_dims[0];
	}

	template<typename T>
	template<Mode device>
	CUDATensor2D<T>::CUDATensor2D(const Tensor<T, device>& tensor)
		: CTBase(tensor.gpu_mem, tensor.size(), nullptr, tensor.Dims())
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, tensor.Shape().data(), sizeof(size_t) * dims, cudaMemcpyHostToDevice));

		auto new_dims = tensor.FlattenDims(2);
		m_length = new_dims[1];
		m_width = new_dims[0];
	}

	template<typename T>
	CUDATensor2D<T>::CUDATensor2D(const CUDATensor2D<T>& tensor)
		: CTBase(other.gpu_mem, other.m_size, nullptr, other.dims), m_length(other.m_length), m_width(other.m_width)
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, other.dim_arr, sizeof(size_t) * dims, cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	template<Mode device>
	CUDATensor3D<T>::CUDATensor3D(Tensor<T, device>& tensor)
		:CTBase(tensor.gpu_mem, tensor.size(), nullptr, tensor.Dims())
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, tensor.Shape().data(), sizeof(size_t) * dims, cudaMemcpyHostToDevice));

		auto new_dims = tensor.FlattenDims(3);
		m_length = new_dims[2];
		m_width = new_dims[1];
		m_height = new_dims[0];
	}

	template<typename T>
	template<Mode device>
	CUDATensor3D<T>::CUDATensor3D(const Tensor<T, device>& tensor)
		:CTBase(tensor.gpu_mem, tensor.size(), nullptr, tensor.Dims())
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, tensor.Shape().data(), sizeof(size_t) * dims, cudaMemcpyHostToDevice));

		auto new_dims = tensor.FlattenDims(3);
		m_length = new_dims[2];
		m_width = new_dims[1];
		m_height = new_dims[0];
	}

	template<typename T>
	CUDATensor3D<T>::CUDATensor3D(const CUDATensor3D<T>& tensor)
		: CTBase(other.gpu_mem, other.m_size, nullptr, other.dims), m_length(other.m_length), m_width(other.m_heihgt), m_length(other.m_height)
	{
		CER(cudaMalloc(&dim_arr, sizeof(size_t) * dims));
		CER(cudaMemcpy(dim_arr, other.dim_arr, sizeof(size_t) * dims, cudaMemcpyDeviceToDevice));
	}

	/// <summary>
	/// CUDATensor children public functions
	/// </summary>

	template<typename T>
	inline __device__ size_t CUDATensor1D<T>::X()
	{
		return threadIdx.x + blockIdx.x * blockDim.x;
	}

	template<typename T>
	inline __device__ size_t CUDATensor2D<T>::X()
	{
		return threadIdx.x + blockIdx.x * blockDim.x;
	}

	template<typename T>
	inline __device__ size_t CUDATensor2D<T>::Y()
	{
		return threadIdx.y + blockIdx.y * blockDim.y;
	}

	template<typename T>
	inline __device__ size_t CUDATensor3D<T>::X()
	{
		return threadIdx.x + blockIdx.x * blockDim.x;
	}
	template<typename T>
	inline __device__ size_t CUDATensor3D<T>::Y()
	{
		return threadIdx.y + blockIdx.y * blockDim.y;
	}
	template<typename T>
	inline __device__ size_t CUDATensor3D<T>::Z()
	{
		return threadIdx.z + blockIdx.z * blockDim.z;
	}

	template<typename T>
	__device__ T& CUDATensor1D<T>::At(size_t x)
	{
		return get_gpu()[x];
	}

	template<typename T>
	__device__ T CUDATensor1D<T>::At(size_t x) const
	{
		return get_gpu()[x];
	}

	template<typename T>
	__device__ T& CUDATensor1D<T>::At()
	{
		return get_gpu()[threadIdx.x + blockIdx.x * blockDim.x];
	}

	template<typename T>
	__device__ T CUDATensor1D<T>::At() const
	{
		return get_gpu()[threadIdx.x + blockIdx.x * blockDim.x];
	}

	template<typename T>
	__device__ T& CUDATensor1D<T>::Offset(size_t x)
	{
		return get_gpu()[threadIdx.x + blockIdx.x * blockDim.x + x];
	}

	template<typename T>
	__device__ T CUDATensor1D<T>::Offset(size_t x) const
	{
		return get_gpu()[threadIdx.x + blockIdx.x * blockDim.x + x];
	}

	template<typename T>
	__device__ bool CUDATensor1D<T>::in_bounds() const
	{
		return ((threadIdx.x + blockIdx.x * blockDim.x) < m_length);
	}

	template<typename T>
	__device__ bool CUDATensor1D<T>::offset_bounds(size_t x) const
	{
		return ((threadIdx.x + blockIdx.x * blockDim.x + x) < m_length);
	}

	template<typename T>
	__device__ T& CUDATensor2D<T>::At(size_t x, size_t y)
	{
		return get_gpu()[x + y * m_length];
	}

	template<typename T>
	__device__ T CUDATensor2D<T>::At(size_t x, size_t y) const
	{
		return get_gpu()[x + y * m_length];
	}

	template<typename T>
	__device__ T& CUDATensor2D<T>::At()
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x) +
			(threadIdx.y + blockIdx.y * blockDim.y) * m_length];
	}

	template<typename T>
	__device__ T CUDATensor2D<T>::At() const
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x) +
			(threadIdx.y + blockIdx.y * blockDim.y) * m_length];
	}

	template<typename T>
	__device__ T& CUDATensor2D<T>::Offset(size_t x, size_t y)
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x + x) +
			(threadIdx.y + blockIdx.y * blockDim.y + y) * m_length];
	}

	template<typename T>
	__device__ T CUDATensor2D<T>::Offset(size_t x, size_t y) const
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x + x) +
			(threadIdx.y + blockIdx.y * blockDim.y + y) * m_length];
	}

	template<typename T>
	__device__ bool CUDATensor2D<T>::in_bounds() const
	{
		return ((threadIdx.x + blockIdx.x * blockDim.x) < m_length) &&
			((threadIdx.y + blockIdx.y * blockDim.y) < m_width);
	}

	template<typename T>
	__device__ bool CUDATensor2D<T>::offset_bounds(size_t x, size_t y) const
	{
		return ((threadIdx.x + blockIdx.x * blockDim.x + x) < m_length) &&
			((threadIdx.y + blockIdx.y * blockDim.y + y) < m_width);
	}

	template<typename T>
	__device__ T& CUDATensor3D<T>::At(size_t x, size_t y)
	{
		return get_gpu()[x + y * m_length + z * m_length * m_width];
	}

	template<typename T>
	__device__ T CUDATensor3D<T>::At(size_t x, size_t y) const
	{
		return get_gpu()[x + y * m_length + z * m_length * m_width];
	}

	template<typename T>
	__device__ T& CUDATensor3D<T>::At()
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x) +
			(threadIdx.y + blockIdx.y * blockDim.y) * m_length +
			(threadIdx.z + blockIdx.z * blockDim.z) * m_length * m_width];
	}

	template<typename T>
	__device__ T CUDATensor3D<T>::At() const
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x) +
			(threadIdx.y + blockIdx.y * blockDim.y) * m_length +
			(threadIdx.z + blockIdx.z * blockDim.z) * m_length * m_width];
	}

	template<typename T>
	__device__ T& CUDATensor3D<T>::Offset(size_t x, size_t y, size_t z)
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x + x) +
			(threadIdx.y + blockIdx.y * blockDim.y + y) * m_length +
			(threadIdx.z + blockIdx.z * blockDim.z + z) * m_length * m_width];
	}

	template<typename T>
	__device__ T CUDATensor3D<T>::Offset(size_t x, size_t y, size_t z) const
	{
		return get_gpu()[(threadIdx.x + blockIdx.x * blockDim.x + x) +
			(threadIdx.y + blockIdx.y * blockDim.y + y) * m_length +
			(threadIdx.z + blockIdx.z * blockDim.z + z) * m_length * m_width];
	}

	template<typename T>
	__device__ bool CUDATensor3D<T>::in_bounds() const
	{
		return ((threadIdx.x + blockIdx.x * blockDim.x) < m_length) &&
			((threadIdx.y + blockIdx.y * blockDim.y) < m_width) &&
			((threadIdx.z + blockIdx.z * blockDim.z) < m_height);
	}

	template<typename T>
	__device__ bool CUDATensor3D<T>::offset_bounds(size_t x, size_t y, size_t z) const
	{
		return ((threadIdx.x + blockIdx.x * blockDim.x + x) < m_length) &&
			((threadIdx.y + blockIdx.y * blockDim.y + y) < m_width) &&
			((threadIdx.z + blockIdx.z * blockDim.z + z) < m_height);
	}

	/// Tensor class CUDATensor cast operators

	template<typename T, Mode device>
	Tensor<T, device>::operator CUDATensor1D<T>()
	{
		return CUDATensor1D<T>(*this);
	}

	template<typename T, Mode device>
	Tensor<T, device>::operator const CUDATensor1D<T>() const
	{
		return const CUDATensor1D<T>(*this);
	}

	template<typename T, Mode device>
	Tensor<T, device>::operator CUDATensor2D<T>()
	{
		return CUDATensor2D<T>(*this);
	}

	template<typename T, Mode device>
	Tensor<T, device>::operator const CUDATensor2D<T>() const
	{
		return const CUDATensor2D<T>(*this);
	}

	template<typename T, Mode device>
	Tensor<T, device>::operator CUDATensor3D<T>()
	{
		return CUDATensor3D<T>(*this);
	}

	template<typename T, Mode device>
	Tensor<T, device>::operator const CUDATensor3D<T>() const
	{
		return const CUDATensor3D<T>(*this);
	}

	/// Tensor class cuda specific functions

	template<typename T, Mode device>
	inline __host__ void Tensor<T, device>::allocate()
	{
		//Allocate the cpu memory on the gpu
		//this will have to be done every time the Tensor gets resized or deallocated

		//assert if trying to allocate already allocated memory FIX: deallocate before allocating
		#ifdef _TS_DEBUG
		assert("GPU memory was not deallocated before calling allocate" && isDeallocated());
		#endif

		CER(cudaMalloc(&(gpu_mem), size() * sizeof(T)));
		allocated = true;
	}

	template<typename T, Mode device>
	void Tensor<T, device>::deallocate()
	{
		//Deallocate the gpu memmorry

		//assert if memory is allready deallocated
		#ifdef _TS_DEBUG
		assert("GPU memory was already deallocated" && isAllocated());
		#endif

		CER(cudaFree(gpu_mem));
		allocated = false;
	}

	template<typename T, Mode device>
	bool Tensor<T, device>::isAllocated() const
	{
		return allocated;
	}

	template<typename T, Mode device>
	bool Tensor<T, device>::isDeallocated() const
	{
		return !allocated;
	}

	template<typename T, Mode device>
	inline void Tensor<T, device>::copyGPU()
	{
		//assert if memory is not allocated on gpu
		//this error might be thrown if you forget to allocate after a resize
		#ifdef _TS_DEBUG
		assert("Memory is not allocated on the gpu unable to copy" && isAllocated());
		#endif

		//copy source cpu memory to gpu memory
		CER(cudaMemcpy(gpu_mem, Data(), size() * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T, Mode device>
	void Tensor<T, device>::copyCPU()
	{
		//assert if memory is not allocated on gpu
		//this error might be thrown if you forget to allocate after a resize
		#ifdef _TS_DEBUG
		assert("Memory is not allocated on the gpu unable to copy" && isAllocated());
		#endif

		//copy source gpu memory to cpu memory
		CER(cudaMemcpy(Data(), gpu_mem, size() * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T, Mode device>
	inline void Tensor<T, device>::push()
	{
		allocate();
		copyGPU();
	}

	template<typename T, Mode device>
	void Tensor<T, device>::pull()
	{
		copyCPU();
		deallocate();
	}

	template<typename T, Mode device>
	__host__ __device__ T* Tensor<T, device>::getGPU()
	{
		return gpu_mem;
	}

	template<typename T, Mode device>
	__host__ __device__ const T* Tensor<T, device>::getGPU() const
	{
		return gpu_mem;
	}

	template<typename T, Mode device>
	void Tensor<T, device>::setTargetThreads(const unsigned short& value)
	{
		#ifdef _TS_DEBUG
		if (value > 1024)
			throw BadThreadTarget(value);
		#endif

		m_threads = value;
	}

	template<typename T, Mode device>
	short Tensor<T, device>::getTargetThreads() const
	{
		m_threads = value;
	}

	/// Tensor class kernel functions
	/// Ways of calculating kernel dimsions:
	///
	/// Cube:
	///		create a X*Y*Z cube where the cube is the best fit for the target threads
	///
	/// Plane:
	///		create a X*Y*1 plane where the plane is the best fit for the target threads
	///
	/// Line:
	///		create an X*1*1 line where the line is the best fit for the target threads
	///
	///
	/// NOTE: block layout is not affected by these options
	/// NOTE: lower dimension kernels can use higher dimension thread layouts

	template<typename T, Mode device>
	template<Mode layout, typename FT, typename RT, typename ...Args>
	Tensor<RT, device> Tensor<T, device>::Kernel1DR(CUDALayout<layout> layout, FT(kernel_p), Args && ...args)
	{
		#if 1
		#ifdef _TS_DEBUG
		assert("There can not be more than 1024 total threads in each block" && m_threads <= 1024);
		if (isDeallocated())
			std::cout << "warning: No gpu memory is allocated, illegal memory access may occur\n";
		#endif

		//create result Tensor: This will hold all of the return values
		Tensor<RT, device> result(Shape(), RT());

		result.allocate();

		//Get flattended version of Tensor so it can fit in the desired dimensions
		size_t dims = FlattenDims();

		//calculate block sizes from thread size
		auto threads = layout.apply(this, m_threads);
		unsigned int blocks = (unsigned int)(dims + std::get<0>(threads) - 1) / std::get<0>(threads);

		kernel_p << < blocks, { std::get<0>(threads), std::get<1>(threads) , std::get<2>(threads) } >> > (*this, result, args...);

		#ifdef _TS_DEBUG
		CER(cudaGetLastError());
		#endif

		CER(cudaDeviceSynchronize());

		result.pull();

		return result;
		#endif
	}

	template<typename T, Mode device>
	template<Mode layout, typename FT, typename ...Args>
	void Tensor<T, device>::Kernel1D(CUDALayout<layout> layout, FT(kernel_p), Args && ...args)
	{
		#ifdef _TS_DEBUG
		assert("There can not be more than 1024 total threads in each block" && m_threads <= 1024);
		if (isDeallocated())
			std::cout << "warning: No gpu memory is allocated, illegal memory access may occur\n";
		#endif

		//Get flattended version of Tensor so it can fit in the desired dimensions
		size_t dims = FlattenDims();

		//calculate block sizes from thread size
		std::tuple<unsigned int, unsigned int, unsigned int> threads = layout.apply(this, m_threads);
		unsigned int blocks = (unsigned int)(dims + std::get<0>(threads) - 1) / std::get<0>(threads);

		kernel_p << < blocks, { std::get<0>(threads), std::get<1>(threads), std::get<2>(threads) } >> > (*this, args...);

		#ifdef _TS_DEBUG
		CER(cudaGetLastError());
		#endif

		CER(cudaDeviceSynchronize());
	}

	template<typename T, Mode device>
	template<Mode layout, typename FT, typename RT, typename ...Args>
	Tensor<RT, device> Tensor<T, device>::Kernel2DR(CUDALayout<layout> layout, FT(kernel_p), Args && ...args)
	{
		#ifdef _TS_DEBUG
		assert("There can not be more than 1024 total threads in each block" && m_threads <= 1024);
		if (isDeallocated())
			std::cout << "warning: No gpu memory is allocated, illegal memory access may occur\n";
		#endif

		//create result Tensor: This will hold all of the return values
		Tensor<RT, device> result(Shape(), RT());
		result.allocate();

		//Get flattended version of Tensor so it can fit in the desired dimensions

		std::vector<size_t> dims = FlattenDims(2);

		//calculate block sizes from thread size

		auto threads = layout.apply(this, m_threads);
		dim3 blocks((uint32_t)std::ceil((dims[0] + std::get<0>(threads) - 1) / std::get<0>(threads)),
			(uint32_t)std::ceil((dims[1] + std::get<1>(threads) - 1) / std::get<1>(threads)));

		kernel_p << < blocks, { std::get<0>(threads), std::get<1>(threads) , std::get<2>(threads) } >> > (*this, result, args...);

		#ifdef _TS_DEBUG
		CER(cudaGetLastError());
		#endif

		CER(cudaDeviceSynchronize());

		result.pull();

		return result;
	}

	template<typename T, Mode device>
	template<Mode layout, typename FT, typename ...Args>
	void Tensor<T, device>::Kernel2D(CUDALayout<layout> layout, FT(kernel_p), Args && ...args)
	{
		#ifdef _TS_DEBUG
		assert("There can not be more than 1024 total threads in each block" && m_threads <= 1024);
		if (isDeallocated())
			std::cout << "warning: No gpu memory is allocated, illegal memory access may occur\n";
		#endif

		//Get flattended version of Tensor so it can fit in the desired dimensions
		std::vector<size_t> dims = FlattenDims(2);

		//calculate block sizes from thread size

		auto threads = layout.apply(this, m_threads);
		dim3 blocks((uint32_t)std::ceil((dims[1] + std::get<0>(threads) - 1) / std::get<0>(threads)),
			(uint32_t)std::ceil((dims[0] + std::get<1>(threads) - 1) / std::get<1>(threads)));

		kernel_p << < blocks, { std::get<0>(threads), std::get<1>(threads) , std::get<2>(threads) } >> > (*this, args...);

		#ifdef _TS_DEBUG
		CER(cudaGetLastError());
		#endif

		CER(cudaDeviceSynchronize());
	}

	template<typename T, Mode device>
	template<Mode layout, typename FT, typename RT, typename ...Args>
	Tensor<RT, device> Tensor<T, device>::Kernel3DR(CUDALayout<layout> layout, FT(kernel_p), Args && ...args)
	{
		#ifdef _TS_DEBUG
		assert("There can not be more than 1024 total threads in each block" && m_threads <= 1024);
		if (isDeallocated())
			std::cout << "warning: No gpu memory is allocated, illegal memory access may occur\n";
		#endif

		//create result Tensor: This will hold all of the return values
		Tensor<RT, device> result(Shape(), RT());

		result.allocate();

		//Get flattended version of Tensor so it can fit in the desired dimensions
		std::vector<size_t> dims = FlattenDims(3);

		//calculate block sizes from thread size

		auto threads = layout.apply(this, m_threads);
		dim3 blocks((uint32_t)std::ceil((dims[2] + std::get<0>(threads) - 1) / std::get<0>(threads)),
			(uint32_t)std::ceil((dims[1] + std::get<1>(threads) - 1) / std::get<1>(threads)),
			(uint32_t)std::ceil((dims[0] + std::get<2>(threads) - 1) / std::get<2>(threads)));

		kernel_p << < blocks, { std::get<0>(threads), std::get<1>(threads) , std::get<2>(threads) } >> > (*this, result, args...);

		#ifdef _TS_DEBUG
		CER(cudaGetLastError());
		#endif

		CER(cudaDeviceSynchronize());

		result.pull();

		return result;
	}

	template<typename T, Mode device>
	template<Mode layout, typename FT, typename ...Args>
	void Tensor<T, device>::Kernel3D(CUDALayout<layout> layout, FT(kernel_p), Args && ...args)
	{
		#ifdef _TS_DEBUG
		assert("There can not be more than 1024 total threads in each block" && m_threads <= 1024);
		if (isDeallocated())
			std::cout << "warning: No gpu memory is allocated, illegal memory access may occur\n";
		#endif

		//Get flattended version of Tensor so it can fit in the desired dimensions
		std::vector<size_t> dims = FlattenDims(3);

		//calculate block sizes from thread size

		auto threads = layout.apply(this, m_threads);
		dim3 blocks((uint32_t)std::ceil((dims[0] + std::get<0>(threads) - 1) / std::get<0>(threads)),
			(uint32_t)std::ceil((dims[1] + std::get<1>(threads) - 1) / std::get<1>(threads)),
			(uint32_t)std::ceil((dims[2] + std::get<2>(threads) - 1) / std::get<2>(threads)));

		kernel_p << < blocks, { std::get<0>(threads), std::get<1>(threads) , std::get<2>(threads) } >> > (*this, args...);

		#ifdef _TS_DEBUG
		CER(cudaGetLastError());
		#endif

		CER(cudaDeviceSynchronize());
	}

	void CUDAInitialize(int device)
	{
		/*
		* this is primarily used to initialize the cuda api. This oftens takes some time to load so this function makes it possible to have more control over when this pause will happen.
		*/
		cudaSetDevice(device);
		cudaDeviceSynchronize();
		devcount = device;
		cudaGetDeviceProperties(&props, devcount);

		#ifdef _TS_DEBUG
		CUDA_IS_INITIALIZED = true;
		#endif
	}

	#else
#pragma message("warning: cuda is not enabled, this header file should not be included.")
#endif
}