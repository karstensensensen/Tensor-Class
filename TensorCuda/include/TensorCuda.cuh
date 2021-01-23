#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>
#include <tuple>

#define __kernel__ __global__ void
#define ROUND(x)

namespace TSlib
{
	namespace
	{
		static cudaDeviceProp props;
		static int devcount;
		#ifdef _TS_DEBUG
		bool CUDA_IS_INITIALIZED = false;

		#define CER(ans) { gpuAssert((ans), __FILE__, __LINE__); }
		inline void gpuAssert(cudaError_t code, const char* file, int line)
		{
			#ifdef _TS_DEBUG
			#if 0
			if (!CUDA_IS_INITIALIZED)
				std::cout << "WARNING: CUDA library has not been initialized. CUDA dependent functions are not guaranteed to work.\nUse the function TSlib::CUDAInitialize() to initialize the CUDA library\n";
			#endif
			#endif
			if (code != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s Code: %d \n%s %d\n", cudaGetErrorString(code), code, file, line);
				std::cin.get();
				exit(code);
			}
		}

		#else
		#define CER(ans) ans
		#endif
	}

	double_t round(double_t x, double_t place);
	void CUDAInitialize(int device = 0);

	template<typename T>
	class CUDATensor1D;

	template<typename T>
	class CUDATensor2D;

	template<typename T>
	class CUDATensor3D;

	template<typename T>

	/// <summary>
	/// Declaration of CUDATensor base
	/// </summary>

	class CTBase
	{
	protected:
		friend CUDATensor1D<T>;
		friend CUDATensor2D<T>;
		friend CUDATensor3D<T>;

		T* gpu_mem;
		size_t m_size = NULL;
		size_t* dim_arr;
		size_t dims = NULL;

		template<typename First>
		__device__ void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord) const;

		template<typename First, typename ... Args>
		__device__ void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining) const;

		CTBase(T* gpu_mem, size_t m_size, size_t* dim_arr, size_t dims);
		~CTBase();

	public:

		__device__ T* get_gpu();
		__device__ const T* get_gpu() const;

		__device__ size_t size() const;

		template<typename ... Args>
		__device__ T& Get(Args ... coords);

		__device__ T& operator[](size_t index);

		template<typename ... Args>
		__device__ T& operator()(Args ... args);
	};

	/// <summary>
	/// Declaration of CUDATensor1D
	/// </summary>

	template<typename T>
	class CUDATensor1D : public CTBase<T>
	{
		size_t m_length = NULL;

	public:

		__device__ size_t X();

		__device__ size_t get_length();

		template<Mode device>
		CUDATensor1D(Tensor<T, device>& tensor);
		template<Mode device>
		CUDATensor1D(const Tensor<T, device>& tensor);
		CUDATensor1D(const CUDATensor1D<T>& other);

		__device__ T& At(size_t x);
		__device__ T At(size_t x) const;

		__device__ T& At();
		__device__ T At() const;

		__device__ T& Offset(size_t x);
		__device__ T Offset(size_t x) const;

		__device__ bool in_bounds() const;

		__device__ bool offset_bounds(size_t x) const;
	};

	/// <summary>
	/// Declaration of CUDATensor2D
	/// </summary>

	template<typename T>
	class CUDATensor2D : public CTBase<T>
	{
		size_t m_length = NULL;
		size_t m_width = NULL;

	public:
		// = threadIdx.x + blockIdx.x * blockDim.x;
		__device__ size_t X();
		__device__ size_t Y();

		__device__ size_t get_length();
		__device__ size_t get_width();

		template<Mode device>
		CUDATensor2D(Tensor<T, device>& tensor);
		template<Mode device>
		CUDATensor2D(const Tensor<T, device>& tensor);
		CUDATensor2D(const CUDATensor2D<T>& other);

		__device__ T& At(size_t x, size_t y);
		__device__ T At(size_t x, size_t y) const;

		__device__ T& At();
		__device__ T At() const;

		__device__ T& Offset(size_t x, size_t y = 0);
		__device__ T Offset(size_t x, size_t y = 0) const;

		__device__ bool in_bounds() const;

		__device__ bool offset_bounds(size_t x, size_t y = 0) const;
	};

	/// <summary>
	/// Declaration of CUDATensor3D
	/// </summary>

	template<typename T>
	class CUDATensor3D : public CTBase<T>
	{
		size_t m_length = NULL;
		size_t m_width = NULL;
		size_t m_height = NULL;

	public:

		__device__ size_t X();
		__device__ size_t Y();
		__device__ size_t Z();

		__device__ size_t get_length();
		__device__ size_t get_width();
		__device__ size_t get_height();

		template<Mode device>
		CUDATensor3D(Tensor<T, device>& tensor);
		template<Mode device>
		CUDATensor3D(const Tensor<T, device>& tensor);
		CUDATensor3D(const CUDATensor3D<T>& other);

		__device__ T& At(size_t x, size_t y);
		__device__ T At(size_t x, size_t y) const;

		__device__ T& At();
		__device__ T At() const;

		__device__ T& Offset(size_t x, size_t y = 0, size_t z = 0);
		__device__ T Offset(size_t x, size_t y = 0, size_t z = 0) const;

		__device__ bool in_bounds() const;

		__device__ bool offset_bounds(size_t x, size_t y = 0, size_t z = 0) const;
	};

	/// <summary>
	/// This layout type is only avaliable for the 3D kernel. This layout type will create a cube with equal length sides.
	/// If the sides shouldnt be the same length it can be specified.
	/// </summary>

	template<Mode layout>
	class CUDALayout;

	template<>
	class CUDALayout<Mode::Cube>
	{
		double_t X_ratio;
		double_t Y_ratio;
		double_t Z_ratio;

		double_t get_cubed(unsigned int target_threads)
		{
			double_t NT = NULL;

			for (unsigned int i = 0; target_threads - i != 0; i += 32)
			{
				double_t cube = std::cbrt(target_threads - i);
				if (cube == std::floor(cube))
				{
					NT = cube;
					break;
				}
			}

			if (NT == NULL)
			{
				for (size_t i = 0; target_threads + i <= 1024; i += 32)
				{
					double_t cube = std::cbrt(target_threads + i);
					if (cube == std::floor(cube))
					{
						//this value is one iteration too high

						NT = std::sqrt(cube * 2);
						break;
					}
				}
			}

			#ifdef _TS_DEBUG
			assert(NT != NULL && "Value for NT was not found");
			#endif

			return NT;
		}

	public:
		CUDALayout(double_t X, double_t Y, double_t Z)
			:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
		{
			#ifdef _TS_DEBUG

			if (X_ratio * Y_ratio * Z_ratio != 1)
				throw BadShape("There must be an equal number of multiplications and divisions, when initializing PlaneLayout.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

			#endif
		}

		template<typename T, Mode device>
		std::tuple<unsigned int, unsigned int, unsigned int> apply(const Tensor<T, device>* tensor, unsigned int target_threads)
		{
			double_t threads_cubed = get_cubed(target_threads);

			double_t length = threads_cubed;
			double_t width = threads_cubed;
			double_t height = threads_cubed;

			length *= double_t(length) * X_ratio;
			#ifdef _TS_DEBUG
			if (round(length * double_t(length) * X_ratio, 1000) != std::floor(round(length * double_t(length) * X_ratio, 1000)))
			{
				throw BadValue("Length ratio does not divide cleanly into thread length", ExceptValue<double_t>("ratio", X_ratio), ExceptValue<double_t>("cubed threads", threads_cubed));
			}
			#endif

			#ifdef _TS_DEBUG
			if (round(width * double_t(width) * Y_ratio, 1000) != std::floor(round(width * double_t(width) * Y_ratio, 1000)))
			{
				throw BadValue("Width ratio does not divide cleanly into thread width", ExceptValue<double_t>("ratio", Y_ratio), ExceptValue<double_t>("cubed threads", threads_cubed));
			}
			#endif

			width *= double_t(width) * Y_ratio;

			#ifdef _TS_DEBUG
			if (round(height * double_t(height) * Z_ratio, 1000) != std::floor(round(height * double_t(height) * Z_ratio, 1000)))
			{
				throw BadValue("Height ratio does not divide cleanly into thread height", ExceptValue<double_t>("ratio", Z_ratio), ExceptValue<double_t>("cubed threads", threads_cubed));
			}
			#endif

			height *= double_t(height) * Z_ratio;

			return { unsigned int(length), unsigned int(width), unsigned int(height) };
		}
	};

	template<>
	class CUDALayout<Mode::Plane>
	{
		double_t X_ratio = NULL;
		double_t Y_ratio = NULL;
		double_t Z_ratio = NULL;

		double_t get_squared(size_t target_threads)
		{
			double_t NT = NULL;

			for (unsigned int i = 0; target_threads - i != 0; i += 32)
			{
				double_t square = std::sqrt(target_threads - i);
				if (square == std::floor(square))
				{
					NT = square;
					break;
				}
			}

			if (NT == NULL)
			{
				for (unsigned int i = 0; target_threads + i <= 1024; i += 32)
				{
					double_t square = std::sqrt(target_threads + i);
					if (square == std::floor(square))
					{
						NT = std::sqrt(square) * 2.0;
						break;
					}
				}
			}

			#ifdef _TS_DEBUG
			assert(NT != NULL && "Value for NT was not found");
			#endif

			return NT;
		}

	public:
		CUDALayout(double_t X, double_t Y, double_t Z)
			:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
		{
			#ifdef _TS_DEBUG

			if (X_ratio * Y_ratio * Z_ratio != 1)
				throw BadShape("There must be an equal number of multiplications and divisions, when initializing PlaneLayout.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

			#endif
		}

		template<typename T, Mode device>
		std::tuple<unsigned int, unsigned int, unsigned int> apply(const Tensor<T, device>* tensor, unsigned int target_threads)
		{
			#pragma warning(disable: 4244)

			double_t threads_squared = get_squared(target_threads);

			double_t length = threads_squared;
			double_t width = threads_squared;
			double_t height = 1;

			#ifdef _TS_DEBUG
			if (round(length * length * X_ratio, 1000.0) != std::floor(round(length * length * X_ratio, 1000.0)))
			{
				double_t ratio = X_ratio;
				BadValue("Length ratio does not divide cleanly into thread length", ExceptValue("ratio", ratio), ExceptValue("squared threads", threads_squared));
			}
			#endif

			length *= double_t(length) * X_ratio;

			#ifdef _TS_DEBUG
			if (round(width * width * Y_ratio, 1000.0) != std::floor(round(width * width * Y_ratio, 1000.0)))
			{
				double_t ratio = Y_ratio;
				BadValue("Length ratio does not divide cleanly into thread width", ExceptValue("ratio", ratio), ExceptValue("squared threads", threads_squared));
			}
			#endif

			width *= double_t(width) * Y_ratio;

			#ifdef _TS_DEBUG
			if (round(height * height * Z_ratio, 1000.0) != std::floor(round(height * height * Z_ratio, 1000.0)))
			{
				double_t ratio = Z_ratio;
				BadValue("Length ratio does not divide cleanly into thread height", ExceptValue("ratio", ratio), ExceptValue("squared threads", threads_squared));
			}
			#endif

			height *= double_t(height) * Z_ratio;

			return { unsigned int(length), unsigned int(width), unsigned int(height) };

			#pragma warning(default: 4244)
		}
	};

	template<>
	class CUDALayout<Mode::Line>
	{
		double_t X_ratio;
		double_t Y_ratio;
		double_t Z_ratio;

	public:
		CUDALayout(double_t X, double_t Y, double_t Z)
			:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
		{
			#ifdef _TS_DEBUG

			if (X_ratio * Y_ratio * Z_ratio != 1)
				throw BadShape("The total area must not be changed by the new ratios.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

			#endif
		}

		template<typename T, Mode device>
		std::tuple<unsigned int, unsigned int, unsigned int> apply(const Tensor<T, device>* tensor, unsigned int target_threads)
		{
			#pragma warning(disable: 4244)

			unsigned int length = target_threads;
			unsigned int width = 1;
			unsigned int height = 1;

			#ifdef _TS_DEBUG
			if (double_t(length) * X_ratio != std::floor(double_t(length) * X_ratio))
			{
				double_t ratio = X_ratio;
				BadValue("Length ratio does not divide cleanly into thread length", ExceptValue("ratio", ratio), ExceptValue("squared threads", target_threads));
			}
			#endif

			length *= unsigned int(double_t(length) * X_ratio);

			#ifdef _TS_DEBUG
			if (double_t(width) * Y_ratio != std::floor(double_t(width) * Y_ratio))
			{
				double_t ratio = Y_ratio;
				BadValue("Length ratio does not divide cleanly into thread width", ExceptValue("ratio", ratio), ExceptValue("squared threads", target_threads));
			}
			#endif

			width *= unsigned int(double_t(width) * Y_ratio);

			#ifdef _TS_DEBUG
			if (double_t(height) * Z_ratio != std::floor(double_t(height) * Z_ratio))
			{
				double_t ratio = Z_ratio;
				BadValue("Length ratio does not divide cleanly into thread height", ExceptValue("ratio", ratio), ExceptValue("squared threads", target_threads));
			}
			#endif

			height *= unsigned int(double_t(height) * Z_ratio);

			return { length, width, height };
			#pragma warning(default: 4244)
		}
	};

	inline const CUDALayout<Mode::Cube>		Layout3D(double_t X = 1, double_t Y = 1, double_t Z = 1)
	{
		return CUDALayout<Mode::Cube>(X, Y, Z);
	}

	inline const CUDALayout<Mode::Plane>	Layout2D(double_t X = 1, double_t Y = 1, double_t Z = 1)
	{
		return CUDALayout<Mode::Plane>(X, Y, Z);
	}

	inline const CUDALayout<Mode::Line>		Layout1D(double_t X = 1, double_t Y = 1, double_t Z = 1)
	{
		return CUDALayout<Mode::Line>(X, Y, Z);
	}
}