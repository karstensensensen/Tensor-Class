#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>
#include <tuple>

#define __kernel__ __global__ void

namespace TSlib
{
	namespace CUDebug
	{
		static cudaDeviceProp props;
		static int devcount;

		#ifdef _TS_DEBUG
		static bool CUDA_IS_INITIALIZED = false;

		#define CER(ans) { TSlib::CUDebug::cudaErr((ans), __FILE__, __LINE__); }
		void cudaErr(cudaError_t code, const char* file, int line);

		#else
		#define CER(ans) ans
		#endif
	}

	double round(double x, double place);
	void CUDAInitialize(int device = 0);

	template<typename T>
	class CUDATensor1D;

	template<typename T>
	class CUDATensor2D;

	template<typename T>
	class CUDATensor3D;

	template<typename T>
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

		__device__ T* GetGPU();
		__device__ const T* GetGPU() const;

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

		__device__ size_t GetLength();

		template<Device device>
		CUDATensor1D(Tensor<T, device>& tensor);
		template<Device device>
		CUDATensor1D(const Tensor<T, device>& tensor);
		CUDATensor1D(const CUDATensor1D<T>& other);

		__device__ T& At(size_t x);
		__device__ T At(size_t x) const;

		__device__ T& At();
		__device__ T At() const;

		__device__ T& Offset(size_t x);
		__device__ T Offset(size_t x) const;

		__device__ bool InBounds() const;

		__device__ bool OffsetBounds(size_t x) const;
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

		__device__ size_t GetLength();
		__device__ size_t GetWidth();

		template<Device device>
		CUDATensor2D(Tensor<T, device>& tensor);
		template<Device device>
		CUDATensor2D(const Tensor<T, device>& tensor);
		CUDATensor2D(const CUDATensor2D<T>& other);

		__device__ T& At(size_t x, size_t y);
		__device__ T At(size_t x, size_t y) const;

		__device__ T& At();
		__device__ T At() const;

		__device__ T& Offset(size_t x, size_t y = 0);
		__device__ T Offset(size_t x, size_t y = 0) const;

		__device__ bool InBounds() const;

		__device__ bool OffsetBounds(size_t x, size_t y = 0) const;
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

		__device__ size_t GetLength();
		__device__ size_t GetWidth();
		__device__ size_t GetHeight();

		template<Device device>
		CUDATensor3D(Tensor<T, device>& tensor);
		template<Device device>
		CUDATensor3D(const Tensor<T, device>& tensor);
		CUDATensor3D(const CUDATensor3D<T>& other);

		__device__ T& At(size_t x, size_t y);
		__device__ T At(size_t x, size_t y) const;

		__device__ T& At();
		__device__ T At() const;

		__device__ T& Offset(size_t x, size_t y = 0, size_t z = 0);
		__device__ T Offset(size_t x, size_t y = 0, size_t z = 0) const;

		__device__ bool InBounds() const;

		__device__ bool OffsetBounds(size_t x, size_t y = 0, size_t z = 0) const;
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
		double X_ratio;
		double Y_ratio;
		double Z_ratio;

		double GetCubed(unsigned int target_threads)
		{
			double NT = NULL;

			for (unsigned int i = 0; target_threads - i != 0; i += 32)
			{
				double cube = std::cbrt(target_threads - i);
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
					double cube = std::cbrt(target_threads + i);
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
		CUDALayout(double X, double Y, double Z)
			:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
		{
			#ifdef _TS_DEBUG

			if (X_ratio * Y_ratio * Z_ratio != 1)
				throw BadShape("There must be an equal number of multiplications and divisions, when initializing PlaneLayout.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

			#endif
		}

		template<typename T, Device device>
		std::tuple<unsigned int, unsigned int, unsigned int> Apply(const Tensor<T, device>* tensor, unsigned int target_threads)
		{
			double threads_cubed = GetCubed(target_threads);

			double length = threads_cubed;
			double width = threads_cubed;
			double height = threads_cubed;

			length *= double(length) * X_ratio;
			#ifdef _TS_DEBUG
			if (round(length * double(length) * X_ratio, 1000) != std::floor(round(length * double(length) * X_ratio, 1000)))
			{
				throw BadValue("Length ratio does not divide cleanly into thread length", ExceptValue<double>("ratio", X_ratio), ExceptValue<double>("cubed threads", threads_cubed));
			}
			#endif

			#ifdef _TS_DEBUG
			if (round(width * double(width) * Y_ratio, 1000) != std::floor(round(width * double(width) * Y_ratio, 1000)))
			{
				throw BadValue("Width ratio does not divide cleanly into thread width", ExceptValue<double>("ratio", Y_ratio), ExceptValue<double>("cubed threads", threads_cubed));
			}
			#endif

			width *= double(width) * Y_ratio;

			#ifdef _TS_DEBUG
			if (round(height * double(height) * Z_ratio, 1000) != std::floor(round(height * double(height) * Z_ratio, 1000)))
			{
				throw BadValue("Height ratio does not divide cleanly into thread height", ExceptValue<double>("ratio", Z_ratio), ExceptValue<double>("cubed threads", threads_cubed));
			}
			#endif

			height *= double(height) * Z_ratio;

			return { unsigned int(length), unsigned int(width), unsigned int(height) };
		}
	};

	template<>
	class CUDALayout<Mode::Plane>
	{
		double X_ratio = NULL;
		double Y_ratio = NULL;
		double Z_ratio = NULL;

		double GetSquared(size_t target_threads)
		{
			double NT = NULL;

			for (unsigned int i = 0; target_threads - i != 0; i += 32)
			{
				double square = std::sqrt(target_threads - i);
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
					double square = std::sqrt(target_threads + i);
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
		CUDALayout(double X, double Y, double Z)
			:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
		{
			#ifdef _TS_DEBUG

			if (X_ratio * Y_ratio * Z_ratio != 1)
				throw BadShape("There must be an equal number of multiplications and divisions, when initializing PlaneLayout.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

			#endif
		}

		template<typename T, Device device>
		std::tuple<unsigned int, unsigned int, unsigned int> Apply(const Tensor<T, device>* tensor, unsigned int target_threads)
		{
			#pragma warning(disable: 4244)

			double threads_squared = GetSquared(target_threads);

			double length = threads_squared;
			double width = threads_squared;
			double height = 1;

			#ifdef _TS_DEBUG
			if (round(length * length * X_ratio, 1000.0) != std::floor(round(length * length * X_ratio, 1000.0)))
			{
				double ratio = X_ratio;
				BadValue("Length ratio does not divide cleanly into thread length", ExceptValue("ratio", ratio), ExceptValue("squared threads", threads_squared));
			}
			#endif

			length *= double(length) * X_ratio;

			#ifdef _TS_DEBUG
			if (round(width * width * Y_ratio, 1000.0) != std::floor(round(width * width * Y_ratio, 1000.0)))
			{
				double ratio = Y_ratio;
				BadValue("Length ratio does not divide cleanly into thread width", ExceptValue("ratio", ratio), ExceptValue("squared threads", threads_squared));
			}
			#endif

			width *= double(width) * Y_ratio;

			#ifdef _TS_DEBUG
			if (round(height * height * Z_ratio, 1000.0) != std::floor(round(height * height * Z_ratio, 1000.0)))
			{
				double ratio = Z_ratio;
				BadValue("Length ratio does not divide cleanly into thread height", ExceptValue("ratio", ratio), ExceptValue("squared threads", threads_squared));
			}
			#endif

			height *= double(height) * Z_ratio;

			return { unsigned int(length), unsigned int(width), unsigned int(height) };

			#pragma warning(default: 4244)
		}
	};

	template<>
	class CUDALayout<Mode::Line>
	{
		double X_ratio;
		double Y_ratio;
		double Z_ratio;

	public:
		CUDALayout(double X, double Y, double Z)
			:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
		{
			#ifdef _TS_DEBUG

			if (X_ratio * Y_ratio * Z_ratio != 1)
				throw BadShape("The total area must not be changed by the new ratios.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

			#endif
		}

		template<typename T, Device device>
		std::tuple<unsigned int, unsigned int, unsigned int> Apply(const Tensor<T, device>* tensor, unsigned int target_threads)
		{
			#pragma warning(disable: 4244)

			unsigned int length = target_threads;
			unsigned int width = 1;
			unsigned int height = 1;

			#ifdef _TS_DEBUG
			if (double(length) * X_ratio != std::floor(double(length) * X_ratio))
			{
				double ratio = X_ratio;
				BadValue("Length ratio does not divide cleanly into thread length", ExceptValue("ratio", ratio), ExceptValue("squared threads", target_threads));
			}
			#endif

			length *= unsigned int(double(length) * X_ratio);

			#ifdef _TS_DEBUG
			if (double(width) * Y_ratio != std::floor(double(width) * Y_ratio))
			{
				double ratio = Y_ratio;
				BadValue("Length ratio does not divide cleanly into thread width", ExceptValue("ratio", ratio), ExceptValue("squared threads", target_threads));
			}
			#endif

			width *= unsigned int(double(width) * Y_ratio);

			#ifdef _TS_DEBUG
			if (double(height) * Z_ratio != std::floor(double(height) * Z_ratio))
			{
				double ratio = Z_ratio;
				BadValue("Length ratio does not divide cleanly into thread height", ExceptValue("ratio", ratio), ExceptValue("squared threads", target_threads));
			}
			#endif

			height *= unsigned int(double(height) * Z_ratio);

			return { length, width, height };
			#pragma warning(default: 4244)
		}
	};

	inline const CUDALayout<Mode::Cube>		Layout3D(double X = 1, double Y = 1, double Z = 1)
	{
		return CUDALayout<Mode::Cube>(X, Y, Z);
	}

	inline const CUDALayout<Mode::Plane>	Layout2D(double X = 1, double Y = 1, double Z = 1)
	{
		return CUDALayout<Mode::Plane>(X, Y, Z);
	}

	inline const CUDALayout<Mode::Line>		Layout1D(double X = 1, double Y = 1, double Z = 1)
	{
		return CUDALayout<Mode::Line>(X, Y, Z);
	}
}