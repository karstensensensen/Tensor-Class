#include "TSlib.h"

namespace TSlib
{
	#ifdef _TS_DEBUG
	void CUDebug::cudaErr(cudaError_t code, const char* file, int line)
	{
		if (!CUDA_IS_INITIALIZED)
			std::cout << "WARNING: CUDA library has not been initialized. CUDA dependent functions are not guaranteed to work.\nUse the function TSlib::CUDAInitialize() to initialize the CUDA library\n";

		if (code != cudaSuccess)
		{
			fprintf(stderr, "CUDAerror: %s Code: %d \n%s %d\n", cudaGetErrorString(code), code, file, line);
			std::cin.get();
			exit(code);
		}
	}
	#endif

	// this function lets you initialize the CUDA library and set the target device to peform computations on
	// device 0 is default
	void CUDAInitialize(int device)
	{
		cudaSetDevice(device);
		cudaDeviceSynchronize();
		CUDebug::devcount = device;
		cudaGetDeviceProperties(&CUDebug::props, CUDebug::devcount);

		#ifdef _TS_DEBUG
		CUDebug::CUDA_IS_INITIALIZED = true;
		#endif
	}

	CUDALayout<Mode::Cube>::CUDALayout(double X, double Y, double Z)
		:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
	{
		#ifdef _TS_DEBUG

		if (X_ratio * Y_ratio * Z_ratio != 1)
			throw BadShape("The total area must not be changed by the new ratios.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

		#endif
	}

	double CUDALayout<Mode::Cube>::GetCubed(unsigned int target_threads)
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

	CUDALayout<Mode::Plane>::CUDALayout(double X, double Y, double Z)
		:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
	{
		#ifdef _TS_DEBUG

		if (X_ratio * Y_ratio * Z_ratio != 1)
			throw BadShape("The total area must not be changed by the new ratios.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

		#endif
	}

	double CUDALayout<Mode::Plane>::GetSquared(size_t target_threads)
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

	CUDALayout<Mode::Line>::CUDALayout(double X, double Y, double Z)
		:X_ratio(X), Y_ratio(Y), Z_ratio(Z)
	{
		#ifdef _TS_DEBUG

		if (X_ratio * Y_ratio * Z_ratio != 1)
			throw BadShape("The total area must not be changed by the new ratios.\nOtherwise the thread target wouldn't be hit: ", { X_ratio, Y_ratio, Z_ratio });

		#endif
	}

}