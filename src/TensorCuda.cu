#include "TensorCudaBones.cuh"

namespace TSlib
{
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
}