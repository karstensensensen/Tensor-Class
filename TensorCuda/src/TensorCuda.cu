#include "Tensor.h"

#ifdef _TS_DEBUG
void TSlib::CUDebug::cudaErr(cudaError_t code, const char* file, int line)
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
void TSlib::CUDAInitialize(int device)
{
	cudaSetDevice(device);
	cudaDeviceSynchronize();
	CUDebug::devcount = device;
	cudaGetDeviceProperties(&CUDebug::props, CUDebug::devcount);

	#ifdef _TS_DEBUG
	CUDebug::CUDA_IS_INITIALIZED = true;
	#endif
}
