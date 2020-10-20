#define PROFILING 2
#include <iostream>
#include "Tensor.h"
#include <Profiler.h>
#include <fstream>
#include <intrin.h>

using namespace Profiling;
using namespace TSlib;

__kernel__ kernel(CUDATensor3D<uint32_t> tensor, int val)
{
	if (tensor.in_bounds())
	{
		for (int i = 0; i < 10000; i++)
		{
			tensor.At() += val;
		}
		
		if (tensor.off_bounds(2))
		{
			for (int i = 0; i < 10000; i++)
			{
			
					tensor.Offset(2) += val;
			}
		}
	}
}


int main()
{

	Profiler::Get().BeginSession("CUDA_SESSION", "CUDA_PROFILE.json");
	Profiler::Get().margin = 50;
	
	TSlib::CUDAInitialize();

	Tensor<uint32_t> tensor({ 1024, 1024, 1024 }, 25);

	try
	{

		tensor.push();

		tensor.Kernel3D(Layout3D(1, 2, 0.5), kernel, 25);

		tensor.pull();

		std::cout << tensor;
	}
	catch (std::exception& e)
	{
		std::cout << e.what();
	}

	Profiler::Get().EndSession();
}
