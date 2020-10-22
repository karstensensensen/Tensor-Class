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
		
		if (tensor.offset_bounds(2))
		{
			for (int i = 0; i < 10000; i++)
			{
			
					tensor.Offset(2) += val;
			}
		}
	}
}

uint32_t generator(const std::vector<size_t>& index)
{
	return index[0] * index[1] * index[2];
}

int main()
{

	Profiler::Get().BeginSession("CUDA_SESSION", "CUDA_PROFILE.json");
	Profiler::Get().margin = 50;
	
	TSlib::CUDAInitialize();

	Tensor<uint32_t> tensor({ 3, 3, 3 }, generator);

	try
	{
		std::cout << tensor;
	}
	catch (std::exception& e)
	{
		std::cout << e.what();
	}

	Profiler::Get().EndSession();
}
