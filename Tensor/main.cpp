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

size_t generator(const size_t& i)
{
	return i % 10;
}

size_t generator2(const size_t& i)
{
	return i % 10;
}

int main()
{

	Profiler::Get().BeginSession("CUDA_SESSION", "CUDA_PROFILE.json");
	Profiler::Get().margin = 50;
	
	CUDAInitialize();

	Tensor<size_t> tensor({ 5, 5, 5 }, generator);
	Tensor<size_t> tensor2({ 5, 5, 5 }, generator);

	try
	{
		auto tensor_result = tensor == tensor2;
		std::cout << tensor_result;
		
	}
	catch (std::exception& e)
	{
		std::cout << e.what();
	}

	Profiler::Get().EndSession();
}
