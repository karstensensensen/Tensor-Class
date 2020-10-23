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

size_t generator(const std::vector<size_t>& index, const size_t& i)
{
	size_t res = rand();
	return res;
}

int main()
{

	Profiler::Get().BeginSession("CUDA_SESSION", "CUDA_PROFILE.json");
	Profiler::Get().margin = 50;
	
	CUDAInitialize();

	Tensor<size_t> tensor({ 5, 5, 5 }, generator);

	try
	{
		TensorSlice<size_t, Mode::GPU> slice = tensor.Slice({ 1, -1 }, {1, -1});
		Tensor<size_t> tensor2 = slice;
		std::cout << slice << "\n\n\n";
		std::cout << tensor2;
	}
	catch (std::exception& e)
	{
		std::cout << e.what();
	}

	Profiler::Get().EndSession();
}
