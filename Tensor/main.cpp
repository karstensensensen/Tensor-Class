////#define PROFILING 2
//#include <iostream>
//#include "Tensor.h"
//#include <Profiler.h>
//#include <fstream>
//#include <intrin.h>
//
//using namespace Profiling;
//using namespace TSlib;
//
////__kernel__ kernel(CUDATensor3D<uint32_t> tensor, int val)
////{
////	if (tensor.in_bounds())
////	{
////		for (int i = 0; i < 10000; i++)
////		{
////			tensor.At() += val;
////		}
////		
////		if (tensor.offset_bounds(2))
////		{
////			for (int i = 0; i < 10000; i++)
////			{
////			
////					tensor.Offset(2) += val;
////			}
////		}
////	}
////}
////
////size_t generator(const size_t& i)
////{
////	return i % 10;
////}
////
////size_t generator2(const size_t& i)
////{
////	return i % 10;
////}
////
////__kernel__ k3D(CUDATensor3D<char> tensor)
////{
////	tensor.At() += 2;
////	tensor.At() += rand()%100;
////	tensor.At() *= 2;
////	tensor.At() += tensor.Offset(1);
////}
////
//int main()
//{
//	Profiler::Get().BeginSession("CUDA_SESSION", "CUDA_PROFILE.json");
//	Profiler::Get().margin = 50;
//	
//	//CUDAInitialize();
//
//	try
//	{
//		Tensor<char> tensor({ 10, 10, 10}, 8);
//		std::cout << (Tensor<int>)tensor;
//		//tensor.Kernel3D(Layout3D(), k3D);
//	}
//	catch (std::exception& e)
//	{
//		std::cout << e.what();
//	}
//
//	Profiler::Get().EndSession();
//}
