#include <functional>
#include <iostream>
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

//using namespace TSlib;
//
//__kernel__ k3D(CUDATensor3D<char> tensor)
//{
//	tensor.At() += 2;
//	tensor.At() += 100;
//	tensor.At() *= 2;
//	tensor.At() += tensor.Offset(1);
//}
//

int main() {
	//CUDAInitialize();

	/*Tensor<int, Mode::CPU> tensor1({ 5, 5, 5 }, 3);
	Tensor<int, Mode::CPU> tensor2({ 5, 5, 5 }, 5);

	tensor2.push();*/

	//const Tensor<int>& tensor3 = tensor2;

	//std::cout << (tensor1 + tensor3);

	return 0;
}