//#include "Tensor.h"
//
//using TSlib::Tensor;
//using TSlib::Mode;
//
//template<typename T1, typename T2, TSlib::Tools::enable_if_tensor<T1, T2> = 0>
//int A()
//{
//	std::cout << "UNSIGNED\n";
//	return 25;
//}
//
//int main()
//{
//	TSlib::CUDAInitialize();
//
//	Tensor<int> tensor({ 5, 5 }, []() {return rand() % 128; });
//	auto slice = tensor.Slice({ TSlib::All, {1, 3} });
//	auto shape = slice.Shape();
//
//	std::cout << tensor << "\n\n\n";
//
//	std::cout << slice << "\n\n\n";
//
//	std::cout << slice.Compute([](int& new_elem, const int& elem) {
//		new_elem = std::max(new_elem, elem);
//		}, 1);
//}