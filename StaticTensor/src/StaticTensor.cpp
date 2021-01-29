#include <iostream>
#include "StaticTensor.h"

template<typename T, size_t N1, size_t N2>
constexpr std::array<T, N1 + N2> append_arr(std::array<T, N1> arr1, std::array<T, N2> arr2)
{
	std::array<T, N1 + N2> arr;

	arr.fill(arr1);
	arr.fill(arr2, N1);

	return arr;
}


template<int a, int b>
constexpr int add()
{
	return a + b;
}

using namespace TSlib;

template<typename T, size_t dimensions, const std::array<size_t, dimensions>& var_shape>
std::ostream& operator << (std::ostream& stream, const StaticTensor<T, dimensions, var_shape>& tensor)
{
	return tensor.printable(stream);
}

int main()
{
	//static auto arr = make_arr<3, 3, 3>();
	//StaticTensor<int, 3, arr> tensor;

	constexpr auto a = add<5, 2>();
	static auto arr = make_shape<3, 3, 3>;

	make_tensor<int, 256, 256> tensor('Z');

	tensor.Compute([](int& elem) {elem = rand(); });
	
	make_tensor<size_t, 256, 256> tensor2 = tensor;
	tensor2.Compute([](size_t& elem) {elem = std::sqrt(elem) + std::pow(elem, 4); });

	std::cout << tensor;

	std::cin.get();
}