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

int main()
{
	//static auto arr = make_arr<3, 3, 3>();
	//StaticTensor<int, 3, arr> tensor;

	constexpr auto a = add<5, 2>();
	static auto arr = make_shape<3, 3, 3>;

	std::cout << typeid(arr).name() << '\n';

	make_tensor<int, 3, 3, 3> tensor;

	std::cout << product<size_t, 3, make_shape<3, 3, 3>>();

	std::cin.get();
}