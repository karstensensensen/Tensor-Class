#include <iostream>
#include "Tensor.h"

using namespace TSlib;

int main(int argc, const char * argv[]) {
	std::cout << "Tensor:\n";
	
	TSlib::Tensor<int> tensor({1024, 1024, 1024}, 1);
	TSlib::Tensor<int> tensor2({1, 1, 1}, 1);
	
	tensor2.MultiResize(tensor.DimSizes(), 3);
	
	std::cout << (tensor+tensor2).sum();
	return 0;
}
