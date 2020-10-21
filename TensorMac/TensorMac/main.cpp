#include <iostream>
#include "Tensor.h"

using namespace TSlib;

int main(int argc, const char * argv[]) {
	std::cout << "Tensor:\n";
	
	TSlib::Tensor<int> tensor({5, 5, 5}, 1);
	TSlib::Tensor<int> tensor2({1, 1, 1}, 1);
	
	tensor2.MultiResize({5, 5, 5});
	
	std::cout << tensor2+tensor;
	return 0;
}
