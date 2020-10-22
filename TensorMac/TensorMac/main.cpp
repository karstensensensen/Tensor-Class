#include <iostream>
#include "Tensor.h"

using namespace TSlib;

int main(int argc, const char * argv[]) {
	std::cout << "Tensor:\n";
	
	TSlib::Tensor<int> tensor({128, 128, 128}, 1);
	TSlib::Tensor<int> tensor2({1, 1, 1}, 1);
	
	tensor2.Resize(tensor.Shape(), tensor2(0, 0, 0) + 2);
	
	tensor += tensor2;
	
	tensor.Resize({25, 3, 2});
	
	std::cout << tensor;
	
	return 0;
}
