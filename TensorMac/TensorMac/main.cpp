#include <iostream>
#include "Tensor.h"

int main(int argc, const char * argv[]) {
	std::cout << "Tensor:\n";
	TSlib::Tensor<int> tensor({5, 5, 5}, 25);
	TSlib::Tensor<int> tensor2({3, 3, 3}, 25);
	tensor2.Reshape({5, 5, 5});
	std::cout << tensor + tensor2;
	return 0;
}
