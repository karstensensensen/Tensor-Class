//
//  main.cpp
//  TensorMac
//
//  Created by Simon Levin on 21/10/2020.
//  Copyright © 2020 Simon Levin. All rights reserved.
//

#include <iostream>
#include "Tensor.h"

int main(int argc, const char * argv[]) {
	std::cout << "Tensor:\n";
	TSlib::Tensor<int> tensor({5, 5, 5}, 25);
	std::cout << tensor + 5;
	return 0;
}
